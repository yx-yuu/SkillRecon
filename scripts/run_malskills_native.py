#!/usr/bin/env python3
"""Execute native MalSkills and record its model and analysis stages."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True, type=Path)
    args, native_args = parser.parse_known_args()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text('', encoding='utf-8')
    lock = threading.Lock()

    def record(value):
        with lock, args.trace.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')

    import malskills.llm_runtime as runtime
    original_invoke = runtime.invoke_structured_json
    original_post = runtime._post_json
    original_run = subprocess.run

    def post(endpoint, payload, headers, *, timeout_sec):
        started = time.perf_counter()
        row = {'stage':'http', 'model':payload.get('model')}
        try:
            result = original_post(endpoint,payload,headers,timeout_sec=timeout_sec)
            row.update(success=True,response_model=result.get('model'),usage=result.get('usage'))
            return result
        except Exception as error:
            cause = error.__cause__
            row.update(success=False,error_type=type(error).__name__,http_status=getattr(cause,'code',None))
            if getattr(cause,'code',None) is not None:
                detail = cause.read().decode('utf-8',errors='replace')
                for value in headers.values():
                    detail = detail.replace(value,'[redacted]').replace(value.removeprefix('Bearer '),'[redacted]')
                row['error_detail'] = detail
            raise
        finally:
            record({**row,'seconds':time.perf_counter()-started})

    def invoke(**kwargs):
        config = kwargs.get('config') or runtime.build_llm_runtime_config()
        started = time.perf_counter()
        try:
            result = original_invoke(**kwargs)
            if result is None:
                raise RuntimeError(f'Native MalSkills {config.stage} returned no structured model response')
            record({'stage':'llm','native_stage':config.stage,'model':config.model,
                    'success':True,'seconds':time.perf_counter()-started,'response':result})
            return result
        except Exception as error:
            record({'stage':'llm','native_stage':config.stage,'model':config.model,
                    'success':False,'seconds':time.perf_counter()-started,'error_type':type(error).__name__})
            raise

    def run(command,*positional,**kwargs):
        is_yasa = isinstance(command,(list,tuple)) and len(command)>1 and str(command[1])=='dist/main.js'
        if not is_yasa:
            return original_run(command,*positional,**kwargs)
        started = time.perf_counter()
        language = command[command.index('--language')+1]
        try:
            result = original_run(command,*positional,**kwargs)
            sarif = Path(command[command.index('--report')+1])/'report.sarif'
            saved = args.trace.parent/f'yasa-{language}.sarif'
            if sarif.is_file():
                shutil.copy2(sarif,saved)
            record({'stage':'yasa','language':language,'success':result.returncode==0,
                    'returncode':result.returncode,'seconds':time.perf_counter()-started,
                    'sarif':str(saved) if sarif.is_file() else None})
            if result.returncode:
                raise RuntimeError(f'Native YASA failed with exit {result.returncode}: {result.stderr}')
            return result
        except subprocess.TimeoutExpired:
            record({'stage':'yasa','language':language,'success':False,'error_type':'TimeoutExpired',
                    'seconds':time.perf_counter()-started})
            raise RuntimeError('Native YASA timed out')

    runtime._post_json = post
    runtime.invoke_structured_json = invoke
    subprocess.run = run
    from malskills.sdg.yasa import YasaAdapter
    original_extract = YasaAdapter.extract

    def extract(self, skill_root, artifacts):
        targets = self._build_targets(artifacts)
        planned = self._planned_languages(targets)
        present = sorted({target.language for target in targets})
        available = self.available()
        record({'stage':'yasa_coverage','available':available,'present_languages':present,
                'planned_languages':planned,'unsupported_languages':sorted(set(present)-set(planned))})
        if planned and not available:
            raise RuntimeError('Native YASA engine is unavailable for applicable source files')
        return original_extract(self,skill_root,artifacts)

    YasaAdapter.extract = extract
    from malskills.cli import main as native_main
    return native_main(native_args)


if __name__ == '__main__':
    sys.exit(main())
