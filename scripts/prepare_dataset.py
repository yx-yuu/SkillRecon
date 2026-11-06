#!/usr/bin/env python3
"""Prepare versioned skill sources from the dataset catalog."""
import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,help='Copy packages from a local owner/skill collection')
    parser.add_argument('--skills-file',type=Path,help='Prepare only the listed identities')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.catalog.read_text().splitlines() if line.strip()]
    if args.skills_file:
        selected=set(args.skills_file.read_text().splitlines())
        rows=[row for row in rows if row['skill_id'] in selected]
    if args.limit is not None:rows=rows[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    archives={}
    with tempfile.TemporaryDirectory(prefix='skillrecon-sources-') as temp:
        temp=Path(temp)
        for row in rows:
            destination=args.output/row['skill_id']
            if args.source_root:
                shutil.copytree(args.source_root/row['skill_id'],destination)
            elif 'repository' in row:
                url=row['download_url']
                if url not in archives:
                    bundle=temp/str(len(archives));bundle.mkdir()
                    archive=bundle/'repository.tar.gz'
                    urllib.request.urlretrieve(url,archive)
                    with tarfile.open(archive) as tar:tar.extractall(bundle/'tree',filter='data')
                    archives[url]=next((bundle/'tree').iterdir())
                shutil.copytree(archives[url]/row['source_path'],destination)
            else:
                with tempfile.TemporaryDirectory(dir=temp) as package_temp:
                    archive=Path(package_temp)/'package.zip'
                    urllib.request.urlretrieve(row['download_url'],archive)
                    destination.mkdir(parents=True)
                    with zipfile.ZipFile(archive) as package:package.extractall(destination)
            print(row['skill_id'],flush=True)


if __name__=='__main__':main()
