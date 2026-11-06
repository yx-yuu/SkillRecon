"""Source-located credential inputs in operational JSON-RPC Shell commands."""

from __future__ import annotations

import json
import re

from tree_sitter import Language, Parser
import tree_sitter_bash

from skillrecon.contract.parser import parse_document
from skillrecon.core.types import CapabilityEvent, ResourceUse
from skillrecon.loader.inline import check_binding_signal


def recover_rpc_credentials(events, resources, doc_contents, doc_paths):
    """Check literal params[0] credentials in executable curl command blocks.

    Dynamic Shell payloads remain outside this literal model. An unresolved
    expression does not establish a failed credential-origin requirement.
    """
    changes = []
    parser = Parser(Language(tree_sitter_bash.language()))
    existing = {(e.file_path, e.line, e.api_call) for e in events}

    def walk(node):
        yield node
        for child in node.named_children:
            yield from walk(child)

    for doc_id, content in doc_contents.items():
        if 'jsonrpc' not in content or 'token:' not in content:
            continue
        blocks = parse_document(doc_id, content)
        for index, block in enumerate(blocks):
            if block.block_type != 'code_block' or block.language_hint not in {'bash','sh','shell','zsh'}:
                continue
            if re.search(r'\b(?:example|schema|reference|sample)\b|示例|规范|例如', block.heading_context, re.I):
                continue
            preceding = blocks[index-1] if index else None
            binding = check_binding_signal(preceding, block)
            if binding is None:
                continue
            tree = parser.parse(block.content.encode())
            for command in walk(tree.root_node):
                if command.type != 'command':
                    continue
                name = command.child_by_field_name('name')
                if name is None or name.text != b'curl':
                    continue
                arguments = command.children_by_field_name('argument')
                for arg_index, argument in enumerate(arguments[:-1]):
                    if argument.text not in {b'-d',b'--data',b'--data-raw',b'--data-binary'}:
                        continue
                    payload_node = arguments[arg_index+1]
                    if payload_node.type != 'raw_string':
                        continue
                    try:
                        payload = json.loads(payload_node.text.decode()[1:-1])
                    except json.JSONDecodeError:
                        continue  # Invalid or non-JSON payloads are not registered RPC calls.
                    if (not isinstance(payload, dict) or payload.get('jsonrpc') != '2.0'
                            or not isinstance(payload.get('method'), str)
                            or not isinstance(payload.get('params'), list) or not payload['params']
                            or not isinstance(payload['params'][0], str)
                            or not payload['params'][0].startswith('token:')):
                        continue
                    credential = payload['params'][0][len('token:'):]
                    if not credential:
                        continue
                    line = content[:block.start_offset].count('\n') + 2 + command.start_point.row
                    relative = doc_paths[doc_id]
                    key = (relative,line,'JSON-RPC credential argument')
                    if key in existing:
                        continue
                    existing.add(key)
                    # A documented substitution marker is a user input site;
                    # a fixed literal credential does not meet that requirement.
                    result = 'true' if re.fullmatch(r'<YOUR_[A-Z_]+>', credential) else 'false'
                    event = CapabilityEvent(
                        event_id=f'e{len(events)}',unit_id=f'doc::{doc_id}',capability='api_key_use',
                        api_call=key[2],file_path=relative,line=line,location=f'{relative}:{line}',
                        language='bash',tier='registered_source',
                        arguments=[payload['method'],'params[0]'],detail='Operational JSON-RPC credential use',
                    )
                    events.append(event)
                    resources.append(ResourceUse(
                        resource_id=f'credential-protocol-{event.event_id}',unit_id=event.unit_id,
                        event_id=event.event_id,resource_type='credential_protocol',value='jsonrpc',
                        location=event.location,resolved=True,origin_kind='source_binding',
                        origin_hint='Parsed jsonrpc=2.0 request with a params[0] credential',
                    ))
                    evidence = {'model':'json_rpc_credential_source_v1','kind':'credential_source',
                                'key':'configured_input','result':result,
                                'locations':[event.location],'method':payload['method'],
                                'parameter_slot':'params[0]','binding_instruction':binding,
                                'source_block_id':block.block_id,
                                'explanation':'Literal Shell JSON payload; credential value omitted from artifact'}
                    resources.append(ResourceUse(
                        resource_id=f'credential-{event.event_id}',unit_id=event.unit_id,
                        event_id=event.event_id,resource_type='control_condition',
                        value='credential_source:configured_input',location=event.location,
                        resolved=True,origin_kind='static_condition',origin_hint=json.dumps(evidence,ensure_ascii=False),
                    ))
                    changes.append(evidence)
    return changes
