"""
openapi_utils.py

Utilities for parsing OpenAPI specifications, caching, extracting endpoints,
building execution sequences (DAG), and generating placeholder payloads.
"""
import json
import os
from typing import Any, Dict, List, Optional
from functools import lru_cache
import networkx as nx
import yaml
from openapi_parser import resolve_schema  # assumes existing parser

CACHE_DIR = os.path.expanduser("~/.openapi_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def parse_openapi_with_cache(spec_path: str) -> Dict[str, Any]:
    """
    Load an OpenAPI spec (YAML or JSON) and cache the parsed dict to avoid re-parsing.
    """
    cache_file = os.path.join(CACHE_DIR, os.path.basename(spec_path) + ".json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    # read source
    with open(spec_path) as f:
        if spec_path.endswith(('.yaml', '.yml')):
            spec = yaml.safe_load(f)
        else:
            spec = json.load(f)

    # resolve all $ref, allOf, oneOf, anyOf
    spec = resolve_schema(spec)

    # write cache
    with open(cache_file, 'w') as f:
        json.dump(spec, f, indent=2)

    return spec


def get_endpoints(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all (method, path) endpoints and their metadata from the OpenAPI spec.
    Returns a list of dicts: {path, method, operationId, parameters, requestBodySchema}.
    """
    endpoints: List[Dict[str, Any]] = []
    for path, methods in spec.get('paths', {}).items():
        for method, op in methods.items():
            if method.lower() not in ('get', 'post', 'put', 'delete', 'patch', 'options', 'head'):
                continue
            endpoints.append({
                'path': path,
                'method': method.lower(),
                'operationId': op.get('operationId'),
                'parameters': op.get('parameters', []),
                'requestBody': op.get('requestBody')
            })
    return endpoints


def build_execution_graph(apis: List[Dict[str, Any]]) -> List[str]:
    """
    Given a list of API endpoints, build a DAG of dependencies and return
a topologically sorted list of "<method> <path>" strings.
    Simple heuristic: POST/create operations come before GET/list; DELETE last.
    """
    # build graph
    g = nx.DiGraph()
    nodes = [f"{api['method']} {api['path']}" for api in apis]
    g.add_nodes_from(nodes)

    # add heuristic edges
    for api in apis:
        name = f"{api['method']} {api['path']}"
        # POST -> GET on same resource
        if api['method'] == 'post':
            for other in apis:
                if other['path'].startswith(api['path']) and other['method'] in ('get', 'put', 'delete'):
                    g.add_edge(name, f"{other['method']} {other['path']}")
        # DELETE should run after create/get
        if api['method'] == 'delete':
            for other in apis:
                if other['path'] == api['path'] and other['method'] in ('get', 'post', 'put'):
                    g.add_edge(f"{other['method']} {other['path']}", name)

    # cycle check
    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        raise ValueError(f"Cycle detected in API graph: {cycles}")

    # return topo order
    return list(nx.topological_sort(g))


def generate_payload_from_schema(schema: Dict[str, Any], placeholder: bool = False) -> Dict[str, Any]:
    """
    Recursively build a dict matching the schema. If placeholder=True,
    fill leaf values with placeholders (e.g. "<id>").
    """
    def _gen(schema_node: Dict[str, Any]) -> Any:
        if 'type' in schema_node:
            t = schema_node['type']
            if t == 'object':
                props = schema_node.get('properties', {})
                obj: Dict[str, Any] = {}
                for k, subschema in props.items():
                    obj[k] = _gen(subschema)
                return obj
            if t == 'array':
                item_schema = schema_node.get('items', {})
                return [_gen(item_schema)]
            # primitives
            if placeholder:
                return f"<{t}>"
            # basic example values
            if t == 'string':
                return schema_node.get('example', 'string')
            if t == 'integer':
                return schema_node.get('example', 0)
            if t == 'number':
                return schema_node.get('example', 0.0)
            if t == 'boolean':
                return schema_node.get('example', True)
        # fallback
        return None

    content = schema.get('content', {})
    # assume application/json
    media = content.get('application/json', {})
    schema_def = media.get('schema', schema)
    return _gen(schema_def)
