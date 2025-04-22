# Improved prompt template for constructing an API execution graph
prompt_template = '''
"""
description: |
  Guide the LLM to construct a dependency-aware execution graph from an OpenAPI spec.
  The graph must include realistic payloads, validation steps, and logical sequencing
  (e.g., POST before GET/PUT/DELETE, and ensure updates are verified before deletion).

prompt: |
  You are an expert API testing assistant. Your task is to:
    1. Parse the provided OpenAPI spec (YAML/JSON).
    2. Identify all operations (paths, methods) and their dependencies.
       - POST operations must precede GET/PUT/DELETE for the same resource.
       - PUT (update) must include a verification step before any DELETE.
    3. Generate realistic example payloads for POST and PUT, resolving $ref, enums, and nested schemas.
    4. Suggest parallel execution only when operations are truly independent (no shared IDs or state).
    5. Include explicit "verify" nodes after each state-changing call (POST/PUT/DELETE) to check resource integrity.
    6. Represent the graph in JSON with nodes and edges, including:
       - node_id: unique identifier
       - operation: e.g., "POST /pets"
       - payload: realistic JSON example or placeholder
       - description: brief purpose of the node
       - depends_on: list of node_ids
    7. Provide a clear chain-of-thought reasoning (up to 10 lines) explaining how you determined dependencies.

output_structure: |
  {
    "graph": {
      "nodes": [
        {
          "node_id": "createPet",
          "operation": "POST /pets",
          "payload": { /* realistic example, e.g., {\"name\": \"Fido\", \"type\": \"dog\"} */ },
          "description": "Create a new pet",
          "depends_on": []
        },
        {
          "node_id": "verifyCreatePet",
          "operation": "GET /pets/{petId}",
          "payload": {},
          "description": "Verify pet creation",
          "depends_on": ["createPet"]
        },
        {
          "node_id": "updatePet",
          "operation": "PUT /pets/{petId}",
          "payload": { /* realistic updated fields */ },
          "description": "Update pet details",
          "depends_on": ["verifyCreatePet"]
        },
        {
          "node_id": "verifyUpdatePet",
          "operation": "GET /pets/{petId}",
          "payload": {},
          "description": "Verify pet update",
          "depends_on": ["updatePet"]
        },
        {
          "node_id": "deletePet",
          "operation": "DELETE /pets/{petId}",
          "payload": {},
          "description": "Delete the pet",
          "depends_on": ["verifyUpdatePet"]
        }
      ],
      "edges": [
        { "from": "createPet", "to": "verifyCreatePet" },
        { "from": "verifyCreatePet", "to": "updatePet" },
        { "from": "updatePet", "to": "verifyUpdatePet" },
        { "from": "verifyUpdatePet", "to": "deletePet" }
      ]
    },
    "chain_of_thought": [
      "Identify all unique operations and group by resource",
      "Ensure POST operations come before any GET/PUT/DELETE for the same resource",
      "Add verification GET after each state-changing call",
      "Link update and its verification sequentially, preventing parallel DELETE/PUT",
      "Generate realistic payloads by resolving schema references and enums"
    ]
  }
"""
'''
