# Tool catalog

The 41 JSON files describe tools for routing and evaluation. The filename stem
is the tool identifier used by the routers and expected-tool datasets.

| Field | Meaning |
|---|---|
| `summary` | Short semantic description indexed by every ChromaDB router. |
| `description` | Detailed tool context indexed by CDBv3 and supplied to the LLM router. |
| `keywords` | Extra vocabulary indexed by CDBv3. |

Add a `Tools/<name>.json` file with these descriptive fields to extend the
catalog. Rebuild the router to load changes. V3 refreshes cached embeddings
when descriptive text changes.

Descriptions preserve the original robot-domain text so routing comparisons
remain reproducible. Execution settings have been removed from the catalog.
