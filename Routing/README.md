# How routing works

This document describes selection logic and the Python interfaces. Installation,
available mechanisms, and single-prompt commands are in the
[root README](../README.md).

## Shared data and runtime

Tool identifiers are the filename stems of `Tools/*.json`. The descriptive
fields are documented in the [catalog schema](../Tools/README.md).
`build_router(config)` loads the catalog once; rebuild the router after changing
it. Relative catalog and cache paths resolve from the project root, independently
of the current working directory.

The ChromaDB routers create a uniquely named, in-memory collection per instance,
with one vector per indexed tool. A custom embedding function sends document and
query text to Ollama's embeddings endpoint. Collections use the default squared
L2 distance: smaller distances indicate closer vectors. Distances are neither
probabilities nor confidence scores.

By default each search retrieves all indexed tools in distance order.
`routing.chromadb_n_results` can cap retrieval before the router applies its
selection rules. V1 and V2 embed tool summaries at every startup; V3 caches tool
embeddings on disk. Query embeddings are computed live.

## CDBv1: fixed threshold

`router_CDBv1.py` indexes each tool's `summary` only.

1. Embed the complete prompt as one query.
2. Retrieve the nearest tools.
3. Return every retrieved tool with `distance <= distance_threshold`.

The CLI reads `routing.distance_threshold`, currently `0.325`. The Python
`find_candidates` method has its own default of `1.5`; callers using it directly
must pass the configured threshold to reproduce CLI behavior.

For illustrative distances `battery=0.20`, `wind_speed=0.30`, `land=0.40`, a
threshold of `0.325` selects `battery` and `wind_speed`.

One threshold applies to every prompt. A low value can return an empty selection
even when a tool is needed; a high value admits unrelated tools. Embedding a
compound request as one query can hide secondary actions.

## CDBv2: relative threshold

`router_CDBv2.py` inherits V1's summary indexing and single-query retrieval.
It replaces the fixed threshold with:

```text
cutoff = nearest_distance * (1 + similarity_margin)
```

The default `similarity_margin` is `0.60`. The nearest result always passes;
additional tools pass when their distance is at most `nearest_distance * 1.6`.
For a nearest distance of `0.30`, the cutoff is `0.48`.

With a nonempty catalog and successful retrieval, V2 always selects at least one
tool, including for unrelated prompts. The relative band adapts to each query
but can admit many extras. The `distance_threshold` argument is ignored.
`similarity_margin` can be changed in the constructor; it is not read from config.

## CDBv3: clause searches and aggregation

`router_CDBv3.py` indexes one richer document per tool:

```text
Tool: <filename stem>
Summary: <summary>
Description: <description>
Keywords: <keywords joined by commas>
```

### Embedding cache

Tool embeddings are stored in `Routing/.cache/CDBv3_embeddings.json`. Cache
entries depend on the embedding model name and full indexed text. New or changed
documents are embedded again; changing the model name invalidates their entries.
The in-memory collection is rebuilt even when embeddings come from the cache.
The cache does not include the endpoint or model contents in its key.

Override the location or disable caching through the `routing` section:

```json
{
  "routing": {
    "CDBv3_embedding_cache": "Routing/.cache/custom_embeddings.json"
  }
}
```

Set this key to `null` to disable the file cache.

### Query generation

`split_prompt` normalizes whitespace and includes the full prompt. It then adds
pieces split on commas, semicolons, `and`, and `then`, and action fragments to
the left of internal `if`, `when`, `before`, and `after` markers. Added fragments
must contain at least four characters. Queries are deduplicated without regard
to case and limited to eight.

For example:

```text
check the battery and land if wind is high
```

produces:

```text
check the battery and land if wind is high
check the battery
land if wind is high
land
```

The condition to the right of a marker is not added as a separate query.
`or`, `unless`, and `while` are not split points.

### Per-query selection and scoring

Each query keeps its nearest tool plus up to two additional tools within 20%
of the nearest distance. Every query's nearest tool is a guaranteed selection.

For each retained tool, the router records its smallest distance, the number of
queries that retained it (`hits`), and the index of its first matching query.
It ranks tools by:

```text
score = best_distance - 0.06 * ln(1 + hits)
sort key = (score, first_query, best_distance)
```

Lower scores rank first. Repeated matches give a small bonus. The returned
`distance` remains the raw best distance, rather than this adjusted score.

### Final selection

Let `winners` be the unique nearest tools from all queries. Default limits are:

```text
min_results = max(1, min(4, number_of_queries), number_of_winners)
max_results = min(8, max(min_results, number_of_winners + 3))
```

The router starts with the winners, then walks the ranked tools to reach the
minimum without applying a gap check. This loop always processes its first
entry before checking the minimum. It then adds remaining tools while below the
maximum and while `score <= best_score + 0.28`. Results are sorted by the same
ranking key.

For example, with four queries and three winners, the minimum is four and the
maximum is six. A fourth tool can be added just to reach the minimum; further
extras must pass the score gap. The minimum is a target and cannot create tools
absent from the retained matches.

The constructor accepts `score_gap` (default `0.28`) and `max_results` (default
`8`). `build_router` uses those defaults; only the cache path and retrieval cap
are configurable through the shared config. `distance_threshold` is ignored.
A custom maximum below the number of winners does not discard winners.

Clause searches can recover secondary actions, but counting queries toward the
minimum also encourages extra tools. A condition can still be missed when it
never becomes a useful standalone query.

## LLM: catalog selection

`router_LLM.py` loads all tools without vector search. `context.json` defines the
role, task, rules, requested output schema, and tool fields. By default each tool
is sent with its name, `summary`, and `description`.

The system prompt asks for the smallest sufficient set, including sensors needed
to evaluate conditions and actions explicitly requested by the user. It requests:

```json
{"tools": ["battery", "land"]}
```

The `LLMClient` in `router_LLM.py` sends the system prompt and user request to the configured
Ollama chat endpoint. It strips `<think>` blocks and retries once on a failed or
empty response. Parsing accepts an object with a `tools` list or a top-level
list, and can extract a JSON object from surrounding text or Markdown fences.
Names are trimmed and deduplicated; invalid names and non-string entries are
flagged. A nonempty malformed routing response is not retried by the parser.

`find_candidates` raises `RuntimeError` on request or parsing failure, including
unknown tool names. A valid `{"tools": []}` returns an empty selection. Successful
candidates have `distance: 0.0`, since there is no vector distance.

The CLI helper returns `selected_tools`, `raw_response`, `model`, and
`parse_error`. It retains unknown names with an error for the caller to inspect;
this differs from the stricter `find_candidates` API.

Every request sends the full catalog. Latency and selection quality depend on
the model, service, and prompt; outputs can vary between runs.

## Configuration and Python API

| Config key | Effect |
|---|---|
| `routing.scripts_dir` | Catalog directory for all routers. |
| `routing.distance_threshold` | V1 filtering threshold used by CLI helpers. |
| `routing.chromadb_n_results` | Retrieval cap for V1/V2/V3; `null` retrieves all. |
| `routing.CDBv3_embedding_cache` | V3 cache path; `null` disables caching. |
| `llm.embedding_model`, `llm.ollama_embed_url` | Embedding model and endpoint for V1/V2/V3. |
| `llm.routing_model`, `llm.ollama_url` | Model and chat endpoint for LLM selection. |

Each module exposes `build_router(config)`,
`router.find_candidates(prompt, distance_threshold)`, and
`router.get_description(tool_name)`. Only V1 uses the threshold argument.
Candidates have this shape (values are illustrative):

```json
[{"worker_name": "battery", "summary": "monitor battery level...", "distance": 0.2}]
```

V1/V2 and LLM return summaries; V3 returns its full indexed document in the
`summary` field. `get_description` returns that same stored text, or `null` for
an unknown name. An empty V1 selection means nothing passed the threshold;
it does not prove no tool is needed.

Every module also has `route_prompt_from_config(prompt, config_path)` for a
single prompt. Lower-level `route_prompt` signatures differ: V1/V2/V3 accept a
router instance, while LLM accepts an LLM client and routing context.

Benchmarking, CSV analysis, and plotting are documented in the
[benchmark guide](../README-bench.md).
