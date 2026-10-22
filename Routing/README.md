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

## Optional benchmark

`bench.py` records router selections and timings for supplied JSON prompt files.
It builds each router once and reuses it across prompts and runs. The three paper
datasets are in [`../prompts/`](../prompts/). The benchmark calculates no plots,
precision, accuracy, or other quality scores.

```bash
python3 bench.py CDBv3 LLM --prompt-dir prompts/ --config config.ollama-lan.example.json
python3 bench.py CDBv3 --prompts prompts/type_i.json
```

A prompt file is a list of strings or objects with `prompt` and optional
`expected_tools`. Labels are preserved for comparison outside this repository:

```json
[{"prompt": "monitor the battery", "expected_tools": ["battery"]}]
```

`--runs N` repeats every selected router, `--limit N` limits prompts per file,
`--out DIR` chooses an output directory, and `--no-save` prints each result
without writing files. `--prompt-dir DIR` reads `type_i.json`, `type_ii.json`,
and `type_iii.json`, with `--types i iii` selecting types. `--config` defaults to the repository root's
`config.json`, as it does for the router commands.

Each run writes a CSV under
`output/<timestamp>/<router>/<prompt-set>/run_XX.csv`. A metadata
block records the configuration, model, input path, and run duration, followed
by a blank line and the result table:

| Column | Recorded value |
|---|---|
| `prompt_index`, `prompt` | One-based index and original prompt text. |
| `expected` | Supplied expected tool names, comma-separated. |
| `has_expected` | Distinguishes supplied labels (including an explicit empty list) from omitted labels. |
| `returned` | Selected tool names, comma-separated. |
| `distances` | ChromaDB distances; empty for LLM routing. |
| `latency_ms` | Elapsed routing time per prompt, including query embeddings and any model retries. |
| `raw_response` | Model response for LLM routing. |
| `parse_error` | Request, parsing, or routing error; empty on success. |

Per-prompt timing excludes router construction and initial catalog embedding.
A failed request records its error and elapsed time, then the benchmark continues.

## Optional CSV analysis

The root-level `analyser.py` uses only the Python standard library. It reads one
or more CSV files or directories recursively, including the metadata-prefixed
format from `bench.py` and historical evaluation CSVs. It deduplicates overlapping
input paths, retains every distinct run, and groups by router and prompt type.
Plain CSV tables are also supported when a parent directory identifies the router.

```bash
python3 analyser.py output/<timestamp>/ --output analysis.json
```

The output JSON includes the parsed per-prompt records and summaries:

| Metric | Definition |
|---|---|
| Accuracy (A) | Returned tool set exactly equals the expected set. |
| Completeness (C) | Every expected tool is included in the returned set. |
| Precision (P) | Fraction of returned tools that belong to the expected set. |
| Composite score (L) | `0.25*A + 0.75*(5*C*P)/(4*P+C)`; the fractional term is zero when its denominator is zero. |
| Timing | Mean, median, p95, and population standard deviation of per-prompt latency, in milliseconds. |

Quality metrics are averaged over labeled prompts within each CSV run, then
equally over runs for that router/type. Their standard deviations are across
run means. L is calculated from the averaged A/C/P values. Timing statistics
pool timed prompts across runs, including failed requests. The p95 uses linear
interpolation. Row and label counts include repeated runs.

Unlabeled rows contribute to timing and error counts but not quality scores.
An explicit empty expected list is a valid label: returning no tools gives full
credit, while returning extras gives zero accuracy and precision. Request or
parsing errors score zero on all quality metrics for labeled rows. Legacy CSVs
without `has_expected` are treated as fully labeled, so an empty `expected`
cell in those files means no tools were expected.

Router and prompt type are inferred from CSV metadata and file/directory names.
Data without a recognized type is grouped as `untyped`. No Ollama service or
router dependency is needed for analysis.

## Optional paper plots

The root-level `plotter.py` uses the earlier paper figures' colors, hatching,
layout, and router comparison groups. It accepts the analyser's JSON output or
reads CSV files/directories directly:

```bash
python3 -m pip install -r requirements-plots.txt
python3 plotter.py analysis.json --output-dir plots/ --timing-scale log
python3 plotter.py output/<timestamp>/ --output-dir plots/
```

The datasets, CSV directories, analysis groups, and plot labels use **Type i**,
**Type ii**, and **Type iii**. Historical numeric Types 1, 3, and 4 are recognized
as i, ii, and iii when reading older CSVs or analysis JSON. The discarded legacy
Type 2 has no bundled inputs and is excluded from plots; `--all-types` can include
it as **Legacy Type 2** when supplied in historical CSVs.

When available, the two comparison groups are CDBv1/CDBv2/CDBv3 and CDBv3/LLM.
Other available routers form an additional group. Use `--routers CDBv3 LLM` to
choose a comparison explicitly. Each group produces:

- `accuracy_completeness_<routers>.png`: stacked correct/incorrect percentages,
  using the mean-per-run metrics.
- `precision_<routers>.png`: per-prompt precision boxplots across labeled runs.
- `timings_<routers>.png`: per-prompt latency boxplots, in seconds, on a logarithmic
  y-axis matching Figure 5 of the paper, including timed errors.

Unavailable metrics are skipped, so unlabeled CSVs can still produce timing
figures. Use `--format pdf` or `--format svg` for vector output, `--dpi 300` for
raster resolution, and `--timing-ymax 20` to reproduce a fixed 20-second timing
axis when appropriate. The default logarithmic timing axis scales to the observed
positive latencies. Zero latencies cannot appear on a logarithmic axis and are
omitted with a message; the CSV and analysis data retain them. Use
`--timing-scale linear` for a linear axis that includes zero latencies.
Untyped data appears separately as **All prompts**. Different input data or
error handling can change the figures; historical result CSVs are not bundled.
