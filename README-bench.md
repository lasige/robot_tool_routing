# Benchmark, analysis, and plotting

Run the commands below from the repository root. See the
[main README](README.md#setup) for environment and model setup.

## Record benchmark results

`bench.py` records router selections and timings for supplied JSON prompt files.
It builds each router once and reuses it across prompts and runs. The three paper
datasets are in [`prompts/`](prompts/). The benchmark calculates no plots,
precision, accuracy, or other quality scores.

```bash
python3 bench.py CDBv3 LLM --prompt-dir prompts/ --config config.ollama-lan.example.json
python3 bench.py CDBv3 --prompts prompts/type_i.json
```

A prompt file is a list of strings or objects with `prompt` and optional
`expected_tools`. Labels are preserved for comparison by `analyser.py`:

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

## Analyse CSV results

The root-level `analyser.py` uses only the Python standard library. It reads one
or more CSV files or directories recursively, including the metadata-prefixed
format from `bench.py`. It deduplicates overlapping input paths, retains every
distinct run, and groups by router and prompt type.
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
parsing errors score zero on all quality metrics for labeled rows. CSVs
without `has_expected` are treated as fully labeled, so an empty `expected`
cell in those files means no tools were expected.

Router and prompt type are inferred from CSV metadata and file/directory names.
Data without a recognized type is grouped as `untyped`. No Ollama service or
router dependency is needed for analysis.

## Generate paper plots

The root-level `plotter.py` uses the paper figures' colors, hatching,
layout, and router comparison groups. It accepts the analyser's JSON output or
reads CSV files/directories directly:

```bash
python3 -m pip install -r requirements-plots.txt
python3 plotter.py analysis.json --output-dir plots/ --timing-scale log
python3 plotter.py output/<timestamp>/ --output-dir plots/
```

The datasets, CSV directories, analysis groups, and plot labels use **Type i**,
**Type ii**, and **Type iii**.

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
error handling can change the figures.
