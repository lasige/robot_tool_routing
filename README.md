# Robot Tool Routing

This work has been accepted at **CAIESA 2026 in Barcelona**.

Select tools from a natural-language prompt using one of four routing mechanisms:

| Router | Mechanism |
|---|---|
| CDBv1 | Tool-summary embeddings with a fixed distance threshold. |
| CDBv2 | Tool-summary embeddings with a threshold relative to the nearest match. |
| CDBv3 | Full-prompt and clause searches over summaries, descriptions, and keywords. |
| LLM | A language model selects from the complete tool catalog. |

The catalog contains 41 tools. Routers select tool names; they do not execute
robot actions.

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Configure a running Ollama service in `config.json`. The defaults use localhost,
`mxbai-embed-large` for embeddings, and `gpt-oss:120b-cloud` for LLM routing.
The embedding model must be available on the service; the cloud model requires Ollama cloud sign-in.

## Run a prompt

Choose a router explicitly:

```bash
python3 Routing/router_CDBv1.py "monitor the battery and land when it drops below 25%"
python3 Routing/router_CDBv2.py "monitor the battery and land when it drops below 25%"
python3 Routing/router_CDBv3.py "monitor the battery and land when it drops below 25%"
python3 Routing/router_LLM.py "monitor the battery and land when it drops below 25%"
```

Each command prints a JSON result containing `selected_tools` and router-specific
metadata. LLM results also include `parse_error`; check this field before using
the selection. Diagnostic logs may precede the JSON.

Every router defaults to `config.json` in the repository root, regardless of the
current working directory. Use `--config PATH` to select another configuration,
for example:

```bash
python3 Routing/router_CDBv3.py "monitor the battery" --config config.ollama-lan.example.json
```

The LLM command also accepts `--model MODEL` and `--context PATH`.

See [how the routing algorithms work](Routing/README.md) for implementation
details and configuration, and [the tool catalog schema](Tools/README.md) to
add or edit tools.

## Record, analyse, and plot results

`bench.py` records selected tools, optional expected labels, and per-prompt
timings to CSV. It does not calculate quality scores or create plots.

```bash
python3 bench.py CDBv3 LLM --prompt-dir prompts/ --config config.ollama-lan.example.json
python3 analyser.py output/<timestamp>/ --output analysis/analysis.json
```

The labeled datasets are in [prompts/](prompts/):
[Type i](prompts/type_i.json) (41 prompts),
[Type ii](prompts/type_ii.json) (52), and
[Type iii](prompts/type_iii.json) (69).

Run one file with
`--prompts prompts/type_i.json`, or all three with `--prompt-dir prompts/`.

The analyser prints quality and timing summaries and saves data for plotting.
To generate the paper figures, install the optional plotting dependencies:

```bash
python3 -m pip install -r requirements-plots.txt
python3 plotter.py analysis/analysis.json --output-dir plots/ --timing-scale log
```

`--timing-scale log` uses the paper's logarithmic latency axis and is the default.
Use `--timing-scale linear` for a linear axis.

Paper plots use Types i, ii, and iii. See the
[benchmark and analysis details](Routing/README.md#optional-benchmark) for the
input format, metrics, and plotting options.

## Citation

Provisional BibTeX entry using the repository title. The DOI is not yet available.

```bibtex
@inproceedings{sousa2026robottoolrouting,
  author    = {Sousa, Miguel and Pinto, Luis R and Carvalho, Joao},
  title     = {Robot Tool Routing},
  booktitle = {CAIESA 2026},
  address   = {Barcelona},
  year      = {2026},
  note      = {Accepted; DOI pending}
}
```
