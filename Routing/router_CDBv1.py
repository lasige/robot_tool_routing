"""
router_CDBv1.py - ChromaDB semantic router

Receives a text request and returns a list of matching worker candidates.
Each candidate is {"worker_name": str, "summary": str, "distance": float}.

Returns all candidates within the distance threshold (no hard cap on count).
The router only selects tools; it does not execute them.
"""

import argparse
import os
import json
import sys
import uuid
import chromadb
import requests
from pathlib import Path

SINGLE_ROBOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SINGLE_ROBOT_DIR / "config.json"

if str(SINGLE_ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(SINGLE_ROBOT_DIR))

from logger import log
from project_paths import resolve_from_project


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Router:
    """
    Populates a ChromaDB collection with worker descriptions
    and searches by semantic similarity.
    """

    def __init__(
        self,
        scripts_dir: str,
        ollama_embed_url: str,
        embedding_model: str,
        n_results: int | None = None,
    ):
        self.scripts_dir = resolve_from_project(scripts_dir)
        self.ollama_embed_url = ollama_embed_url
        self.embedding_model = embedding_model
        if n_results is not None and n_results <= 0:
            raise ValueError("n_results must be positive, or None to query all tools")
        self.n_results = n_results

        # in-memory client — rebuilt fresh on every boot
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=f"workers_{uuid.uuid4().hex}",
            embedding_function=self._build_embedding_function()
        )

        # local cache: worker_name → description (returned alongside query matches)
        self._summaries: dict[str, str] = {}

        self._populate()

    def _build_embedding_function(self):
        """Build a ChromaDB-compatible embedding function backed by Ollama."""
        ollama_url = self.ollama_embed_url
        model = self.embedding_model

        class OllamaEmbeddingFunction:
            def _embed(self, texts: list[str]) -> list[list[float]]:
                embeddings = []
                for text in texts:
                    response = requests.post(
                        ollama_url,
                        json={"model": model, "prompt": text},
                        timeout=30
                    )
                    response.raise_for_status()
                    embeddings.append(response.json()["embedding"])
                return embeddings

            def __call__(self, input: list[str]) -> list[list[float]]:
                return self._embed(input)

            def embed_documents(self, input: list[str]) -> list[list[float]]:
                return self._embed(input)

            def embed_query(self, input: list[str]) -> list[list[float]]:
                return self._embed(input)

        return OllamaEmbeddingFunction()

    def _populate(self):
        """Read all .json files from the scripts dir and add them to the collection."""
        worker_files = [f for f in os.listdir(self.scripts_dir) if f.endswith(".json")]

        if not worker_files:
            log("ROUTER", "no workers found", dir=self.scripts_dir)
            return

        ids, documents, metadatas = [], [], []

        for filename in worker_files:
            worker_name = filename.replace(".json", "")
            filepath = os.path.join(self.scripts_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                worker_config = json.load(f)

            summary = worker_config.get("summary", "")
            if not summary:
                log("ROUTER", "worker has no summary, skipping", worker=worker_name)
                continue

            self._summaries[worker_name] = summary
            ids.append(worker_name)
            documents.append(summary)
            metadatas.append({"worker_name": worker_name})

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        log("ROUTER", f"{len(ids)} workers loaded into ChromaDB")

    def _query_n_results(self) -> int:
        """Return how many nearest neighbors ChromaDB should retrieve."""
        if self.n_results is None:
            return len(self._summaries)
        return min(self.n_results, len(self._summaries))

    def find_candidates(self, user_request: str,
                        distance_threshold: float = 1.5) -> list[dict]:
        """
        Find workers semantically closest to the user request.

        Returns a list of dicts (may be empty):
            [{"worker_name": "battery", "summary": "...", "distance": 0.42}, ...]

        Filters by distance_threshold only — no hard cap on count.
        An empty list means no matching tool passed the threshold.
        """
        if not self._summaries:
            return []

        results = self.collection.query(
            query_texts=[user_request],
            n_results=self._query_n_results(),
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        candidates = []
        for i, worker_name in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            if distance > distance_threshold:
                continue
            candidates.append({
                "worker_name": worker_name,
                "summary": self._summaries.get(worker_name, ""),
                "distance":    distance,
            })

        log("ROUTER", "candidates found",
            n=len(candidates),
            workers=[c["worker_name"] for c in candidates])

        return candidates

    def get_description(self, worker_name: str) -> str | None:
        """Return the indexed summary for a tool."""
        return self._summaries.get(worker_name)


def build_router(config: dict, n_results: int | None = None) -> Router:
    """Build this router from the shared routing config."""
    return Router(
        scripts_dir=config["routing"]["scripts_dir"],
        ollama_embed_url=config["llm"]["ollama_embed_url"],
        embedding_model=config["llm"]["embedding_model"],
        n_results=n_results if n_results is not None else config["routing"].get("chromadb_n_results"),
    )


def route_prompt(prompt: str, router: Router, threshold: float) -> dict:
    """Route one prompt and return evaluator-friendly output."""
    candidates = router.find_candidates(prompt, distance_threshold=threshold)
    return {
        "prompt": prompt,
        "selected_tools": [candidate["worker_name"] for candidate in candidates],
        "distances": [round(candidate["distance"], 4) for candidate in candidates],
        "threshold": threshold,
        "n_results": router.n_results,
    }


def route_prompt_from_config(
    prompt: str,
    config_path: str | os.PathLike = DEFAULT_CONFIG,
) -> dict:
    """Convenience helper for direct one-prompt CLI-style use."""
    config = load_config(config_path)
    router = build_router(config)
    return route_prompt(prompt, router, config["routing"]["distance_threshold"])


def main():
    parser = argparse.ArgumentParser(description="Route one prompt using only ChromaDB.")
    parser.add_argument("prompt", help="User prompt to route")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    args = parser.parse_args()

    result = route_prompt_from_config(args.prompt, args.config)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
