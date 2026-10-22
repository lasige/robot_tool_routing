"""
router_CDBv3.py - Multi-query ChromaDB semantic router

Indexes summary, description, and keywords for each tool. For each user request, it
queries the full prompt plus intent-like clauses, then aggregates matches by
tool. The nearest match for each clause is kept, and a small score/cap cutoff
limits unrelated extras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from router_CDBv1 import (
    DEFAULT_CONFIG,
    Router as BaseRouter,
    load_config,
    resolve_from_project,
)
from logger import log


THRESHOLD_LABEL = "multiquery_summary_description_gap_v3"
CLAUSE_NEAR_MARGIN = 0.20
MAX_RESULTS = 8
DEFAULT_SCORE_GAP = 0.28
ROUTING_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_CACHE = ROUTING_DIR / ".cache" / "CDBv3_embeddings.json"

def split_prompt(prompt: str) -> list[str]:
    """Return full prompt plus useful intent-like clauses."""
    normalized = " ".join(prompt.strip().split())
    clauses = [normalized]

    for part in re.split(r"\s*(?:,|;|\band\b|\bthen\b)\s*", normalized, flags=re.IGNORECASE):
        part = part.strip(" .")
        if len(part) >= 4:
            clauses.append(part)

        for marker in (" if ", " when ", " before ", " after "):
            if marker in f" {part.lower()} ":
                left = re.split(marker, part, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .")
                if len(left) >= 4:
                    clauses.append(left)

    seen = set()
    unique = []
    for clause in clauses:
        key = clause.lower()
        if key not in seen:
            seen.add(key)
            unique.append(clause)
    return unique[:8]

class Router(BaseRouter):
    """
    ChromaDB router that indexes summary + description + keywords and searches
    multiple subqueries per prompt.
    """

    def __init__(
        self,
        *args,
        score_gap: float = DEFAULT_SCORE_GAP,
        max_results: int = MAX_RESULTS,
        embedding_cache_path: str | os.PathLike | None = DEFAULT_EMBEDDING_CACHE,
        **kwargs,
    ):
        self._score_gap = score_gap
        self._max_results = max_results
        self._embedding_cache_path = (
            Path(resolve_from_project(os.fspath(embedding_cache_path)))
            if embedding_cache_path
            else None
        )
        super().__init__(*args, **kwargs)

    def _cache_key(self, document: str) -> str:
        payload = f"{self.embedding_model}\0{document}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_embedding_cache(self) -> dict:
        if self._embedding_cache_path is None or not self._embedding_cache_path.exists():
            return {"version": 1, "entries": {}}

        try:
            with open(self._embedding_cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log("ROUTER", "embedding cache ignored", path=str(self._embedding_cache_path), error=str(exc))
            return {"version": 1, "entries": {}}

        if not isinstance(cache, dict) or not isinstance(cache.get("entries"), dict):
            log("ROUTER", "embedding cache ignored", path=str(self._embedding_cache_path), error="invalid format")
            return {"version": 1, "entries": {}}
        return cache

    def _save_embedding_cache(self, cache: dict) -> None:
        if self._embedding_cache_path is None:
            return

        try:
            self._embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._embedding_cache_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(tmp_path, self._embedding_cache_path)
        except OSError as exc:
            log("ROUTER", "embedding cache save failed", path=str(self._embedding_cache_path), error=str(exc))

    def _populate(self):
        """Read all tool JSON files and index summary + description + keywords."""
        worker_files = [f for f in os.listdir(self.scripts_dir) if f.endswith(".json")]

        if not worker_files:
            log("ROUTER", "no workers found", dir=self.scripts_dir)
            return

        ids, documents, metadatas = [], [], []

        for filename in worker_files:
            worker_name = filename.removesuffix(".json")
            filepath = os.path.join(self.scripts_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                worker_config = json.load(f)

            summary = worker_config.get("summary", "")
            description = worker_config.get("description", "")
            keywords = worker_config.get("keywords", [])
            if isinstance(keywords, list):
                keywords_text = ", ".join(str(keyword) for keyword in keywords)
            else:
                keywords_text = str(keywords)
            if not summary and not description and not keywords_text:
                log("ROUTER", "worker has no summary, description, or keywords, skipping", worker=worker_name)
                continue

            document = (
                f"Tool: {worker_name}\n"
                f"Summary: {summary}\n"
                f"Description: {description}\n"
                f"Keywords: {keywords_text}"
            )
            self._summaries[worker_name] = document
            ids.append(worker_name)
            documents.append(document)
            metadatas.append({"worker_name": worker_name})

        cache = self._load_embedding_cache()
        entries = cache.setdefault("entries", {})
        embeddings: list[list[float] | None] = [None] * len(documents)
        missing: list[tuple[int, str, str, str]] = []
        cached_count = 0

        for index, (worker_name, document) in enumerate(zip(ids, documents)):
            cache_key = self._cache_key(document)
            entry = entries.get(worker_name)
            embedding = entry.get("embedding") if isinstance(entry, dict) else None
            if (
                isinstance(entry, dict)
                and entry.get("model") == self.embedding_model
                and entry.get("cache_key") == cache_key
                and isinstance(embedding, list)
                and embedding
            ):
                embeddings[index] = embedding
                cached_count += 1
            else:
                missing.append((index, worker_name, document, cache_key))

        if missing:
            embedding_function = self._build_embedding_function()
            new_embeddings = embedding_function.embed_documents([item[2] for item in missing])
            for (index, worker_name, document, cache_key), embedding in zip(missing, new_embeddings):
                embeddings[index] = embedding
                entries[worker_name] = {
                    "model": self.embedding_model,
                    "cache_key": cache_key,
                    "document": document,
                    "embedding": embedding,
                }
            self._save_embedding_cache(cache)

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        log(
            "ROUTER",
            f"{len(ids)} workers loaded into ChromaDB",
            cached_embeddings=cached_count,
            embedded=len(missing),
        )

    def _query_candidates(self, query: str) -> list[tuple[str, float]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=self._query_n_results(),
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        pairs = []
        for worker_name, distance in zip(results["ids"][0], results["distances"][0]):
            pairs.append((worker_name, distance))
        return pairs

    def find_candidates(
        self,
        user_request: str,
        distance_threshold: float | None = None,
    ) -> list[dict]:
        """
        Search the full request and its clauses, aggregate by tool, and return
        a capped set of strongest matches.
        """
        if not self._summaries:
            return []

        queries = split_prompt(user_request)
        tools: dict[str, dict] = {}
        selected = set()

        for query_index, query in enumerate(queries):
            pairs = self._query_candidates(query)
            if not pairs:
                continue

            best_distance = pairs[0][1]
            near_threshold = best_distance * (1 + CLAUSE_NEAR_MARGIN)
            per_query_selected = []

            for rank, (worker_name, distance) in enumerate(pairs):
                if rank > 0 and (distance > near_threshold or len(per_query_selected) >= 3):
                    break

                per_query_selected.append(worker_name)
                entry = tools.setdefault(
                    worker_name,
                    {
                        "worker_name": worker_name,
                        "summary": self._summaries.get(worker_name, ""),
                        "best_distance": distance,
                        "hits": 0,
                        "first_query": query_index,
                    },
                )
                entry["best_distance"] = min(entry["best_distance"], distance)
                entry["hits"] += 1
                entry["first_query"] = min(entry["first_query"], query_index)

            if per_query_selected:
                selected.add(per_query_selected[0])

        for entry in tools.values():
            entry["score"] = (
                entry["best_distance"]
                - 0.06 * math.log1p(entry["hits"])
            )

        ranked = sorted(
            tools.values(),
            key=lambda item: (item["score"], item["first_query"], item["best_distance"]),
        )

        min_results = max(1, min(4, len(queries)), len(selected))
        max_results = min(self._max_results, max(min_results, len(selected) + 3))

        for entry in ranked:
            selected.add(entry["worker_name"])
            if len(selected) >= min_results:
                break

        ranked_selected = [entry for entry in ranked if entry["worker_name"] in selected]
        best_score = ranked[0]["score"] if ranked else 0.0
        for entry in ranked:
            if entry["worker_name"] in selected:
                continue
            if len(selected) >= max_results:
                break
            if entry["score"] > best_score + self._score_gap and len(selected) >= min_results:
                break
            selected.add(entry["worker_name"])
            ranked_selected.append(entry)

        ranked_selected = sorted(
            ranked_selected,
            key=lambda item: (item["score"], item["first_query"], item["best_distance"]),
        )

        candidates = [
            {
                "worker_name": entry["worker_name"],
                "summary": entry["summary"],
                "distance": entry["best_distance"],
            }
            for entry in ranked_selected
        ]

        log(
            "ROUTER",
            "multi-query candidates found",
            n=len(candidates),
            queries=queries,
            workers=[candidate["worker_name"] for candidate in candidates],
        )
        return candidates


def build_router(config: dict, n_results: int | None = None) -> Router:
    """Build this router from the shared routing config."""
    return Router(
        scripts_dir=config["routing"]["scripts_dir"],
        ollama_embed_url=config["llm"]["ollama_embed_url"],
        embedding_model=config["llm"]["embedding_model"],
        n_results=n_results if n_results is not None else config["routing"].get("chromadb_n_results"),
        embedding_cache_path=config["routing"].get("CDBv3_embedding_cache", DEFAULT_EMBEDDING_CACHE),
    )


def route_prompt(prompt: str, router: Router, threshold: float | str | None = None) -> dict:
    """Route one prompt and return evaluator-friendly output."""
    candidates = router.find_candidates(prompt)
    return {
        "prompt": prompt,
        "selected_tools": [candidate["worker_name"] for candidate in candidates],
        "distances": [round(candidate["distance"], 4) for candidate in candidates],
        "threshold": THRESHOLD_LABEL,
        "threshold_label": THRESHOLD_LABEL,
        "n_results": router.n_results,
    }


def route_prompt_from_config(
    prompt: str,
    config_path: str | os.PathLike = DEFAULT_CONFIG,
) -> dict:
    """Convenience helper for direct one-prompt CLI-style use."""
    config = load_config(config_path)
    router = build_router(config)
    return route_prompt(prompt, router)


def main():
    parser = argparse.ArgumentParser(description="Route one prompt using multi-query ChromaDB v3.")
    parser.add_argument("prompt", help="User prompt to route")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    args = parser.parse_args()

    result = route_prompt_from_config(args.prompt, args.config)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
