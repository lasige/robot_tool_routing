"""
router_CDBv2.py - Dynamic ChromaDB semantic router

Receives a text request and returns the closest worker plus any workers whose
distance is within 60% of the closest worker's distance.

This avoids empty ChromaDB-only results for single-tool prompts while keeping
near-ties when ChromaDB sees multiple similarly relevant tools.
"""

import argparse
import json
import os

from router_CDBv1 import (
    DEFAULT_CONFIG,
    Router as BaseRouter,
    load_config,
)
from logger import log


SIMILARITY_MARGIN = 0.60
THRESHOLD_LABEL = "dynamic_best_distance_plus_60pct"


class Router(BaseRouter):
    """
    ChromaDB router with a dynamic per-query threshold.

    The first nearest neighbor is always returned. Additional candidates are
    returned only when their distance is no more than 60% above the best
    distance for the current query.
    """

    def __init__(
        self,
        scripts_dir: str,
        ollama_embed_url: str,
        embedding_model: str,
        n_results: int | None = None,
        similarity_margin: float = SIMILARITY_MARGIN,
    ):
        if similarity_margin < 0:
            raise ValueError("similarity_margin must be non-negative")
        self.similarity_margin = similarity_margin
        super().__init__(scripts_dir, ollama_embed_url, embedding_model, n_results)

    def find_candidates(
        self,
        user_request: str,
        distance_threshold: float | None = None,
    ) -> list[dict]:
        """
        Return the nearest worker and near-ties within the dynamic threshold.

        The distance_threshold argument is accepted for API compatibility with
        the original router, but this v2 router does not use it for filtering.
        """
        if not self._summaries:
            return []

        results = self.collection.query(
            query_texts=[user_request],
            n_results=self._query_n_results(),
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        best_distance = results["distances"][0][0]
        dynamic_threshold = best_distance * (1 + self.similarity_margin)

        candidates = []
        for i, worker_name in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            if i > 0 and distance > dynamic_threshold:
                continue
            candidates.append({
                "worker_name": worker_name,
                "summary": self._summaries.get(worker_name, ""),
                "distance": distance,
            })

        log(
            "ROUTER",
            "dynamic candidates found",
            n=len(candidates),
            best_distance=round(best_distance, 4),
            dynamic_threshold=round(dynamic_threshold, 4),
            workers=[c["worker_name"] for c in candidates],
        )

        return candidates


def build_router(config: dict, n_results: int | None = None) -> Router:
    """Build this router from the shared routing config."""
    return Router(
        scripts_dir=config["routing"]["scripts_dir"],
        ollama_embed_url=config["llm"]["ollama_embed_url"],
        embedding_model=config["llm"]["embedding_model"],
        n_results=n_results if n_results is not None else config["routing"].get("chromadb_n_results"),
    )


def route_prompt(prompt: str, router: Router, threshold: float | str | None = None) -> dict:
    """Route one prompt and return evaluator-friendly output."""
    candidates = router.find_candidates(prompt)
    dynamic_threshold = ""
    if candidates:
        dynamic_threshold = round(candidates[0]["distance"] * (1 + router.similarity_margin), 4)

    return {
        "prompt": prompt,
        "selected_tools": [candidate["worker_name"] for candidate in candidates],
        "distances": [round(candidate["distance"], 4) for candidate in candidates],
        "threshold": dynamic_threshold,
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
    parser = argparse.ArgumentParser(description="Route one prompt using dynamic ChromaDB v2.")
    parser.add_argument("prompt", help="User prompt to route")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    args = parser.parse_args()

    result = route_prompt_from_config(args.prompt, args.config)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
