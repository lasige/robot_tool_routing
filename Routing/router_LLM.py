#!/usr/bin/env python3
"""
router_LLM.py - pure LLM router

Provides the shared standalone Router interface:
    find_candidates(user_request, distance_threshold)
    get_description(worker_name)

It also keeps the one-prompt helpers used by the batch evaluator.

Input:
    python router_LLM.py "monitor the battery and land when it drops below 25%"

Output:
    {"prompt": "...", "selected_tools": [...], "model": "...", ...}
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests


ROUTING_DIR = Path(__file__).resolve().parent
SINGLE_ROBOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SINGLE_ROBOT_DIR / "config.json"
DEFAULT_CONTEXT = ROUTING_DIR / "context.json"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_MODEL = "gpt-oss:120b-cloud"

# Allow importing project helpers when this script is run directly.
if str(SINGLE_ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(SINGLE_ROBOT_DIR))

from logger import log
from project_paths import resolve_from_project


class LLMClient:
    """
    Client for the Ollama API (OpenAI-compatible format).

    The model passed to the constructor is the default.
    Each call to ask() can override it with the model= argument.
    """

    def __init__(self, ollama_url: str, default_model: str):
        self.ollama_url = ollama_url
        self.default_model = default_model

    def ask(self, system_prompt: str, user_message: str,
            model: str | None = None) -> str | None:
        """
        Single-turn LLM call.
        Retries one failed or empty response. Returns the cleaned response
        (<think> blocks stripped) or None after both attempts fail.

        Args:
            system_prompt: system prompt
            user_message:  user message
            model:         model to use (default: self.default_model)
        """
        chosen_model = model or self.default_model

        payload = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
        }

        for attempt in range(2):
            try:
                log("LLM", "calling", model=chosen_model, attempt=attempt + 1)
                response = requests.post(self.ollama_url, json=payload, timeout=120)
                response.raise_for_status()

                raw = response.json()["choices"][0]["message"]["content"]
                cleaned = self._strip_think(raw)
                if not cleaned:
                    raise ValueError("empty LLM response")

                log("LLM", "response received")
                return cleaned

            except (requests.exceptions.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
                log("ERROR", "LLM call failed", error=str(error), attempt=attempt + 1)
        return None

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks that some models generate."""
        if not text:
            return text
        cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tools(tools_dir: str, tool_fields: list[str]) -> list[dict]:
    """Load available tools and selected descriptive fields from Tools/*.json."""
    tools = []

    for filename in sorted(os.listdir(tools_dir)):
        if not filename.endswith(".json"):
            continue

        worker_name = filename.removesuffix(".json")
        path = os.path.join(tools_dir, filename)
        tool_config = load_json(path)

        tool = {"name": worker_name}
        for field in tool_fields:
            value = tool_config.get(field)
            if value:
                tool[field] = value

        tools.append(tool)

    return tools


def build_system_prompt(context: dict, tools: list[dict]) -> str:
    """Build the LLM routing prompt from context.json and loaded tools."""
    rules = "\n".join(f"- {rule}" for rule in context.get("rules", []))
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    output_schema = json.dumps(context.get("output_schema", {"tools": []}), ensure_ascii=False)

    return (
        f"{context.get('role', 'You are a tool router.')}\n\n"
        f"Task: {context.get('task', 'Select the required workers.')}\n\n"
        f"Rules:\n{rules}\n\n"
        f"Available workers:\n{tools_json}\n\n"
        f"Return exactly this JSON shape:\n{output_schema}\n"
    )


def parse_tools(raw_response: str | None, allowlist: set[str]) -> tuple[list[str], str | None]:
    """Parse and normalize selected tools from the model response."""
    if not raw_response:
        return [], "empty_response"

    text = raw_response.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return [], "invalid_json"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], "invalid_json"

    if isinstance(parsed, list):
        selected = parsed
    elif isinstance(parsed, dict):
        if "tools" not in parsed:
            return [], "missing_tools_key"
        selected = parsed["tools"]
    else:
        return [], "invalid_json_shape"

    if not isinstance(selected, list):
        return [], "invalid_tools_shape"

    returned = []
    invalid = []
    for item in selected:
        if not isinstance(item, str):
            invalid.append(str(item))
            continue

        worker_name = item.strip()
        if worker_name not in returned:
            returned.append(worker_name)

        if worker_name not in allowlist:
            invalid.append(worker_name)

    error = f"invalid_tools:{','.join(invalid)}" if invalid else None
    return returned, error


def build_routing_context(config: dict, context: dict, model: str | None = None) -> dict:
    """Load tools and build reusable state for one or many LLM routing calls."""
    selected_model = model or config["llm"]["routing_model"]
    tools_dir = resolve_from_project(config["routing"]["scripts_dir"])
    tool_fields = context.get("tool_fields", ["summary", "description"])
    tools = load_tools(tools_dir, tool_fields)

    return {
        "model": selected_model,
        "tools_dir": tools_dir,
        "tools": tools,
        "allowlist": {tool["name"] for tool in tools},
        "system_prompt": build_system_prompt(context, tools),
    }


def build_llm(config: dict, model: str) -> LLMClient:
    """Build the shared LLM client for routing."""
    return LLMClient(
        ollama_url=config["llm"]["ollama_url"],
        default_model=model,
    )


class Router:
    """
    Standalone pure-LLM tool router.

    It loads all worker JSON files, asks an LLM to choose worker names from that
    allowlist, and returns candidate dicts shaped like the ChromaDB router.
    `ollama_embed_url` and `embedding_model` are accepted for constructor
    compatibility with the ChromaDB-only router, but are not used.
    """

    def __init__(
        self,
        scripts_dir: str,
        ollama_embed_url: str | None = None,
        embedding_model: str | None = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        context_path: str | os.PathLike = DEFAULT_CONTEXT,
        tool_fields: list[str] | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.scripts_dir = resolve_from_project(scripts_dir)
        self.ollama_url = ollama_url
        self.model = model
        self.context_path = context_path
        self.context = load_json(context_path)
        self.tool_fields = tool_fields or self.context.get(
            "tool_fields", ["summary", "description"]
        )
        self.tools = load_tools(self.scripts_dir, self.tool_fields)
        self.allowlist = {tool["name"] for tool in self.tools}
        self.system_prompt = build_system_prompt(self.context, self.tools)
        self.llm = llm_client or LLMClient(
            ollama_url=self.ollama_url,
            default_model=self.model,
        )
        self._summaries = {
            tool["name"]: tool.get("summary", "") for tool in self.tools
        }

    def find_candidates(
        self,
        user_request: str,
        distance_threshold: float = 1.5,
    ) -> list[dict]:
        """
        Select candidate workers through the LLM.

        The `distance_threshold` argument is accepted so callers can use all
        routers the same way. Pure LLM routing has no distance metric, so
        selected workers are returned with `distance: 0.0`.
        """
        routed = route_prompt(user_request, self.llm, {
            "model": self.model,
            "allowlist": self.allowlist,
            "system_prompt": self.system_prompt,
        })

        if routed["parse_error"]:
            raise RuntimeError(f"LLM routing failed: {routed['parse_error']}")

        candidates = []
        for worker_name in routed["selected_tools"]:
            if worker_name not in self.allowlist:
                continue
            candidates.append({
                "worker_name": worker_name,
                "summary": self._summaries.get(worker_name, ""),
                "distance": 0.0,
            })

        return candidates

    def get_description(self, worker_name: str) -> str | None:
        """Return a worker summary by name."""
        return self._summaries.get(worker_name)


def build_router(
    config: dict,
    context_path: str | os.PathLike = DEFAULT_CONTEXT,
    model: str | None = None,
) -> Router:
    """Build this router from the shared routing config."""
    return Router(
        scripts_dir=config["routing"]["scripts_dir"],
        ollama_url=config["llm"]["ollama_url"],
        model=model or config["llm"]["routing_model"],
        context_path=context_path,
    )


def route_prompt(prompt: str, llm: LLMClient, routing_context: dict) -> dict:
    """Route one prompt through the pure-LLM tool selector."""
    raw_response = llm.ask(
        routing_context["system_prompt"],
        prompt,
        routing_context["model"],
    )
    returned, parse_error = parse_tools(raw_response, routing_context["allowlist"])

    return {
        "prompt": prompt,
        "selected_tools": returned,
        "model": routing_context["model"],
        "raw_response": raw_response or "",
        "parse_error": parse_error or "",
    }


def route_prompt_from_config(
    prompt: str,
    config_path: str | os.PathLike = DEFAULT_CONFIG,
    context_path: str | os.PathLike = DEFAULT_CONTEXT,
    model: str | None = None,
) -> dict:
    """Convenience helper for direct one-prompt CLI-style use."""
    config = load_json(config_path)
    context = load_json(context_path)
    routing_context = build_routing_context(config, context, model)
    llm = build_llm(config, routing_context["model"])
    return route_prompt(prompt, llm, routing_context)


def main():
    parser = argparse.ArgumentParser(description="Route one prompt using only an LLM.")
    parser.add_argument("prompt", help="User prompt to route")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    parser.add_argument("--context", default=str(DEFAULT_CONTEXT), help="Path to routing context JSON")
    parser.add_argument("--model", help="LLM model to use; defaults to config llm.routing_model")
    args = parser.parse_args()

    result = route_prompt_from_config(args.prompt, args.config, args.context, args.model)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
