#!/usr/bin/env python3
"""
Web tool — search, fetch, and save web pages within the agent workspace.

Usage:
  python tools/web.py <workspace_dir> '{"action": "search", "query": "python pathlib"}'
  python tools/web.py <workspace_dir> '{"action": "read", "url": "https://example.com"}'
"""

TOOL_NAME = "web"

TOOL_PROMPTS = {
    "full": """\
--- Tool: web ---
Search the web, read web pages, and save them to your workspace.

  search        DuckDuckGo search. Returns titles, URLs, snippets.
    {"tool_call": {"tool": "web", "action": "search", "query": "<query>", "max_results": 5}}

  read          Fetch a URL as Markdown (cached). Returns content directly.
    {"tool_call": {"tool": "web", "action": "read", "url": "<https://...>", "use_cache": true}}

  save          Fetch a URL and save its Markdown to your workspace.
    {"tool_call": {"tool": "web", "action": "save", "url": "<https://...>", "path": "<filename.md>", "use_cache": true}}
    path is relative to your workspace. Auto-generated from URL if omitted.

  read_and_save Fetch a URL, return its content AND save it in one call.
    {"tool_call": {"tool": "web", "action": "read_and_save", "url": "<https://...>", "path": "<filename.md>"}}
""",
}

TOOL_SCHEMAS = {
    "full": [
        {
            "type": "function",
            "function": {
                "name": "web__search",
                "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Maximum results (default 5)"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web__read",
                "description": "Fetch a web page and return it as Markdown. Results are cached by URL hash.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "use_cache": {"type": "boolean", "description": "Use cached result if available (default true)"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web__save",
                "description": "Fetch a web page and save its Markdown to the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "path": {"type": "string", "description": "Destination path relative to workspace. Auto-generated from URL if omitted."},
                        "use_cache": {"type": "boolean", "description": "Use cached result if available (default true)"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web__read_and_save",
                "description": "Fetch a web page, return its Markdown content, and save it to the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "path": {"type": "string", "description": "Destination path relative to workspace. Auto-generated if omitted."},
                        "use_cache": {"type": "boolean", "description": "Use cached result if available (default true)"},
                    },
                    "required": ["url"],
                },
            },
        },
    ],
}

import hashlib
import json
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _url_to_filename(url: str) -> str:
    name = re.sub(r"^https?://", "", url)
    name = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-")
    return f"{name[:80]}.md"


def _parse_front_matter(text: str) -> tuple:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    fm_block = text[4:end]
    body = text[end + 5:]
    meta = {}
    for line in fm_block.splitlines():
        if ": " in line:
            k, _, v = line.partition(": ")
            meta[k.strip()] = v.strip()
    return meta, body


def _make_front_matter(meta: dict) -> str:
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append("---\n")
    return "\n".join(lines)


def _fetch_markdown(url: str, workspace_dir: Path, use_cache: bool = True) -> tuple:
    """Fetch URL and return (markdown, title, fetched_at, from_cache)."""
    import html2text

    cache_dir = workspace_dir / "web_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_url_hash(url)}.md"

    if use_cache and cache_file.exists():
        raw = cache_file.read_text()
        meta, body = _parse_front_matter(raw)
        return body, meta.get("title", ""), meta.get("fetched_at", ""), True

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; bsa-agent/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html_bytes = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Fetch failed: {e}")

    html = html_bytes.decode("utf-8", errors="replace")

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    markdown = h.handle(html)

    fetched_at = _now()
    front_matter = _make_front_matter({"url": url, "fetched_at": fetched_at, "title": title})
    cache_file.write_text(front_matter + markdown)

    return markdown, title, fetched_at, False


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def search(query: str, workspace_dir: Path, max_results: int = 5) -> dict:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return {"query": query, "results": results, "count": len(results)}


def read(url: str, workspace_dir: Path, use_cache: bool = True) -> dict:
    markdown, title, fetched_at, from_cache = _fetch_markdown(url, workspace_dir, use_cache)
    return {
        "url": url,
        "markdown": markdown,
        "title": title,
        "fetched_at": fetched_at,
        "from_cache": from_cache,
    }


def save(url: str, workspace_dir: Path, path: str = None, use_cache: bool = True) -> dict:
    markdown, title, fetched_at, from_cache = _fetch_markdown(url, workspace_dir, use_cache)
    filename = path or _url_to_filename(url)
    dest = (workspace_dir / filename).resolve()
    if not str(dest).startswith(str(workspace_dir.resolve())):
        raise ValueError(f"Path must be within workspace, got: {path!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown)
    return {
        "url": url,
        "path": str(dest.relative_to(workspace_dir.resolve())),
        "title": title,
        "fetched_at": fetched_at,
        "from_cache": from_cache,
        "bytes_written": len(markdown.encode()),
    }


def read_and_save(url: str, workspace_dir: Path, path: str = None, use_cache: bool = True) -> dict:
    markdown, title, fetched_at, from_cache = _fetch_markdown(url, workspace_dir, use_cache)
    filename = path or _url_to_filename(url)
    dest = (workspace_dir / filename).resolve()
    if not str(dest).startswith(str(workspace_dir.resolve())):
        raise ValueError(f"Path must be within workspace, got: {path!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown)
    return {
        "url": url,
        "markdown": markdown,
        "title": title,
        "fetched_at": fetched_at,
        "from_cache": from_cache,
        "path": str(dest.relative_to(workspace_dir.resolve())),
        "bytes_written": len(markdown.encode()),
    }


ACTIONS = {
    "search": search,
    "read": read,
    "save": save,
    "read_and_save": read_and_save,
}


def dispatch(payload: dict, workspace_dir: str) -> dict:
    action = payload.get("action")
    if not action:
        return {"success": False, "error": "Missing 'action' field"}
    fn = ACTIONS.get(action)
    if fn is None:
        return {"success": False, "error": f"Unknown action: {action!r}. Valid: {list(ACTIONS)}"}
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    kwargs = {k: v for k, v in payload.items() if k != "action"}
    try:
        result = fn(workspace_dir=ws, **kwargs)
        return {"success": True, "result": result}
    except TypeError as e:
        return {"success": False, "error": f"Bad arguments: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/web.py <workspace_dir> '<json payload>'")
        sys.exit(1)
    try:
        payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)
    print(json.dumps(dispatch(payload, sys.argv[1])))
