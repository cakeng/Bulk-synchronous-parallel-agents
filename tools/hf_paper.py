#!/usr/bin/env python3
"""
HuggingFace Papers tool — list, search, get info, and read HF Daily Papers.

Uses the huggingface_hub Python library (already a project dependency).

Usage:
  python tools/hf_paper.py <workspace_dir> '{"action": "list"}'
  python tools/hf_paper.py <workspace_dir> '{"action": "list", "sort": "trending", "limit": 5}'
  python tools/hf_paper.py <workspace_dir> '{"action": "list", "date": "today"}'
  python tools/hf_paper.py <workspace_dir> '{"action": "search", "query": "vision language"}'
  python tools/hf_paper.py <workspace_dir> '{"action": "info", "arxiv_id": "2601.15621"}'
  python tools/hf_paper.py <workspace_dir> '{"action": "read", "arxiv_id": "2601.15621"}'
"""

TOOL_NAME = "hf_paper"

TOOL_PROMPTS = {
    "full": """\
--- Tool: hf_paper ---
Access HuggingFace Daily Papers: list, search, get structured metadata, and read full paper content.

  list          List HuggingFace Daily Papers. All filters are optional.
    {"tool_call": {"tool": "hf_paper", "action": "list"}}
    {"tool_call": {"tool": "hf_paper", "action": "list", "sort": "trending", "limit": 10}}
    {"tool_call": {"tool": "hf_paper", "action": "list", "date": "today"}}
    {"tool_call": {"tool": "hf_paper", "action": "list", "date": "2025-01-23"}}
    {"tool_call": {"tool": "hf_paper", "action": "list", "week": "2025-W09"}}
    {"tool_call": {"tool": "hf_paper", "action": "list", "month": "2025-02"}}
    {"tool_call": {"tool": "hf_paper", "action": "list", "submitter": "akhaliq", "limit": 5}}
    sort: "trending" | "latest" (default "latest")

  search        Search papers by keyword. Returns title, arxiv_id, authors, upvotes, summary.
    {"tool_call": {"tool": "hf_paper", "action": "search", "query": "vision language models", "limit": 10}}

  info          Get full structured metadata for a paper by ArXiv ID.
    {"tool_call": {"tool": "hf_paper", "action": "info", "arxiv_id": "2601.15621"}}
    Returns: title, summary (abstract), authors, upvotes, published_at, submitted_by,
             ai_summary, ai_keywords, github_repo, organization.

  read          Read a paper's full content as Markdown. Cached per arxiv_id.
    {"tool_call": {"tool": "hf_paper", "action": "read", "arxiv_id": "2601.15621"}}
    {"tool_call": {"tool": "hf_paper", "action": "read", "arxiv_id": "2601.15621", "use_cache": false}}
    Returns the complete paper text (abstract + body) in Markdown format.
""",
}

TOOL_SCHEMAS = {
    "full": [
        {
            "type": "function",
            "function": {
                "name": "hf_paper__list",
                "description": (
                    "List HuggingFace Daily Papers. Supports filtering by date, week, month, "
                    "or submitter and sorting by trending or latest."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sort":      {"type": "string",  "description": "Sort order: 'trending' or 'latest' (default 'latest')"},
                        "date":      {"type": "string",  "description": "Filter by date: 'today' or 'YYYY-MM-DD'"},
                        "week":      {"type": "string",  "description": "Filter by ISO week: 'YYYY-WNN' (e.g. '2025-W09')"},
                        "month":     {"type": "string",  "description": "Filter by month: 'YYYY-MM'"},
                        "submitter": {"type": "string",  "description": "Filter by HuggingFace username of the submitter"},
                        "limit":     {"type": "integer", "description": "Maximum number of results (default 20)"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hf_paper__search",
                "description": (
                    "Search HuggingFace papers by keyword. Returns titles, ArXiv IDs, "
                    "authors, summaries, and upvote counts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",  "description": "Search keywords or phrase"},
                        "limit": {"type": "integer", "description": "Maximum number of results (default 10)"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hf_paper__info",
                "description": (
                    "Get full structured metadata for a HuggingFace paper by its ArXiv ID: "
                    "title, abstract (summary), authors, upvotes, publication date, "
                    "AI-generated summary and keywords, GitHub repo, and organization."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arxiv_id": {"type": "string", "description": "ArXiv paper ID (e.g. '2601.15621')"},
                    },
                    "required": ["arxiv_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hf_paper__read",
                "description": (
                    "Read the full text of a paper as Markdown (abstract + body). "
                    "Results are cached in the workspace by arxiv_id."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arxiv_id":  {"type": "string",  "description": "ArXiv paper ID (e.g. '2601.15621')"},
                        "use_cache": {"type": "boolean", "description": "Return cached result if available (default true)"},
                    },
                    "required": ["arxiv_id"],
                },
            },
        },
    ],
}

import json
import sys
from datetime import date as _date
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_paper(p) -> dict:
    """Convert a huggingface_hub PaperInfo object to a plain JSON-safe dict."""
    def _str(v):
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        return str(v)

    # Authors: list of PaperAuthor objects with a .name attribute
    raw_authors = getattr(p, "authors", []) or []
    authors = [getattr(a, "name", str(a)) for a in raw_authors]

    # submitted_by: User object with .username
    sub = getattr(p, "submitted_by", None)
    submitted_by = getattr(sub, "username", None) if sub else None

    # organization: Organization object with .name
    org = getattr(p, "organization", None)
    organization = getattr(org, "name", None) if org else None

    return {
        "arxiv_id":     getattr(p, "id",           None),
        "title":        getattr(p, "title",         None),
        "summary":      getattr(p, "summary",       None),
        "authors":      authors,
        "upvotes":      getattr(p, "upvotes",       None),
        "published_at": _str(getattr(p, "published_at", None)),
        "submitted_at": _str(getattr(p, "submitted_at", None)),
        "submitted_by": submitted_by,
        "organization": organization,
        "ai_summary":   getattr(p, "ai_summary",   None),
        "ai_keywords":  getattr(p, "ai_keywords",  None),
        "github_repo":  getattr(p, "github_repo",  None),
        "github_stars": getattr(p, "github_stars", None),
        "comments":     getattr(p, "comments",     None),
    }


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def list_papers(workspace_dir: Path, sort: str = None, date: str = None,
                week: str = None, month: str = None,
                submitter: str = None, limit: int = 20) -> dict:
    from huggingface_hub import list_daily_papers

    if date == "today":
        date = _date.today().isoformat()

    kwargs = {}
    if sort:      kwargs["sort"]      = sort
    if date:      kwargs["date"]      = date
    if week:      kwargs["week"]      = week
    if month:     kwargs["month"]     = month
    if submitter: kwargs["submitter"] = submitter
    if limit:     kwargs["limit"]     = limit

    papers = list(list_daily_papers(**kwargs))
    return {
        "count":  len(papers),
        "papers": [_serialize_paper(p) for p in papers],
    }


def search_papers(query: str, workspace_dir: Path, limit: int = 10) -> dict:
    from huggingface_hub import list_papers as _list_papers

    results = list(_list_papers(query=query, limit=limit))
    return {
        "query":  query,
        "count":  len(results),
        "papers": [_serialize_paper(p) for p in results],
    }


def get_paper_info(arxiv_id: str, workspace_dir: Path) -> dict:
    from huggingface_hub import paper_info as _paper_info

    p = _paper_info(arxiv_id)
    return _serialize_paper(p)


def read_paper(arxiv_id: str, workspace_dir: Path, use_cache: bool = True) -> dict:
    from huggingface_hub import HfApi

    cache_dir = workspace_dir / "hf_paper_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{arxiv_id}.md"

    if use_cache and cache_file.exists():
        return {
            "arxiv_id":   arxiv_id,
            "markdown":   cache_file.read_text(),
            "from_cache": True,
        }

    api = HfApi()
    markdown = api.read_paper(arxiv_id)
    cache_file.write_text(markdown)
    return {
        "arxiv_id":   arxiv_id,
        "markdown":   markdown,
        "from_cache": False,
    }


ACTIONS = {
    "list":   list_papers,
    "search": search_papers,
    "info":   get_paper_info,
    "read":   read_paper,
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
        print("Usage: python tools/hf_paper.py <workspace_dir> '<json payload>'")
        sys.exit(1)
    try:
        payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)
    print(json.dumps(dispatch(payload, sys.argv[1]), default=str))
