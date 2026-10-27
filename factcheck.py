"""Small LLM fact-checker: claim -> Wikipedia evidence -> Supported / Refuted / Not Enough Info.

Usage:
    python factcheck.py "The Eiffel Tower is in Berlin."
    python factcheck.py "The Eiffel Tower is in Berlin." --page "Eiffel Tower"
    python factcheck.py "The Eiffel Tower is in Berlin." --no-evidence
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import anthropic
import requests

MODEL = "claude-opus-5-5"
LABELS = ["Supported", "Refuted", "Not Enough Info"]

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API clients to identify themselves.
HEADERS = {"User-Agent": "student-fact-checker/0.1 (research project)"}
CACHE_DIR = Path(__file__).parent / "evidence_cache"

SYSTEM_WITH_EVIDENCE = """You are a careful fact-checker. You will be given a claim and evidence text from Wikipedia.
Judge the claim using ONLY the evidence text, not your own background knowledge.
- Supported: the evidence clearly states or directly implies the claim is true.
- Refuted: the evidence clearly contradicts the claim.
- Not Enough Info: the evidence neither confirms nor contradicts the claim."""

SYSTEM_NO_EVIDENCE = """You are a careful fact-checker. You will be given a claim and no supporting documents.
Judge the claim using your own knowledge.
- Supported: you are confident the claim is true.
- Refuted: you are confident the claim is false.
- Not Enough Info: the claim cannot be verified as true or false from reliable, widely known information."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ---------- Evidence retrieval ----------

def wiki_get(params: dict) -> dict:
    """GET the Wikipedia API, backing off politely if rate-limited."""
    for attempt in range(5):
        r = requests.get(WIKI_API, headers=HEADERS, timeout=30, params=params)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        time.sleep(int(r.headers.get("Retry-After", 2 ** attempt * 5)))
    r.raise_for_status()
    return r.json()


def search_wikipedia(query: str) -> str | None:
    """Return the title of the top Wikipedia search result for a query."""
    hits = wiki_get({
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 1, "format": "json",
    })["query"]["search"]
    return hits[0]["title"] if hits else None


def fetch_wikipedia(title: str) -> str:
    """Return the full plain-text article for a Wikipedia title (cached on disk)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / (re.sub(r"[^\w\-]+", "_", title) + ".txt")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    data = wiki_get({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "redirects": 1, "format": "json",
    })
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"Wikipedia page not found: {title!r}")
    text = page["extract"]
    cache_file.write_text(text, encoding="utf-8")
    return text


# ---------- Classification ----------

def classify(claim: str, evidence: str | None, effort: str = "medium") -> dict:
    """Ask the LLM for a label. Pass evidence=None for the no-evidence condition."""
    if evidence is None:
        system = SYSTEM_NO_EVIDENCE
        user = f"Claim: {claim}"
    else:
        system = SYSTEM_WITH_EVIDENCE
        user = f"<evidence>\n{evidence}\n</evidence>\n\nClaim: {claim}"

    response = get_client().beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
        # If a safety classifier declines, retry on a fallback model in the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        return {"label": "ERROR", "reason": "model refused", "usage": response.usage}
    text = next(b.text for b in response.content if b.type == "text")
    result = json.loads(text)
    result["usage"] = response.usage
    return result


def check_claim(claim: str, page: str | None = None, use_evidence: bool = True,
                effort: str = "medium") -> dict:
    """Full pipeline for one claim."""
    evidence = None
    if use_evidence:
        page = page or search_wikipedia(claim)
        if page is None:
            return {"label": "Not Enough Info", "reason": "no Wikipedia page found", "page": None}
        evidence = fetch_wikipedia(page)
    result = classify(claim, evidence, effort=effort)
    result["page"] = page
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("claim")
    p.add_argument("--page", help="Wikipedia page title to use as evidence (default: search for one)")
    p.add_argument("--no-evidence", action="store_true", help="Judge from the model's own knowledge only")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    args = p.parse_args()

    result = check_claim(args.claim, page=args.page, use_evidence=not args.no_evidence, effort=args.effort)
    print(f"Claim:    {args.claim}")
    print(f"Evidence: {'(none)' if args.no_evidence else 'Wikipedia: ' + str(result['page'])}")
    print(f"Label:    {result['label']}")
    print(f"Reason:   {result['reason']}")


if __name__ == "__main__":
    main()
