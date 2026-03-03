#!/usr/bin/env python3
"""Export Substack post metadata and AI summaries to CSV.

Features:
- Reads posts from a Substack RSS feed.
- Fetches each post and extracts title, link, likes, comments, and article text.
- Optionally uses OpenAI Responses API to generate summaries.
- Supports cookie-based auth for paywalled posts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from urllib.error import URLError
from urllib.request import urlopen
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
VIS_NETWORK_CDN_URL = "https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"

ARTICLE_STRIP_SELECTORS = [
    "script",
    "style",
    "noscript",
    "form",
    "button",
    "svg",
    "figure",
    ".digest-post-embed",
    ".subscription-widget-wrap",
    '[class*="subscription-widget"]',
    '[data-component-name*="Subscription"]',
    '[data-component-name*="Subscribe"]',
    '[data-component-name*="Widget"]',
    '[data-component-name*="comments"]',
    '[data-component-name*="discussion"]',
    '[data-testid*="comment"]',
    '[id*="comment"]',
    '[class*="comment"]',
    '[class*="discussion"]',
]


@dataclass
class PostRecord:
    title: str
    url: str
    publication_date: str
    likes: int | None
    comments: int | None
    restacks: int | None
    article_word_count: int | None
    main_word_count: int | None
    footnote_word_count: int | None
    image_count: int | None
    theme: str
    subthemes: str
    summary_short_neutral: str
    summary_short_mimic: str
    summary_very_short_snappy: str
    summary_dl: str
    best_quote: str
    best_quote_verified: str
    summary_long: str
    status: str
    error: str


CSV_COLUMNS = [
    "title",
    "url",
    "publication_date",
    "likes",
    "comments",
    "restacks",
    "article_word_count",
    "main_word_count",
    "footnote_word_count",
    "image_count",
    "theme",
    "subthemes",
    "summary_short_neutral",
    "summary_short_mimic",
    "summary_very_short_snappy",
    "summary_dl",
    "best_quote",
    "best_quote_verified",
    "summary_long",
    "status",
    "error",
]


def ensure_deps() -> tuple[Any, Any]:
    try:
        import requests as _requests
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: requests. Install with "
            "`python3 -m pip install requests beautifulsoup4`."
        ) from exc

    try:
        from bs4 import BeautifulSoup as _BeautifulSoup
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: beautifulsoup4. Install with "
            "`python3 -m pip install requests beautifulsoup4`."
        ) from exc

    return _requests, _BeautifulSoup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Substack posts to CSV with summaries")
    parser.add_argument("--substack-url", help="Base URL, e.g. https://example.substack.com")
    parser.add_argument("--urls-from-csv", default="", help="Read post URLs from an existing CSV 'url' column")
    parser.add_argument(
        "--update-existing-csv",
        default="",
        help="Update selected columns in an existing CSV by matching URL",
    )
    parser.add_argument(
        "--update-columns",
        default="",
        help="Comma-separated columns to overwrite in update mode, e.g. best_quote,best_quote_verified",
    )
    parser.add_argument(
        "--update-row-range",
        default="",
        help="Optional 1-based inclusive row range in update mode, e.g. 1-20",
    )
    parser.add_argument(
        "--merge-summaries-from-csv",
        default="",
        help="Copy summary columns from an existing CSV by matching URL",
    )
    parser.add_argument(
        "--post-url",
        action="append",
        default=[],
        help="Specific post URL to process (repeatable). If set, feed crawling is skipped.",
    )
    parser.add_argument("--output-csv", default="substack_export.csv", help="Output CSV path")
    parser.add_argument("--max-posts", type=int, default=0, help="Limit number of posts (0 = all found)")
    parser.add_argument("--cookie-header", default="", help="Raw Cookie header for authenticated requests")
    parser.add_argument(
        "--cookies-json",
        default="",
        help="Path to JSON cookies file (list of {name,value,domain,path,...})",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--sleep", type=float, default=0.3, help="Delay between post requests")
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Directory for cached article HTML (reused on future runs)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached HTML and re-fetch pages before updating cache",
    )
    parser.add_argument(
        "--fetch-max-retries",
        type=int,
        default=6,
        help="Max retries for Substack page fetches on 429/5xx",
    )
    parser.add_argument(
        "--fetch-backoff-base",
        type=float,
        default=2.0,
        help="Base backoff seconds for Substack fetch retries",
    )
    parser.add_argument(
        "--summary-sleep",
        type=float,
        default=1.0,
        help="Delay after each summary request to avoid rate limits",
    )
    parser.add_argument(
        "--summary-max-retries",
        type=int,
        default=6,
        help="Max retries for summary requests on 429/5xx",
    )
    parser.add_argument(
        "--summary-backoff-base",
        type=float,
        default=2.0,
        help="Base backoff seconds for summary retries",
    )
    parser.add_argument("--no-summaries", action="store_true", help="Skip AI summaries")
    parser.add_argument("--no-theme", action="store_true", help="Skip LLM theme classification")
    parser.add_argument("--no-subthemes", action="store_true", help="Skip LLM subtheme generation")
    parser.add_argument("--openai-model", default="gpt-4.1-mini", help="Model for summary generation")
    parser.add_argument(
        "--summary-variants",
        default="summary_short_neutral,summary_short_mimic,summary_very_short_snappy,summary_dl,best_quote,summary_long",
        help=(
            "Comma-separated variants to generate. Options: "
            "summary_short_neutral,summary_short_mimic,summary_very_short_snappy,summary_dl,best_quote,summary_long,none"
        ),
    )
    parser.add_argument(
        "--require-full-text",
        action="store_true",
        help="Mark row as auth_error if paywall/locked content is detected",
    )
    parser.add_argument("--resume", action="store_true", help="Skip URLs already in output CSV")
    parser.add_argument(
        "--link-graph-html",
        default="",
        help="Optional output path for an interactive internal-link graph HTML",
    )
    parser.add_argument(
        "--link-graph-html-standalone",
        action="store_true",
        help="Embed vis-network JS directly in HTML for offline sharing",
    )
    parser.add_argument(
        "--link-graph-dot",
        default="",
        help="Optional output path for a Graphviz DOT internal-link graph",
    )
    parser.add_argument(
        "--link-graph-json",
        default="",
        help="Optional output path for graph JSON data (for re-rendering without recrawling)",
    )
    parser.add_argument(
        "--link-graph-only",
        action="store_true",
        help="Only build link graph output (skip CSV row writing and summaries)",
    )
    parser.add_argument(
        "--render-link-graph-from-json",
        default="",
        help="Render graph output(s) from an existing graph JSON file (no crawling)",
    )
    parser.add_argument(
        "--render-link-graph-from-dot",
        default="",
        help="Render graph output(s) from an existing DOT file generated by this script (no crawling)",
    )
    return parser.parse_args()


def parse_summary_variants(raw: str) -> set[str]:
    allowed = {
        "summary_short_neutral",
        "summary_short_mimic",
        "summary_very_short_snappy",
        "summary_dl",
        "best_quote",
        "summary_long",
    }
    variants = {part.strip() for part in raw.split(",") if part.strip()}
    if not variants or variants == {"none"}:
        return set()
    if "none" in variants:
        raise SystemExit("Use either --summary-variants none or one/more variants, not both.")
    invalid = sorted(variants - allowed)
    if invalid:
        raise SystemExit(
            "Invalid --summary-variants value(s): "
            + ", ".join(invalid)
            + ". Allowed: "
            + ", ".join(sorted(allowed))
        )
    return variants


def parse_update_columns(raw: str) -> list[str]:
    if not raw.strip():
        raise SystemExit("In update mode, provide --update-columns.")
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    invalid = [c for c in cols if c not in CSV_COLUMNS or c == "url"]
    if invalid:
        raise SystemExit(f"Invalid --update-columns: {', '.join(invalid)}")
    return cols


def parse_row_range(raw: str) -> tuple[int, int] | None:
    if not raw.strip():
        return None
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", raw)
    if not m:
        raise SystemExit("Invalid --update-row-range format. Use START-END, e.g. 1-20.")
    start = int(m.group(1))
    end = int(m.group(2))
    if start < 1 or end < start:
        raise SystemExit("Invalid --update-row-range values.")
    return start, end


def canonical_base_url(raw: str) -> str:
    parsed = urlparse(raw.strip())
    if not parsed.scheme:
        parsed = urlparse("https://" + raw.strip())
    netloc = parsed.netloc
    if not netloc:
        raise ValueError(f"Invalid Substack URL: {raw}")
    return f"https://{netloc}"


def make_session(requests_module: Any, timeout: int, cookie_header: str = "", cookies_json_path: str = "") -> Any:
    session = requests_module.Session()
    session.headers.update({"User-Agent": DEFAULT_UA})
    session.request = _with_timeout(session.request, timeout)

    if cookie_header.strip():
        session.headers["Cookie"] = cookie_header.strip()

    if cookies_json_path:
        load_cookies_from_json(session, cookies_json_path)

    return session


def _with_timeout(func, timeout: int):
    def wrapped(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return func(*args, **kwargs)

    return wrapped


def load_cookies_from_json(session: Any, path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("cookies"), list):
        data = data["cookies"]  # Playwright storage_state format.
    if not isinstance(data, list):
        raise ValueError("cookies JSON must be a list or a dict with a 'cookies' list")

    for item in data:
        if not isinstance(item, dict) or "name" not in item or "value" not in item:
            continue
        session.cookies.set(
            item["name"],
            item["value"],
            domain=item.get("domain"),
            path=item.get("path", "/"),
        )


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for val in values:
        if val not in seen:
            seen.add(val)
            out.append(val)
    return out


def load_csv_rows(path: str, row_range: tuple[int, int] | None = None) -> tuple[Path, list[str], list[dict[str, str]]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for idx, row in enumerate(reader, start=1):
            if row_range is not None:
                start, end = row_range
                if idx < start or idx > end:
                    continue
            normalized: dict[str, str] = {}
            for key in fieldnames:
                if not key:
                    continue
                normalized[key] = (row.get(key) or "").strip()
            rows.append(normalized)
    return csv_path, fieldnames, rows


def load_urls_from_csv(path: str) -> list[str]:
    csv_path, fieldnames, rows = load_csv_rows(path)
    if "url" not in fieldnames:
        raise SystemExit(f"CSV must include a 'url' column: {csv_path}")
    urls: list[str] = []
    for row in rows:
        raw = (row.get("url") or "").strip()
        if raw:
            urls.append(raw)
    return dedupe_keep_order(urls)


def select_urls_from_existing_csv(path: str, row_range: tuple[int, int] | None) -> list[str]:
    csv_path, fieldnames, rows = load_csv_rows(path, row_range=row_range)
    if "url" not in fieldnames:
        raise SystemExit(f"CSV must include a 'url' column: {csv_path}")
    urls: list[str] = []
    for row in rows:
        u = (row.get("url") or "").strip()
        if u:
            urls.append(u)
    return dedupe_keep_order(urls)


def load_url_date_map_from_csv(path: str, row_range: tuple[int, int] | None = None) -> dict[str, str]:
    csv_path, fieldnames, rows = load_csv_rows(path, row_range=row_range)
    if "url" not in fieldnames:
        raise SystemExit(f"CSV must include a 'url' column: {csv_path}")
    if "publication_date" not in fieldnames:
        return {}

    out: dict[str, str] = {}
    for row in rows:
        url = (row.get("url") or "").strip()
        date = (row.get("publication_date") or "").strip()
        if not url:
            continue
        normalized_url = normalize_post_url(url)
        if date:
            out[normalized_url] = date
    return out


def record_to_row_dict(record: PostRecord) -> dict[str, str]:
    return {
        "title": record.title,
        "url": record.url,
        "publication_date": record.publication_date,
        "likes": "" if record.likes is None else str(record.likes),
        "comments": "" if record.comments is None else str(record.comments),
        "restacks": "" if record.restacks is None else str(record.restacks),
        "article_word_count": "" if record.article_word_count is None else str(record.article_word_count),
        "main_word_count": "" if record.main_word_count is None else str(record.main_word_count),
        "footnote_word_count": "" if record.footnote_word_count is None else str(record.footnote_word_count),
        "image_count": "" if record.image_count is None else str(record.image_count),
        "theme": record.theme,
        "subthemes": record.subthemes,
        "summary_short_neutral": record.summary_short_neutral,
        "summary_short_mimic": record.summary_short_mimic,
        "summary_very_short_snappy": record.summary_very_short_snappy,
        "summary_dl": record.summary_dl,
        "best_quote": record.best_quote,
        "best_quote_verified": record.best_quote_verified,
        "summary_long": record.summary_long,
        "status": record.status,
        "error": record.error,
    }


def load_summary_map_from_csv(path: str) -> dict[str, dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    summary_fields = [
        "theme",
        "subthemes",
        "summary_short_neutral",
        "summary_short_mimic",
        "summary_very_short_snappy",
        "summary_dl",
        "best_quote",
        "best_quote_verified",
        "summary_long",
    ]
    out: dict[str, dict[str, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "url" not in reader.fieldnames:
            raise SystemExit(f"CSV must include a 'url' column: {csv_path}")
        for row in reader:
            raw_url = (row.get("url") or "").strip()
            if not raw_url:
                continue
            values = {field: (row.get(field) or "").strip() for field in summary_fields}
            out[raw_url] = values
            try:
                out[normalize_post_url(raw_url)] = values
            except Exception:
                pass
    return out


def cache_file_for_url(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.html"


def load_cached_html(cache_dir: Path, url: str) -> str | None:
    path = cache_file_for_url(cache_dir, url)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def save_cached_html(cache_dir: Path, url: str, html: str) -> None:
    path = cache_file_for_url(cache_dir, url)
    path.write_text(html, encoding="utf-8")


def fetch_feed_urls(session: Any, base_url: str) -> list[str]:
    feed_url = urljoin(base_url, "/feed")
    res = session.get(feed_url)
    res.raise_for_status()

    root = ET.fromstring(res.text)
    urls: list[str] = []

    for item in root.findall(".//item"):
        link = item.findtext("link")
        if link:
            urls.append(link.strip())

    # Fallback for Atom variants
    if not urls:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns):
            for link_elem in entry.findall("atom:link", ns):
                href = link_elem.attrib.get("href")
                rel = link_elem.attrib.get("rel", "alternate")
                if href and rel in ("alternate", ""):
                    urls.append(href.strip())
                    break

    return dedupe_keep_order(urls)


def normalize_post_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def is_probable_post_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if "/p/" not in path:
        return False
    # Exclude non-article endpoints.
    blocked_suffixes = ("/comments", "/share", "/podcast")
    return not path.endswith(blocked_suffixes)


def fetch_archive_urls(
    session: Any,
    base_url: str,
    bs4_class: Any,
    limit: int = 0,
    max_pages: int = 40,
) -> list[str]:
    urls: list[str] = []
    seen = set()
    base_netloc = urlparse(base_url).netloc.lower()

    for page in range(1, max_pages + 1):
        page_url = f"{base_url}/archive?sort=new"
        if page > 1:
            page_url += f"&page={page}"

        res = session.get(page_url)
        if res.status_code >= 400:
            break

        soup = bs4_class(res.text, "html.parser")
        page_urls: list[str] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href")
            if not isinstance(href, str) or not href.strip():
                continue
            absolute = normalize_post_url(urljoin(base_url, href.strip()))
            if not is_probable_post_url(absolute):
                continue

            netloc = urlparse(absolute).netloc.lower()
            if netloc not in {base_netloc, "substack.com", "www.substack.com"}:
                continue
            page_urls.append(absolute)

        page_urls = dedupe_keep_order(page_urls)
        new_urls = [u for u in page_urls if u not in seen]
        if not new_urls:
            # Stop when pagination no longer yields unseen post URLs.
            break

        for u in new_urls:
            seen.add(u)
            urls.append(u)

        if limit > 0 and len(urls) >= limit:
            break

    return urls


def extract_archive_item_url(item: dict[str, Any], base_url: str) -> str | None:
    direct_keys = ["canonical_url", "post_url", "url", "web_url", "canonicalUrl"]
    for key in direct_keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_post_url(urljoin(base_url, value.strip()))

    slug = item.get("slug")
    if isinstance(slug, str) and slug.strip():
        return normalize_post_url(urljoin(base_url, f"/p/{slug.strip()}"))
    return None


def fetch_archive_api_urls(
    session: Any,
    base_url: str,
    limit: int = 0,
    page_size: int = 12,
    max_pages: int = 80,
) -> list[str]:
    urls: list[str] = []
    seen = set()
    base_netloc = urlparse(base_url).netloc.lower()

    offset = 0
    for _ in range(max_pages):
        params = {"offset": offset, "limit": page_size}
        res = session.get(f"{base_url}/api/v1/archive", params=params)
        if res.status_code >= 400:
            break

        payload = res.json()
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            raw_items = payload.get("posts") or payload.get("items") or payload.get("results") or []
            items = raw_items if isinstance(raw_items, list) else []
        else:
            items = []

        if not items:
            break

        batch_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = extract_archive_item_url(item, base_url)
            if not candidate or not is_probable_post_url(candidate):
                continue
            netloc = urlparse(candidate).netloc.lower()
            if netloc not in {base_netloc, "substack.com", "www.substack.com"}:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            urls.append(candidate)
            batch_count += 1
            if limit > 0 and len(urls) >= limit:
                return urls

        if batch_count == 0:
            break
        if len(items) < page_size:
            break
        offset += page_size

    return urls


def _dedupe_nearby(lines: list[str]) -> list[str]:
    compact: list[str] = []
    last = None
    for line in lines:
        if line != last:
            compact.append(line)
            last = line
    return compact


def _collect_text_blocks(container: Any, selector: str) -> list[str]:
    blocks: list[str] = []
    for node in container.select(selector):
        # Keep only visible anchor text, never href destinations.
        for anchor in node.select("a[href]"):
            anchor.replace_with(anchor.get_text(" ", strip=True))
        txt = " ".join(node.get_text(" ", strip=True).split())
        # Remove raw URL destinations that may appear in inline widgets.
        txt = re.sub(r"https?://\S+", "", txt).strip()
        if txt:
            blocks.append(txt)
    return _dedupe_nearby(blocks)


def clean_anchor_text(raw: str, max_chars: int = 120) -> str:
    text = " ".join(raw.split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _context_words(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", text.strip()) if tok]


def extract_anchor_context(anchor: Any, window_words: int = 10) -> tuple[str, str]:
    marker = "__CODX_ANCHOR_MARKER__"
    block = anchor.find_parent(["p", "li", "blockquote", "h1", "h2", "h3", "h4"])
    if block is None:
        block = anchor.parent
    if block is None:
        return "", ""

    block_text = " ".join(block.get_text(" ", strip=True).split())
    anchor_text = clean_anchor_text(anchor.get_text(" ", strip=True), max_chars=200)
    if not block_text or not anchor_text:
        return "", ""

    anchored = block_text.replace(anchor_text, marker, 1)
    if marker not in anchored:
        return "", ""

    before_raw, after_raw = anchored.split(marker, 1)
    before_words = _context_words(before_raw)
    after_words = _context_words(after_raw)
    before = " ".join(before_words[-window_words:])
    after = " ".join(after_words[:window_words])
    return before, after


def extract_internal_post_links(
    soup: Any,
    bs4_class: Any,
    source_url: str,
    known_post_urls: set[str],
) -> list[dict[str, str]]:
    article = soup.find("article")
    if article is None:
        article = (
            soup.select_one('[data-testid="post-content"]')
            or soup.select_one('[class*="post-content"]')
            or soup.select_one('[class*="available-content"]')
            or soup
        )

    clone = bs4_class(str(article), "html.parser")
    for selector in ARTICLE_STRIP_SELECTORS:
        for node in clone.select(selector):
            node.decompose()

    # Footnote wrappers tend to add navigational artifacts and duplicate references.
    for node in clone.select(".footnote-content, [class*='footnote'], [id*='footnote']"):
        node.decompose()

    # Remove common recommendation/recirculation containers (e.g., "Recent Episodes").
    recommendation_selectors = [
        '[class*="recent"]',
        '[id*="recent"]',
        '[class*="related"]',
        '[id*="related"]',
        '[class*="recommended"]',
        '[id*="recommended"]',
        '[class*="episode"]',
        '[id*="episode"]',
        '[class*="podcast"]',
        '[id*="podcast"]',
        '[data-testid*="post-preview"]',
        '[class*="post-preview"]',
        "aside",
        "nav",
    ]
    for selector in recommendation_selectors:
        for node in clone.select(selector):
            node.decompose()

    links: list[dict[str, str]] = []
    source_norm = normalize_post_url(source_url)
    seen_pairs: set[tuple[str, str, str, str]] = set()
    # Restrict to links found inside body-text blocks only.
    for block in clone.select("p, li, blockquote, h1, h2, h3, h4"):
        for anchor in block.select("a[href]"):
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            href = href.strip()
            if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue

            absolute = normalize_post_url(urljoin(source_url, href))
            if absolute == source_norm or absolute not in known_post_urls:
                continue

            anchor_text = clean_anchor_text(anchor.get_text(" ", strip=True))
            if not anchor_text:
                continue

            before, after = extract_anchor_context(anchor, window_words=10)
            pair = (absolute, anchor_text, before, after)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            links.append(
                {
                    "target_url": absolute,
                    "anchor": anchor_text,
                    "before": before,
                    "after": after,
                }
            )
    return links


def extract_article_parts(soup: Any, bs4_class: Any) -> tuple[str, str, int]:
    article = soup.find("article")
    if article is None:
        article = (
            soup.select_one('[data-testid="post-content"]')
            or soup.select_one('[class*="post-content"]')
            or soup.select_one('[class*="available-content"]')
            or soup
        )

    # Work on a cloned subtree so removals do not affect other parsing.
    clone = bs4_class(str(article), "html.parser")

    # Remove likely non-article blocks, especially comments/discussion UI.
    for selector in ARTICLE_STRIP_SELECTORS:
        for node in clone.select(selector):
            node.decompose()

    footnote_blocks: list[str] = []
    seen_nodes = set()
    # Primary source of footnotes on Substack pages.
    footnote_nodes = clone.select(".footnote-content")
    for node in footnote_nodes:
        node_id = id(node)
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        local_blocks = _collect_text_blocks(node, "p, li")
        footnote_blocks.extend(local_blocks)
        if not local_blocks:
            txt = " ".join(node.get_text(" ", strip=True).split())
            txt = re.sub(r"https?://\S+", "", txt).strip()
            if txt:
                footnote_blocks.append(txt)
        # Decompose the full footnote wrapper so it cannot leak into main text.
        parent = node.parent
        if parent is not None and parent.get("class") and any("footnote" in c for c in parent.get("class", [])):
            parent.decompose()
        else:
            node.decompose()

    image_count = len(clone.select(".captioned-image-container"))
    main_blocks = _collect_text_blocks(clone, "h1, h2, h3, h4, p, li, blockquote")
    footnote_blocks = _dedupe_nearby(footnote_blocks)
    return "\n".join(main_blocks), "\n".join(footnote_blocks), image_count


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def parse_json_from_script(script_text: str) -> Any | None:
    try:
        return json.loads(script_text)
    except Exception:
        return None


def collect_interaction_stat_counts(data: Any, out: dict[str, list[int]]) -> None:
    if isinstance(data, dict):
        interaction_type = data.get("interactionType")
        user_count = coerce_nonnegative_int(data.get("userInteractionCount"))
        if isinstance(interaction_type, str) and user_count is not None:
            low = interaction_type.lower()
            if "likeaction" in low:
                out["likes"].append(user_count)
            elif "commentaction" in low:
                out["comments"].append(user_count)
            elif "shareaction" in low:
                out["restacks"].append(user_count)
        for val in data.values():
            collect_interaction_stat_counts(val, out)
    elif isinstance(data, list):
        for item in data:
            collect_interaction_stat_counts(item, out)


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value.is_integer() and value >= 0:
            return int(value)
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*", value)
        if match:
            return int(match.group(1))
    return None


def collect_numeric_by_key(data: Any, key_match: set[str], out: list[int]) -> None:
    if isinstance(data, dict):
        for key, val in data.items():
            normalized = normalize_key(key)
            if normalized in key_match:
                num = coerce_nonnegative_int(val)
                if num is not None:
                    out.append(num)
            collect_numeric_by_key(val, key_match, out)
    elif isinstance(data, list):
        for item in data:
            collect_numeric_by_key(item, key_match, out)


def extract_post_id(url: str) -> str | None:
    match = re.search(r"/p-(\d+)", url)
    return match.group(1) if match else None


def dict_matches_target(data: dict[str, Any], url: str, post_id: str | None) -> bool:
    target_url = url.split("?", 1)[0].rstrip("/")
    for val in data.values():
        if isinstance(val, str):
            low = val.lower()
            if target_url and target_url.lower() in low:
                return True
            if post_id and post_id in low:
                return True
    return False


def collect_post_count_candidates(
    data: Any,
    url: str,
    post_id: str | None,
    like_keys: set[str],
    comment_keys: set[str],
    restack_keys: set[str],
    out: list[tuple[int | None, int | None, int | None, int]],
) -> None:
    if isinstance(data, dict):
        like_val: int | None = None
        comment_val: int | None = None
        restack_val: int | None = None
        for key, val in data.items():
            normalized = normalize_key(key)
            if normalized in like_keys:
                num = coerce_nonnegative_int(val)
                if num is not None:
                    like_val = num
            if normalized in comment_keys:
                num = coerce_nonnegative_int(val)
                if num is not None:
                    comment_val = num
            if normalized in restack_keys:
                num = coerce_nonnegative_int(val)
                if num is not None:
                    restack_val = num

        if like_val is not None or comment_val is not None or restack_val is not None:
            score = 1
            score += int(like_val is not None)
            score += int(comment_val is not None)
            score += int(restack_val is not None)
            if dict_matches_target(data, url, post_id):
                score += 6
            out.append((like_val, comment_val, restack_val, score))

        for val in data.values():
            collect_post_count_candidates(val, url, post_id, like_keys, comment_keys, restack_keys, out)
    elif isinstance(data, list):
        for item in data:
            collect_post_count_candidates(item, url, post_id, like_keys, comment_keys, restack_keys, out)


def pick_count(values: list[int]) -> int | None:
    if not values:
        return None
    counts: dict[int, int] = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1
    best_freq = max(counts.values())
    best_vals = [val for val, freq in counts.items() if freq == best_freq]
    return min(best_vals)


def parse_counts(soup: Any, html: str, url: str) -> tuple[int | None, int | None, int | None]:
    like_keys = {
        "likecount",
        "likes",
        "likescount",
        "reactioncount",
        "reactionscount",
        "numlikes",
        "heartcount",
        "totalreactions",
        "reactiontotal",
    }
    comment_keys = {
        "commentcount",
        "comments",
        "commentscount",
        "numcomments",
        "discussioncount",
        "totalcomments",
        "replycount",
    }
    restack_keys = {
        "restackcount",
        "restacks",
        "restackscount",
        "numrestacks",
        "repostcount",
        "reposts",
        "sharecount",
        "sharescount",
    }

    likes: list[int] = []
    comments: list[int] = []
    restacks: list[int] = []
    post_candidates: list[tuple[int | None, int | None, int | None, int]] = []
    post_id = extract_post_id(url)

    # Structured scripts: __NEXT_DATA__ and JSON-LD are common.
    for script in soup.find_all("script"):
        payload = (script.string or script.get_text() or "").strip()
        if not payload:
            continue
        data = parse_json_from_script(payload)
        if data is not None:
            collect_numeric_by_key(data, like_keys, likes)
            collect_numeric_by_key(data, comment_keys, comments)
            collect_numeric_by_key(data, restack_keys, restacks)
            collect_interaction_stat_counts(
                data,
                {"likes": likes, "comments": comments, "restacks": restacks},
            )
            collect_post_count_candidates(
                data,
                url,
                post_id,
                like_keys,
                comment_keys,
                restack_keys,
                post_candidates,
            )

    # Regex fallbacks in raw HTML.
    patterns = [
        (r'"like_count"\s*:\s*(\d+)', likes),
        (r'"likes"\s*:\s*(\d+)', likes),
        (r'"reaction_count"\s*:\s*(\d+)', likes),
        (r'"comment_count"\s*:\s*(\d+)', comments),
        (r'"comments"\s*:\s*(\d+)', comments),
        (r'"discussion_count"\s*:\s*(\d+)', comments),
        (r'"restack_count"\s*:\s*(\d+)', restacks),
        (r'"restacks"\s*:\s*(\d+)', restacks),
        (r'"repost_count"\s*:\s*(\d+)', restacks),
    ]

    for pattern, bucket in patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            bucket.append(int(match))

    # Text/attribute fallback for UI labels such as "1 like" or "0 comments".
    label_texts: list[str] = []
    for attr in ("aria-label", "title", "data-tooltip-content"):
        for node in soup.find_all(attrs={attr: True}):
            val = node.get(attr)
            if isinstance(val, str):
                label_texts.append(val)

    like_text_patterns = [
        r"\b(\d+)\s+likes?\b",
        r"\blikes?\s*[:(]?\s*(\d+)\b",
    ]
    comment_text_patterns = [
        r"\b(\d+)\s+comments?\b",
        r"\bcomments?\s*[:(]?\s*(\d+)\b",
    ]
    restack_text_patterns = [
        r"\b(\d+)\s+restacks?\b",
        r"\brestacks?\s*[:(]?\s*(\d+)\b",
        r"\b(\d+)\s+reposts?\b",
    ]

    for text in label_texts:
        for pattern in like_text_patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                likes.append(int(match))
        for pattern in comment_text_patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                comments.append(int(match))
        for pattern in restack_text_patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                restacks.append(int(match))

    # Dedicated fallback for UFI share/restack button labels in post headers.
    ufi_region = soup.select_one("div.post-ufi")
    if ufi_region:
        for button in ufi_region.select("button[id^='radix-']"):
            label = button.select_one("div.label")
            if not label:
                continue
            val = coerce_nonnegative_int(label.get_text(" ", strip=True))
            if val is not None:
                restacks.append(val)

    like_val: int | None = None
    comment_val: int | None = None
    restack_val: int | None = None
    if post_candidates:
        best_like, best_comment, best_restack, _ = max(post_candidates, key=lambda x: x[3])
        like_val = best_like
        comment_val = best_comment
        restack_val = best_restack

    if like_val is None:
        like_val = pick_count(likes)
    if comment_val is None:
        comment_val = pick_count(comments)
    if restack_val is None:
        restack_val = pick_count(restacks)

    return like_val, comment_val, restack_val


def extract_title(soup: Any) -> str:
    candidates = [
        soup.select_one('meta[property="og:title"]'),
        soup.select_one('meta[name="twitter:title"]'),
    ]
    for tag in candidates:
        if tag and tag.get("content"):
            return tag["content"].strip()

    h1 = soup.find("h1")
    if h1:
        txt = h1.get_text(" ", strip=True)
        if txt:
            return txt

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return ""


def extract_publication_date(soup: Any) -> str:
    # Prefer structured JSON-LD datePublished when available.
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        payload = (script.string or script.get_text() or "").strip()
        if not payload:
            continue
        data = parse_json_from_script(payload)
        date_val: str | None = None
        if isinstance(data, dict):
            raw = data.get("datePublished")
            if isinstance(raw, str):
                date_val = raw
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("datePublished"), str):
                    date_val = item["datePublished"]
                    break
        if date_val:
            try:
                return datetime.fromisoformat(date_val.replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                pass

    # Fallback to visible date text in post header metadata.
    date_text_candidates: list[str] = []
    for node in soup.select("article .meta-EgzBVA, article time, article [datetime]"):
        txt = " ".join(node.get_text(" ", strip=True).split())
        if txt:
            date_text_candidates.append(txt)
        dt = node.get("datetime") if hasattr(node, "get") else None
        if isinstance(dt, str) and dt.strip():
            date_text_candidates.append(dt.strip())

    for txt in date_text_candidates:
        # Examples: "Feb 19, 2026", "2026-02-19T13:02:48+00:00"
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(txt, fmt).date().isoformat()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(txt.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            continue
    return ""


def looks_paywalled(soup: Any, html: str, extracted_text: str, article_word_count: int) -> bool:
    # Long extracted content is a strong signal that full text is visible.
    if article_word_count >= 800:
        return False

    lower_html = html.lower()
    button_text = " ".join(node.get_text(" ", strip=True).lower() for node in soup.select("button,a"))
    candidates = [extracted_text.lower(), lower_html, button_text]

    hard_markers = [
        "this post is for paid subscribers",
        "subscribe to read the rest",
        "unlock this post",
        "continue reading with a subscription",
        "paid-subscriber-only post",
        "data-component-name=\"paywall\"",
    ]
    soft_markers = [
        "subscribe to read",
        "become a paid subscriber",
        "upgrade to paid",
        "members only",
        "already a subscriber? sign in",
    ]

    if any(marker in blob for blob in candidates for marker in hard_markers):
        return True
    if article_word_count < 180 and any(marker in blob for blob in candidates for marker in soft_markers):
        return True
    return False


def _retry_after_seconds(response: Any) -> float | None:
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def format_request_exception(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    status = getattr(response, "status_code", None)
    try:
        body = (response.text or "").strip().replace("\n", " ")
    except Exception:
        body = ""
    if len(body) > 220:
        body = body[:220] + "..."
    if status is not None and body:
        return f"HTTP {status}: {body}"
    if status is not None:
        return f"HTTP {status}: {exc}"
    return str(exc)


def http_get_with_retries(
    requests_module: Any,
    session: Any,
    url: str,
    max_retries: int,
    backoff_base: float,
    label: str = "request",
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            res = session.get(url)
            status = res.status_code
            if status == 429 or status >= 500:
                if attempt >= max_retries:
                    res.raise_for_status()
                retry_after = _retry_after_seconds(res)
                wait = retry_after if retry_after is not None else backoff_base * (2**attempt)
                wait = min(wait, 120.0)
                print(
                    f"  {label} rate/server limit (HTTP {status}); retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            return res
        except requests_module.exceptions.RequestException as e:
            if attempt >= max_retries:
                raise
            wait = min(backoff_base * (2**attempt), 120.0)
            print(
                f"  {label} error ({format_request_exception(e)}); retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"{label} failed without a response")


def summarize_with_openai(
    requests_module: Any,
    api_key: str,
    model: str,
    title: str,
    text: str,
    word_limit: int,
    target_words: int | None,
    style_instructions: str = "",
    include_default_content_guidance: bool = True,
    timeout: int = 60,
    max_retries: int = 6,
    backoff_base: float = 2.0,
) -> str:
    if not text.strip():
        return ""

    excerpt = text[:16000]
    length_instruction = (
        f"Use about {target_words} words and keep it under {word_limit} words."
        if target_words is not None
        else f"Use no more than {word_limit} words."
    )
    prompt_parts = [
        "Summarize the following Substack article.",
        length_instruction,
    ]
    if include_default_content_guidance:
        prompt_parts.append(
            "If it is autobiographical, emphasize the narrative arc. "
            "If it is primarily philosophical or economic, emphasize the underlying ideology or philosophy. "
            "Sentence fragments are allowed."
        )
    if style_instructions.strip():
        prompt_parts.append(style_instructions.strip())
    prompt_parts.append("Return only the summary text.")
    prompt_parts.append(f"Title: {title}")
    prompt_parts.append(f"Article:\n{excerpt}")
    prompt = "\n\n".join(prompt_parts)

    payload: dict[str, Any] | None = None
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            res = requests_module.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": 100,
                },
                timeout=timeout,
            )
            status = res.status_code

            if status == 429 or status >= 500:
                if attempt >= max_retries:
                    res.raise_for_status()
                retry_after = _retry_after_seconds(res)
                wait = retry_after if retry_after is not None else backoff_base * (2**attempt)
                wait = min(wait, 90.0)
                print(
                    f"  Summary rate/server limit (HTTP {status}); retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue

            res.raise_for_status()
            payload = res.json()
            break
        except requests_module.exceptions.RequestException as e:
            last_error = format_request_exception(e)
            if attempt >= max_retries:
                raise
            wait = min(backoff_base * (2**attempt), 90.0)
            print(
                f"  Summary request error ({last_error}); retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)

    if payload is None:
        raise RuntimeError(last_error or "Summary request failed without a response payload")

    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()

    texts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])

    return "\n".join(texts).strip()


def classify_theme_with_openai(
    requests_module: Any,
    api_key: str,
    model: str,
    title: str,
    text: str,
    timeout: int = 60,
    max_retries: int = 6,
    backoff_base: float = 2.0,
) -> str:
    if not text.strip():
        return ""

    excerpt = text[:12000]
    allowed = ["politics", "economics", "autobiography", "history", "science", "psychology", "religion"]
    prompt = (
        "Choose exactly one theme label for this essay from this list only:\n"
        "politics, economics, autobiography, history, science, psychology, religion.\n"
        "Return only one lowercase label, nothing else.\n\n"
        f"Title: {title}\n\n"
        f"Article:\n{excerpt}"
    )

    payload: dict[str, Any] | None = None
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            res = requests_module.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": 20,
                },
                timeout=timeout,
            )
            status = res.status_code
            if status == 429 or status >= 500:
                if attempt >= max_retries:
                    res.raise_for_status()
                retry_after = _retry_after_seconds(res)
                wait = retry_after if retry_after is not None else backoff_base * (2**attempt)
                wait = min(wait, 90.0)
                print(
                    f"  Theme rate/server limit (HTTP {status}); retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            res.raise_for_status()
            payload = res.json()
            break
        except requests_module.exceptions.RequestException as e:
            last_error = format_request_exception(e)
            if attempt >= max_retries:
                raise
            wait = min(backoff_base * (2**attempt), 90.0)
            print(
                f"  Theme request error ({last_error}); retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)

    if payload is None:
        raise RuntimeError(last_error or "Theme request failed without a response payload")

    output_text = ""
    if isinstance(payload.get("output_text"), str):
        output_text = payload["output_text"].strip().lower()
    else:
        parts: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        output_text = " ".join(parts).strip().lower()

    for label in allowed:
        if re.search(rf"\b{re.escape(label)}\b", output_text):
            return label
    return ""


def normalize_subtheme_word(raw: str) -> str:
    token = raw.strip().lower()
    token = re.sub(r"^\s*(?:[-*•]|\d+[.)-]?)\s*", "", token)
    token = token.strip(" \t\r\n\"'`[](){}")
    token = token.replace("&", "and")
    if " " in token:
        return ""
    if not re.fullmatch(r"[a-z][a-z0-9-]*", token):
        return ""
    return token


def parse_subthemes_text(raw: str) -> list[str]:
    pieces = re.split(r"[,\n;|/]+", raw)
    out: list[str] = []
    seen = set()
    for piece in pieces:
        token = normalize_subtheme_word(piece)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def parse_subthemes_cell(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return parse_subthemes_text(raw)


def format_subthemes(words: list[str]) -> str:
    return ", ".join(words)


def dedupe_subtheme_words(words: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for raw in words:
        token = normalize_subtheme_word(raw)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def classify_subthemes_with_openai(
    requests_module: Any,
    api_key: str,
    model: str,
    theme: str,
    title: str,
    text: str,
    disallowed_words: set[str] | None = None,
    timeout: int = 60,
    max_retries: int = 6,
    backoff_base: float = 2.0,
) -> list[str]:
    if not text.strip() or not theme.strip():
        return []

    banned = sorted({w for w in (disallowed_words or set()) if w})
    banned_text = ", ".join(banned) if banned else "none"
    excerpt = text[:12000]
    prompt = (
        f"Create 3-5 subthemes for the theme '{theme}'.\n"
        "Rules:\n"
        "- Each subtheme must be one lowercase word.\n"
        "- Output exactly 3 to 5 unique words.\n"
        "- No repeats.\n"
        "- Do not use words from this forbidden list: "
        f"{banned_text}.\n"
        f"- Do not use the parent theme word '{theme}'.\n"
        "- Return only a comma-separated list of words. No numbering, no explanation.\n\n"
        f"Title: {title}\n\n"
        f"Article:\n{excerpt}"
    )

    payload: dict[str, Any] | None = None
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            res = requests_module.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": 80,
                },
                timeout=timeout,
            )
            status = res.status_code
            if status == 429 or status >= 500:
                if attempt >= max_retries:
                    res.raise_for_status()
                retry_after = _retry_after_seconds(res)
                wait = retry_after if retry_after is not None else backoff_base * (2**attempt)
                wait = min(wait, 90.0)
                print(
                    f"  Subtheme rate/server limit (HTTP {status}); retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            res.raise_for_status()
            payload = res.json()
            break
        except requests_module.exceptions.RequestException as e:
            last_error = format_request_exception(e)
            if attempt >= max_retries:
                raise
            wait = min(backoff_base * (2**attempt), 90.0)
            print(
                f"  Subtheme request error ({last_error}); retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)

    if payload is None:
        raise RuntimeError(last_error or "Subtheme request failed without a response payload")

    output_text = ""
    if isinstance(payload.get("output_text"), str):
        output_text = payload["output_text"].strip()
    else:
        parts: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        output_text = " ".join(parts).strip()

    forbidden = set(disallowed_words or set())
    forbidden.add(theme.strip().lower())
    parsed: list[str] = []
    seen = set()
    for token in parse_subthemes_text(output_text):
        if token in forbidden or token in seen:
            continue
        seen.add(token)
        parsed.append(token)
        if len(parsed) >= 5:
            break

    if len(parsed) < 3:
        # Fallback: derive a few candidate words from title/content if model output is malformed.
        stopwords = {
            "the",
            "and",
            "with",
            "from",
            "that",
            "this",
            "what",
            "when",
            "where",
            "which",
            "about",
            "essay",
            "article",
            "story",
            "into",
            "over",
            "under",
            "their",
            "there",
            "then",
            "than",
            "have",
            "has",
            "had",
            "will",
            "would",
            "should",
            "could",
        }
        for word in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", f"{title} {excerpt}".lower()):
            token = normalize_subtheme_word(word)
            if not token:
                continue
            if token in stopwords or token in forbidden or token in seen:
                continue
            seen.add(token)
            parsed.append(token)
            if len(parsed) >= 3:
                break

    return parsed[:5]


def choose_subtheme_heuristic(candidates: list[str], title: str, text: str) -> str:
    if not candidates:
        return ""
    haystack = f"{title}\n{text[:12000]}".lower()
    scored: list[tuple[int, str]] = []
    for word in candidates:
        count = len(re.findall(rf"\b{re.escape(word)}\b", haystack))
        scored.append((count, word))
    best_count = max(count for count, _ in scored)
    if best_count > 0:
        best_words = [word for count, word in scored if count == best_count]
        return sorted(best_words)[0]

    # If none of the labels appears literally in the text, choose a stable fallback.
    digest = hashlib.sha1(f"{title}\n{text[:2000]}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(candidates)
    return candidates[idx]


def best_quote_with_openai(
    requests_module: Any,
    api_key: str,
    model: str,
    title: str,
    text: str,
    timeout: int = 60,
    max_retries: int = 6,
    backoff_base: float = 2.0,
) -> str:
    if not text.strip():
        return ""

    excerpt = text[:16000]
    prompt = (
        "Select the best quote from the article.\n"
        "Return exactly one quote, not multiple options.\n"
        "Prefer 1-2 sentences; use 3 only if absolutely necessary.\n"
        "Quote must be drawn directly from article text, word-for-word.\n"
        "Do not output lists, bullets, numbering, commentary, or explanations.\n"
        "Do not output comma-separated lists of points.\n"
        "If your initial selection is longer, return only the strongest contiguous 1-3 sentence span.\n"
        "Pick lines that are unique, engaging, or thought-provoking, and still on-topic.\n"
        "Return only the quote text.\n\n"
        f"Title: {title}\n\n"
        f"Article:\n{excerpt}"
    )

    payload: dict[str, Any] | None = None
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            res = requests_module.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": 220,
                },
                timeout=timeout,
            )
            status = res.status_code
            if status == 429 or status >= 500:
                if attempt >= max_retries:
                    res.raise_for_status()
                retry_after = _retry_after_seconds(res)
                wait = retry_after if retry_after is not None else backoff_base * (2**attempt)
                wait = min(wait, 90.0)
                print(
                    f"  Quote rate/server limit (HTTP {status}); retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            res.raise_for_status()
            payload = res.json()
            break
        except requests_module.exceptions.RequestException as e:
            last_error = format_request_exception(e)
            if attempt >= max_retries:
                raise
            wait = min(backoff_base * (2**attempt), 90.0)
            print(
                f"  Quote request error ({last_error}); retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)

    if payload is None:
        raise RuntimeError(last_error or "Quote request failed without a response payload")

    if isinstance(payload.get("output_text"), str):
        return clean_best_quote(payload["output_text"])

    texts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    return clean_best_quote("\n".join(texts))


def clean_best_quote(raw: str) -> str:
    text = raw.strip()
    # Keep only first non-empty line to avoid multi-option outputs.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    text = lines[0]

    # Remove leading list markers.
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", text)

    # Split into sentences and cap at 3.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text.strip()
    return " ".join(sentences[:3]).strip()


def normalize_quote_text(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_quote_match(text: str) -> str:
    text = normalize_quote_text(text).lower()
    # Ignore punctuation differences to reduce false negatives.
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def quote_is_in_article(quote: str, article_text: str) -> bool:
    quote_n = normalize_quote_text(quote)
    article_n = normalize_quote_text(article_text)
    if not quote_n:
        return False
    if quote_n in article_n:
        return True

    quote_relaxed = normalize_for_quote_match(quote_n)
    article_relaxed = normalize_for_quote_match(article_n)
    if quote_relaxed and quote_relaxed in article_relaxed:
        return True

    # Extra fallback for tiny spacing/tokenization differences.
    quote_tokens = quote_relaxed.split()
    if len(quote_tokens) >= 8:
        anchor = " ".join(quote_tokens[:8])
        return anchor in article_relaxed
    return False


def load_successful_urls(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    seen: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url = (row.get("url") or "").strip()
            status = (row.get("status") or "").strip().lower()
            if url and status == "ok":
                seen.add(url)
    return seen


def ensure_csv_header(csv_path: Path) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_record(csv_path: Path, record: PostRecord) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writerow(record_to_row_dict(record))


def _fallback_post_label(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[-2] == "p":
        return parts[-1]
    if parts:
        return parts[-1]
    return parsed.netloc


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _unescape_dot(value: str) -> str:
    return value.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def _edge_details_to_labels(
    edge_details: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for edge_key, details in edge_details.items():
        labels = dedupe_keep_order([clean_anchor_text(d.get("anchor", "")) for d in details if d.get("anchor", "").strip()])
        out[edge_key] = labels if labels else ["(no anchor text)"]
    return out


def write_link_graph_dot(
    output_path: Path,
    node_titles: dict[str, str],
    edge_details: dict[tuple[str, str], list[dict[str, str]]],
) -> None:
    edge_labels = _edge_details_to_labels(edge_details)
    urls = sorted(node_titles)
    node_ids = {url: f"n{idx}" for idx, url in enumerate(urls)}

    lines: list[str] = [
        "digraph SubstackInternalLinks {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded"];',
    ]
    for url in urls:
        label = node_titles.get(url, "").strip() or _fallback_post_label(url)
        full_label = f"{label}\\n{url}"
        lines.append(f'  {node_ids[url]} [label="{_escape_dot(full_label)}"];')

    for (source, target), labels in sorted(edge_labels.items()):
        if source not in node_ids or target not in node_ids:
            continue
        combined = " | ".join(labels)
        lines.append(
            f'  {node_ids[source]} -> {node_ids[target]} [label="{_escape_dot(combined)}"];'
        )

    lines.append("}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_link_graph_json(
    output_path: Path,
    node_titles: dict[str, str],
    node_dates: dict[str, str],
    edge_details: dict[tuple[str, str], list[dict[str, str]]],
) -> None:
    payload = {
        "nodes": [
            {
                "url": url,
                "title": (node_titles.get(url, "").strip() or _fallback_post_label(url)),
                "publication_date": (node_dates.get(url) or "").strip(),
            }
            for url in sorted(node_titles)
        ],
        "edges": [
            {"source": source, "target": target, "details": details}
            for (source, target), details in sorted(edge_details.items())
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_link_graph_json(
    input_path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[tuple[str, str], list[dict[str, str]]]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid graph JSON (root object expected): {input_path}")

    node_titles: dict[str, str] = {}
    node_dates: dict[str, str] = {}
    edge_details: dict[tuple[str, str], list[dict[str, str]]] = {}

    for item in payload.get("nodes", []):
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip() if isinstance(item.get("url"), str) else ""
        title = (item.get("title") or "").strip() if isinstance(item.get("title"), str) else ""
        publication_date = (
            (item.get("publication_date") or "").strip() if isinstance(item.get("publication_date"), str) else ""
        )
        if url:
            node_titles[url] = title or _fallback_post_label(url)
            if publication_date:
                node_dates[url] = publication_date

    for item in payload.get("edges", []):
        if not isinstance(item, dict):
            continue
        source = (item.get("source") or "").strip() if isinstance(item.get("source"), str) else ""
        target = (item.get("target") or "").strip() if isinstance(item.get("target"), str) else ""
        details_raw = item.get("details")
        labels_raw = item.get("labels")
        if not source or not target:
            continue
        details: list[dict[str, str]] = []
        if isinstance(details_raw, list):
            for detail in details_raw:
                if not isinstance(detail, dict):
                    continue
                anchor = clean_anchor_text(str(detail.get("anchor") or ""), max_chars=200)
                before = " ".join(str(detail.get("before") or "").split())
                after = " ".join(str(detail.get("after") or "").split())
                if anchor:
                    details.append({"anchor": anchor, "before": before, "after": after})
        elif isinstance(labels_raw, list):
            for label in labels_raw:
                anchor = clean_anchor_text(str(label or ""), max_chars=200)
                if anchor:
                    details.append({"anchor": anchor, "before": "", "after": ""})

        if not details:
            details = [{"anchor": "(no anchor text)", "before": "", "after": ""}]
        edge_details[(source, target)] = details

    for source, target in edge_details:
        node_titles.setdefault(source, _fallback_post_label(source))
        node_titles.setdefault(target, _fallback_post_label(target))

    return node_titles, node_dates, edge_details


def load_link_graph_dot(
    input_path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[tuple[str, str], list[dict[str, str]]]]:
    text = input_path.read_text(encoding="utf-8")
    node_id_to_url: dict[str, str] = {}
    node_titles: dict[str, str] = {}
    edge_details: dict[tuple[str, str], list[dict[str, str]]] = {}

    node_pattern = re.compile(r'^\s*(n\d+)\s+\[label="((?:\\.|[^"])*)"\];\s*$')
    edge_pattern = re.compile(r'^\s*(n\d+)\s*->\s*(n\d+)\s+\[label="((?:\\.|[^"])*)"\];\s*$')
    for line in text.splitlines():
        node_match = node_pattern.match(line)
        if node_match:
            node_id = node_match.group(1)
            label = _unescape_dot(node_match.group(2))
            if "\n" in label:
                title, url = label.split("\n", 1)
            else:
                title = label
                url = ""
            title = title.strip()
            url = url.strip()
            if url:
                node_id_to_url[node_id] = url
                node_titles[url] = title or _fallback_post_label(url)
            continue

        edge_match = edge_pattern.match(line)
        if edge_match:
            src_id = edge_match.group(1)
            dst_id = edge_match.group(2)
            label = _unescape_dot(edge_match.group(3))
            source = node_id_to_url.get(src_id)
            target = node_id_to_url.get(dst_id)
            if not source or not target:
                continue
            labels = [clean_anchor_text(part.strip()) for part in label.split("|") if part.strip()]
            deduped = dedupe_keep_order(labels) if labels else ["(no anchor text)"]
            edge_details[(source, target)] = [{"anchor": val, "before": "", "after": ""} for val in deduped]

    for source, target in edge_details:
        node_titles.setdefault(source, _fallback_post_label(source))
        node_titles.setdefault(target, _fallback_post_label(target))
    if not node_titles:
        raise SystemExit(f"No graph nodes parsed from DOT: {input_path}")
    return node_titles, {}, edge_details


def _parse_publication_date(value: str) -> datetime | None:
    raw = value.strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    h = h % 360.0
    s = max(0.0, min(1.0, s))
    v = max(0.0, min(1.0, v))
    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c

    if h < 60:
        rp, gp, bp = c, x, 0.0
    elif h < 120:
        rp, gp, bp = x, c, 0.0
    elif h < 180:
        rp, gp, bp = 0.0, c, x
    elif h < 240:
        rp, gp, bp = 0.0, x, c
    elif h < 300:
        rp, gp, bp = x, 0.0, c
    else:
        rp, gp, bp = c, 0.0, x

    r = int(round((rp + m) * 255))
    g = int(round((gp + m) * 255))
    b = int(round((bp + m) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def load_vis_network_js() -> str:
    candidate_paths = [
        Path(__file__).resolve().parent / "vis-network.min.js",
        Path.cwd() / "vis-network.min.js",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")

    try:
        with urlopen(VIS_NETWORK_CDN_URL, timeout=25) as resp:
            payload = resp.read().decode("utf-8")
    except URLError as exc:
        raise SystemExit(
            "Unable to download vis-network JS for standalone HTML. "
            "Connect to the internet once, or place vis-network.min.js next to substack_exporter.py."
        ) from exc

    target = Path(__file__).resolve().parent / "vis-network.min.js"
    target.write_text(payload, encoding="utf-8")
    return payload


def write_link_graph_html(
    output_path: Path,
    node_titles: dict[str, str],
    node_dates: dict[str, str],
    edge_details: dict[tuple[str, str], list[dict[str, str]]],
    standalone: bool = False,
) -> None:
    edge_labels = _edge_details_to_labels(edge_details)
    indegree: dict[str, int] = {url: 0 for url in node_titles}
    outdegree: dict[str, int] = {url: 0 for url in node_titles}
    for source, target in edge_details:
        indegree[target] = indegree.get(target, 0) + 1
        outdegree[source] = outdegree.get(source, 0) + 1
        indegree.setdefault(source, indegree.get(source, 0))
        outdegree.setdefault(target, outdegree.get(target, 0))

    dated_ordinals: list[float] = []
    date_ord_by_url: dict[str, float] = {}
    date_label_by_url: dict[str, str] = {}
    for url, raw_date in node_dates.items():
        dt = _parse_publication_date(raw_date)
        if dt is None:
            continue
        ord_val = float(dt.timestamp())
        date_ord_by_url[url] = ord_val
        date_label_by_url[url] = dt.strftime("%Y-%m-%d")
        dated_ordinals.append(ord_val)
    min_ord = min(dated_ordinals) if dated_ordinals else 0.0
    max_ord = max(dated_ordinals) if dated_ordinals else 0.0
    oldest_date = ""
    newest_date = ""
    if date_ord_by_url:
        oldest_url = min(date_ord_by_url, key=date_ord_by_url.get)
        newest_url = max(date_ord_by_url, key=date_ord_by_url.get)
        oldest_date = date_label_by_url.get(oldest_url, "")
        newest_date = date_label_by_url.get(newest_url, "")

    nodes: list[dict[str, Any]] = []
    for url in sorted(node_titles):
        title = node_titles.get(url, "").strip() or _fallback_post_label(url)
        short = title if len(title) <= 55 else (title[:52].rstrip() + "...")
        in_count = indegree.get(url, 0)
        out_count = outdegree.get(url, 0)
        # Scale node area and font from inbound links with a minimum size that fits short labels.
        base_width = min(430, max(160, 10 * len(short) + 36))
        area_scale = 1.0 + (0.18 * ((1.24**min(in_count, 16)) - 1.0))
        width = min(860, int(round(base_width * area_scale)))
        base_height = 38
        height = min(160, int(round(base_height * (area_scale**0.5))))
        font_size = min(34, 12 + int(round(3.2 * ((1.32**min(in_count, 12)) - 1.0))))
        date_str = (node_dates.get(url) or "").strip()
        if url in date_ord_by_url and max_ord > min_ord:
            t = (date_ord_by_url[url] - min_ord) / (max_ord - min_ord)
            # Sweep hue across the HSV wheel from orange (oldest) to blue (newest).
            hue = 36.0 + (220.0 - 36.0) * t
            fill = _hsv_to_hex(hue, 0.4, 0.9)
            border = _hsv_to_hex(hue, 0.4, 0.5)
        elif url in date_ord_by_url:
            fill = _hsv_to_hex(220.0, 0.4, 0.9)
            border = _hsv_to_hex(220.0, 0.4, 0.5)
        else:
            fill = "#e4e7ec"
            border = "#98a2b3"
        nodes.append(
            {
                "id": url,
                "label": short,
                "title": f"{title}\n{url}\nInbound: {in_count}, Outbound: {out_count}",
                "publicationDate": date_str,
                "widthConstraint": {"minimum": width},
                "heightConstraint": {"minimum": height},
                "font": {"size": font_size, "color": "#101828"},
                "borderWidth": 1.5,
                "borderWidthSelected": 2.5,
                "color": {
                    "background": fill,
                    "border": border,
                    "highlight": {"background": fill, "border": border},
                    "hover": {"background": fill, "border": border},
                },
            }
        )

    edges: list[dict[str, Any]] = []
    for (source, target), labels in sorted(edge_labels.items()):
        combined = " | ".join(labels)
        if len(combined) > 180:
            combined = combined[:177].rstrip() + "..."
        details = edge_details.get((source, target), [])
        edges.append(
            {
                "from": source,
                "to": target,
                "arrows": "to",
                "label": combined,
                "title": combined,
                "anchorLabels": labels,
                "details": details,
            }
        )

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    legend_oldest = oldest_date or "oldest"
    legend_newest = newest_date or "newest"
    legend_visible = "true" if (oldest_date and newest_date and oldest_date != newest_date) else "false"
    vis_script_tag = (
        "<script>\n" + load_vis_network_js() + "\n</script>"
        if standalone
        else f'<script src="{VIS_NETWORK_CDN_URL}"></script>'
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Substack Internal Link Graph</title>
  {vis_script_tag}
  <style>
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: #f7f8fa;
      color: #222;
      overflow: hidden;
    }}
    #graph {{
      width: 100vw;
      height: 100vh;
      border: 0;
    }}
    .meta {{
      position: fixed;
      top: 10px;
      left: 10px;
      z-index: 10;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid #d5d8dd;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
    }}
    #legend {{
      position: fixed;
      top: 56px;
      left: 10px;
      z-index: 10;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid #d5d8dd;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 11px;
      color: #344054;
      display: none;
    }}
    #legend .title {{
      font-weight: 600;
      margin-bottom: 6px;
    }}
    #legend .bar {{
      width: 220px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid #cfd4dc;
      background: linear-gradient(
        90deg,
        hsl(36 40% 90%),
        hsl(60 40% 90%),
        hsl(120 40% 90%),
        hsl(180 40% 90%),
        hsl(220 40% 90%)
      );
    }}
    #legend .labels {{
      display: flex;
      justify-content: space-between;
      margin-top: 4px;
      gap: 12px;
    }}
    #sidepanel {{
      display: none;
      position: fixed;
      top: 56px;
      right: 14px;
      z-index: 20;
      width: min(38vw, 560px);
      min-width: 320px;
      max-height: calc(100vh - 76px);
      border: 1px solid #d5d8dd;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 8px 24px rgba(16, 24, 40, 0.16);
      padding: 14px;
      box-sizing: border-box;
      overflow: auto;
    }}
    #sidepanel h2 {{
      margin: 0 0 8px 0;
      font-size: 14px;
      line-height: 1.3;
    }}
    #sidepanel .date {{
      margin: -6px 0 12px 0;
      font-size: 12px;
      color: #475467;
    }}
    #sidepanel h3 {{
      margin: 14px 0 8px 0;
      font-size: 13px;
    }}
    #sidepanel ol {{
      margin: 0;
      padding-left: 18px;
      font-size: 12px;
      line-height: 1.45;
    }}
    #sidepanel li {{
      margin: 0 0 6px 0;
    }}
    #sidepanel .hint {{
      color: #667085;
      font-size: 12px;
      margin-top: 8px;
    }}
    #sidepanel .ctx {{
      color: #475467;
      font-size: 12px;
    }}
    #sidepanel .ctx strong {{
      color: #101828;
      font-weight: 700;
    }}
    #sidepanel .ctx-block {{
      margin-top: 4px;
    }}
    #sidepanel a.node-title {{
      color: #101828;
      text-decoration: none;
      border-bottom: 1px solid rgba(16, 24, 40, 0.25);
    }}
    #sidepanel a.node-title:hover {{
      border-bottom-color: rgba(16, 24, 40, 0.75);
    }}
    #sidepanel a.node-jump {{
      color: #175cd3;
      text-decoration: underline;
      cursor: pointer;
    }}
    #sidepanel .close {{
      float: right;
      border: 1px solid #d0d5dd;
      border-radius: 8px;
      background: #fff;
      color: #344054;
      font-size: 11px;
      padding: 2px 7px;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <div class="meta">Node size reflects inbound links. Click a node for its local link map.</div>
  <button id="physics-toggle" class="meta" style="top: 10px; left: 420px; cursor: pointer;">Pause Physics</button>
  <div id="legend">
    <div class="title">Publication Date Scale</div>
    <div class="bar"></div>
    <div class="labels">
      <span id="legend-oldest">{legend_oldest}</span>
      <span id="legend-newest">{legend_newest}</span>
    </div>
  </div>
  <div id="graph"></div>
  <aside id="sidepanel"></aside>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const container = document.getElementById("graph");
    const panel = document.getElementById("sidepanel");
    const legend = document.getElementById("legend");
    const physicsToggle = document.getElementById("physics-toggle");
    const data = {{ nodes, edges }};
    const options = {{
      nodes: {{
        shape: "box",
        margin: 8,
        borderWidth: 1,
        font: {{ size: 12 }},
      }},
      edges: {{
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
        smooth: {{ type: "dynamic" }},
        color: {{
          color: "rgba(71, 84, 103, 0.6)",
          highlight: "rgba(17, 24, 39, 1.0)",
          hover: "rgba(17, 24, 39, 0.9)",
        }},
        font: {{ size: 10, align: "middle", color: "rgba(71, 84, 103, 0.6)" }},
        selectionWidth: 2,
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 100,
      }},
      physics: {{
        barnesHut: {{
          gravitationalConstant: -9200,
          springLength: 300,
          springConstant: 0.012,
          avoidOverlap: 0.72,
        }},
        maxVelocity: 20,
        minVelocity: 0.25,
        stabilization: {{
          enabled: true,
          iterations: 320,
          updateInterval: 25,
          fit: true,
        }},
      }},
    }};
    const network = new vis.Network(container, data, options);
    let physicsEnabled = true;
    function setPhysicsEnabled(enabled) {{
      physicsEnabled = !!enabled;
      if (physicsEnabled) {{
        network.setOptions({{
          physics: {{
            enabled: true,
            stabilization: {{ enabled: false }},
          }},
        }});
        physicsToggle.textContent = "Pause Physics";
      }} else {{
        network.setOptions({{ physics: false }});
        physicsToggle.textContent = "Play Physics";
      }}
    }}
    physicsToggle.addEventListener("click", function () {{
      setPhysicsEnabled(!physicsEnabled);
    }});
    network.once("stabilizationIterationsDone", function () {{
      // Keep a short post-stabilization window for macro-cluster settling, then freeze.
      setTimeout(function () {{
        setPhysicsEnabled(false);
      }}, 5000);
    }});
    if ({legend_visible}) {{
      legend.style.display = "block";
    }}

    function nodeLabel(nodeId) {{
      const node = nodes.get(nodeId);
      if (!node) return nodeId;
      return node.label || nodeId;
    }}

    function escapeHtml(text) {{
      return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function edgeDetails(edgeObj) {{
      if (Array.isArray(edgeObj.details) && edgeObj.details.length > 0) {{
        return edgeObj.details;
      }}
      if (Array.isArray(edgeObj.anchorLabels) && edgeObj.anchorLabels.length > 0) {{
        return edgeObj.anchorLabels.map((anchor) => ({{ anchor: anchor, before: "", after: "" }}));
      }}
      return [{{ anchor: edgeObj.label || "(no anchor text)", before: "", after: "" }}];
    }}

    function contextHtml(detail) {{
      const before = (detail.before || "").trim();
      const anchor = (detail.anchor || "").trim() || "(no anchor text)";
      const after = (detail.after || "").trim();
      const beforePart = before ? "..." + escapeHtml(before) + " " : "";
      const afterPart = after ? " " + escapeHtml(after) + "..." : "";
      return "<span class='ctx'>" + beforePart + "<strong>" + escapeHtml(anchor) + "</strong>" + afterPart + "</span>";
    }}

    function formatDateLong(dateStr) {{
      const raw = (dateStr || "").trim();
      if (!raw) return "";
      const m = raw.match(/^(\\d{{4}})-(\\d{{2}})-(\\d{{2}})$/);
      if (!m) return raw;
      const year = Number(m[1]);
      const month = Number(m[2]) - 1;
      const day = Number(m[3]);
      const dt = new Date(Date.UTC(year, month, day));
      const monthName = dt.toLocaleString("en-US", {{ month: "long", timeZone: "UTC" }});
      let suffix = "th";
      if (day % 100 < 11 || day % 100 > 13) {{
        if (day % 10 === 1) suffix = "st";
        else if (day % 10 === 2) suffix = "nd";
        else if (day % 10 === 3) suffix = "rd";
      }}
      return monthName + " " + day + suffix + ", " + year;
    }}

    function sortableDateValue(dateStr) {{
      const raw = (dateStr || "").trim();
      const m = raw.match(/^(\\d{{4}})-(\\d{{2}})-(\\d{{2}})$/);
      if (!m) return Number.POSITIVE_INFINITY;
      return Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
    }}

    function renderList(items) {{
      const grouped = new Map();
      for (const item of items) {{
        const key = String(item.otherNodeId || "");
        if (!grouped.has(key)) {{
          grouped.set(key, {{
            otherNodeId: item.otherNodeId,
            otherLabel: item.otherLabel,
            otherDate: item.otherDate || "",
            details: [],
            _seenContexts: new Set(),
          }});
        }}
        const bucket = grouped.get(key);
        const ctxKey = [
          (item.detail && item.detail.before) || "",
          (item.detail && item.detail.anchor) || "",
          (item.detail && item.detail.after) || "",
        ].join("|||");
        if (!bucket._seenContexts.has(ctxKey)) {{
          bucket._seenContexts.add(ctxKey);
          bucket.details.push(item.detail || {{ anchor: "(no anchor text)", before: "", after: "" }});
        }}
      }}

      const ordered = Array.from(grouped.values()).sort((a, b) => {{
        const da = sortableDateValue(a.otherDate);
        const db = sortableDateValue(b.otherDate);
        if (da !== db) return da - db;
        const la = (a.otherLabel || "").toLowerCase();
        const lb = (b.otherLabel || "").toLowerCase();
        if (la < lb) return -1;
        if (la > lb) return 1;
        return 0;
      }});

      let html = "<ol>";
      for (const item of ordered) {{
        html += "<li>" +
          "<a href='#' class='node-jump' data-node-id='" + escapeHtml(item.otherNodeId) + "'>" +
          escapeHtml(item.otherLabel) + "</a>";
        if (item.otherDate) {{
          html += " <span class='hint'>(" + escapeHtml(formatDateLong(item.otherDate)) + ")</span>";
        }}
        for (const detail of item.details) {{
          html += "<div class='ctx-block'>" + contextHtml(detail) + "</div>";
        }}
        html += "</li>";
      }}
      html += "</ol>";
      return html;
    }}

    function setPanelHidden() {{
      panel.style.display = "none";
      panel.innerHTML = "";
    }}

    function setPanelVisible() {{
      panel.style.display = "block";
    }}

    function applyEdgeVisualState(activeEdgeIds) {{
      const active = new Set(activeEdgeIds || []);
      const all = edges.get();
      const updates = [];
      for (const e of all) {{
        const isActive = active.has(e.id);
        updates.push({{
          id: e.id,
          color: {{
            color: isActive ? "rgba(17, 24, 39, 1.0)" : "rgba(71, 84, 103, 0.6)",
            highlight: "rgba(17, 24, 39, 1.0)",
            hover: "rgba(17, 24, 39, 0.9)",
          }},
          font: {{
            color: isActive ? "rgba(17, 24, 39, 1.0)" : "rgba(71, 84, 103, 0.6)",
            size: 10,
            align: "middle",
          }},
        }});
      }}
      edges.update(updates);
    }}

    function renderPanel(nodeId) {{
      const node = nodes.get(nodeId);
      if (!node) return;
      const connectedEdgeIds = network.getConnectedEdges(nodeId);
      const inbound = [];
      const outbound = [];
      for (const edgeId of connectedEdgeIds) {{
        const edge = edges.get(edgeId);
        if (!edge) continue;
        const detailList = edgeDetails(edge);
        if (edge.from === nodeId) {{
          for (const detail of detailList) {{
            const targetNode = nodes.get(edge.to);
            outbound.push({{
              otherNodeId: edge.to,
              otherLabel: nodeLabel(edge.to),
              otherDate: targetNode && targetNode.publicationDate ? targetNode.publicationDate : "",
              detail: detail,
            }});
          }}
        }}
        if (edge.to === nodeId) {{
          for (const detail of detailList) {{
            const sourceNode = nodes.get(edge.from);
            inbound.push({{
              otherNodeId: edge.from,
              otherLabel: nodeLabel(edge.from),
              otherDate: sourceNode && sourceNode.publicationDate ? sourceNode.publicationDate : "",
              detail: detail,
            }});
          }}
        }}
      }}

      const dateHtml = node.publicationDate
        ? "<p class='date'>Published " + escapeHtml(formatDateLong(node.publicationDate)) + "</p>"
        : "<p class='date'>Published: unknown</p>";
      let body = "";
      if (inbound.length > 0) {{
        body += "<h3>Linked from</h3>" + renderList(inbound);
      }}
      if (outbound.length > 0) {{
        body += "<h3>Links to</h3>" + renderList(outbound);
      }}
      if (!body) {{
        body = "<p class='hint'>This post contains no links and was not linked to.</p>";
      }}

      panel.innerHTML =
        "<button class='close' id='panel-close'>Close</button>" +
        "<h2><a class='node-title' href='" + escapeHtml(node.id) + "' target='_blank' rel='noopener noreferrer'>" +
        escapeHtml(node.label || node.id) + "</a></h2>" +
        dateHtml +
        body;
      const closeBtn = document.getElementById("panel-close");
      if (closeBtn) {{
        closeBtn.addEventListener("click", function () {{
          setPanelHidden();
          applyEdgeVisualState([]);
          network.unselectAll();
        }});
      }}
      for (const jumpLink of panel.querySelectorAll("a.node-jump")) {{
        jumpLink.addEventListener("click", function (evt) {{
          evt.preventDefault();
          const targetId = jumpLink.getAttribute("data-node-id");
          if (!targetId) return;
          network.focus(targetId, {{
            scale: network.getScale(),
            animation: {{ duration: 300, easingFunction: "easeInOutQuad" }},
          }});
          network.selectNodes([targetId]);
          renderPanel(targetId);
        }});
      }}
      setPanelVisible();
      applyEdgeVisualState(connectedEdgeIds);
    }}

    network.on("click", function (params) {{
      if (params.nodes && params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        network.focus(nodeId, {{
          scale: network.getScale(),
          animation: {{ duration: 300, easingFunction: "easeInOutQuad" }},
        }});
        renderPanel(nodeId);
        return;
      }}
      if (params.edges && params.edges.length > 0) {{
        applyEdgeVisualState(params.edges);
        return;
      }}
      setPanelHidden();
      applyEdgeVisualState([]);
    }});

    network.on("deselectEdge", function () {{
      if (!network.getSelectedNodes().length) {{
        applyEdgeVisualState([]);
      }}
    }});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def run() -> int:
    args = parse_args()
    update_mode = bool(args.update_existing_csv)
    update_row_range = parse_row_range(args.update_row_range) if update_mode else None
    update_columns = parse_update_columns(args.update_columns) if update_mode else []
    update_source_path = Path(args.update_existing_csv) if update_mode else None
    render_from_json = bool(args.render_link_graph_from_json)
    render_from_dot = bool(args.render_link_graph_from_dot)
    if render_from_json and render_from_dot:
        raise SystemExit("Use only one of --render-link-graph-from-json or --render-link-graph-from-dot.")

    graph_requested = bool(args.link_graph_html or args.link_graph_dot or args.link_graph_json)
    graph_only = bool(args.link_graph_only)
    if graph_only and not graph_requested:
        raise SystemExit("Use --link-graph-html, --link-graph-dot, and/or --link-graph-json with --link-graph-only.")
    if graph_only and update_mode:
        raise SystemExit("--link-graph-only cannot be combined with --update-existing-csv.")
    if (render_from_json or render_from_dot) and update_mode:
        raise SystemExit("Render-from-file mode cannot be combined with --update-existing-csv.")
    if (render_from_json or render_from_dot) and not graph_requested:
        raise SystemExit("Render-from-file mode requires at least one output: --link-graph-html, --link-graph-dot, or --link-graph-json.")

    if render_from_json or render_from_dot:
        if render_from_json:
            node_titles, node_dates, edge_details = load_link_graph_json(Path(args.render_link_graph_from_json))
        else:
            node_titles, node_dates, edge_details = load_link_graph_dot(Path(args.render_link_graph_from_dot))

        if args.link_graph_dot:
            dot_path = Path(args.link_graph_dot)
            write_link_graph_dot(dot_path, node_titles, edge_details)
            print(f"Wrote link graph DOT: {dot_path}")
        if args.link_graph_json:
            json_path = Path(args.link_graph_json)
            write_link_graph_json(json_path, node_titles, node_dates, edge_details)
            print(f"Wrote link graph JSON: {json_path}")
        if args.link_graph_html:
            html_path = Path(args.link_graph_html)
            write_link_graph_html(
                html_path,
                node_titles,
                node_dates,
                edge_details,
                standalone=args.link_graph_html_standalone,
            )
            print(f"Wrote link graph HTML: {html_path}")
        print(f"Link graph stats: nodes={len(node_titles)}, edges={len(edge_details)}")
        return 0

    requests, BeautifulSoup = ensure_deps()

    # In update mode, default to in-place rewrite unless output path is explicitly set.
    if update_mode and args.output_csv == "substack_export.csv":
        out_path = update_source_path
    else:
        out_path = Path(args.output_csv)

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    enable_summaries = (not args.no_summaries) and (not graph_only)
    selected_variants = parse_summary_variants(args.summary_variants) if enable_summaries else set()
    if enable_summaries and not openai_api_key:
        print("OPENAI_API_KEY is not set; disabling summaries.", file=sys.stderr)
        enable_summaries = False

    session = make_session(requests, args.timeout, args.cookie_header, args.cookies_json)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    summary_source_csv = args.merge_summaries_from_csv
    if update_mode and not summary_source_csv:
        # In update mode, default to reading existing summary fields from the target CSV.
        summary_source_csv = args.update_existing_csv
    summary_merge_map = load_summary_map_from_csv(summary_source_csv) if summary_source_csv else {}

    post_urls: list[str] = []
    if update_mode:
        post_urls = select_urls_from_existing_csv(args.update_existing_csv, update_row_range)
        if args.max_posts > 0:
            post_urls = post_urls[: args.max_posts]
    elif args.post_url:
        post_urls = dedupe_keep_order([url.strip() for url in args.post_url if url.strip()])
    elif args.urls_from_csv:
        post_urls = load_urls_from_csv(args.urls_from_csv)
        if args.max_posts > 0:
            post_urls = post_urls[: args.max_posts]
    elif args.substack_url:
        base_url = canonical_base_url(args.substack_url)
        print(f"Fetching feed: {base_url}/feed")
        post_urls = fetch_feed_urls(session, base_url)
        needed = args.max_posts if args.max_posts > 0 else 0
        if needed == 0 or len(post_urls) < needed:
            print(f"Fetching archive API: {base_url}/api/v1/archive")
            archive_api_urls = fetch_archive_api_urls(
                session=session,
                base_url=base_url,
                limit=needed,
            )
            post_urls = dedupe_keep_order(post_urls + archive_api_urls)
        if needed == 0 or len(post_urls) < needed:
            print(f"Fetching archive pages: {base_url}/archive")
            archive_urls = fetch_archive_urls(
                session=session,
                base_url=base_url,
                bs4_class=BeautifulSoup,
                limit=needed,
            )
            post_urls = dedupe_keep_order(post_urls + archive_urls)
        if args.max_posts > 0:
            post_urls = post_urls[: args.max_posts]
    else:
        raise SystemExit("Provide --substack-url, --urls-from-csv, or at least one --post-url.")

    known_post_urls = {normalize_post_url(url) for url in post_urls}
    graph_dates: dict[str, str] = {}
    if graph_requested:
        if args.urls_from_csv:
            graph_dates.update(load_url_date_map_from_csv(args.urls_from_csv))
        elif update_mode:
            graph_dates.update(load_url_date_map_from_csv(args.update_existing_csv, row_range=update_row_range))
    print(f"Found {len(post_urls)} post(s).")

    if not update_mode and not graph_only:
        ensure_csv_header(out_path)
    seen_urls = load_successful_urls(out_path) if (args.resume and not update_mode and not graph_only) else set()
    generated_row_map: dict[str, dict[str, str]] = {}
    graph_titles: dict[str, str] = {}
    graph_edge_details: dict[tuple[str, str], list[dict[str, str]]] = {}
    theme_subthemes_map: dict[str, list[str]] = {}
    used_subtheme_words: set[str] = set()
    if summary_merge_map:
        for merged in summary_merge_map.values():
            merged_theme = (merged.get("theme") or "").strip().lower()
            merged_subthemes = parse_subthemes_cell((merged.get("subthemes") or "").strip())
            if not merged_theme or not merged_subthemes:
                continue
            if merged_theme not in theme_subthemes_map:
                theme_subthemes_map[merged_theme] = dedupe_subtheme_words(merged_subthemes[:5])
            used_subtheme_words.update(theme_subthemes_map[merged_theme])

    processed = 0
    skipped = 0
    failures = 0

    for idx, url in enumerate(post_urls, start=1):
        if (not update_mode) and url in seen_urls:
            skipped += 1
            print(f"[{idx}/{len(post_urls)}] Skipping existing URL: {url}")
            continue

        print(f"[{idx}/{len(post_urls)}] Processing: {url}")
        try:
            html: str | None = None
            if cache_dir and not args.refresh_cache:
                html = load_cached_html(cache_dir, url)
                if html is not None:
                    print("  Using cached HTML")

            if html is None:
                res = http_get_with_retries(
                    requests_module=requests,
                    session=session,
                    url=url,
                    max_retries=max(0, args.fetch_max_retries),
                    backoff_base=max(0.1, args.fetch_backoff_base),
                    label="Post fetch",
                )
                res.raise_for_status()
                html = res.text
                if cache_dir:
                    save_cached_html(cache_dir, url, html)

            soup = BeautifulSoup(html, "html.parser")

            title = extract_title(soup) or url
            if graph_requested:
                source_norm = normalize_post_url(url)
                graph_titles[source_norm] = title
                for link_item in extract_internal_post_links(
                    soup=soup,
                    bs4_class=BeautifulSoup,
                    source_url=url,
                    known_post_urls=known_post_urls,
                ):
                    target_url = link_item.get("target_url", "")
                    anchor_text = link_item.get("anchor", "")
                    before = link_item.get("before", "")
                    after = link_item.get("after", "")
                    if not target_url or not anchor_text:
                        continue
                    edge_key = (source_norm, target_url)
                    if edge_key not in graph_edge_details:
                        graph_edge_details[edge_key] = []
                    detail = {"anchor": anchor_text, "before": before, "after": after}
                    if detail not in graph_edge_details[edge_key]:
                        graph_edge_details[edge_key].append(detail)

                parsed_date = extract_publication_date(soup)
                if parsed_date and source_norm not in graph_dates:
                    graph_dates[source_norm] = parsed_date

            if graph_only:
                processed += 1
                time.sleep(args.sleep)
                continue

            publication_date = extract_publication_date(soup)
            likes, comments, restacks = parse_counts(soup, html, url)
            main_text, footnote_text, image_count = extract_article_parts(soup, BeautifulSoup)
            text = main_text if not footnote_text else f"{main_text}\n\nFootnotes:\n{footnote_text}"
            main_word_count = count_words(main_text)
            footnote_word_count = count_words(footnote_text)
            article_word_count = main_word_count + footnote_word_count
            paywalled = looks_paywalled(soup, html, text, article_word_count)

            summary_short_neutral = ""
            summary_short_mimic = ""
            summary_very_short_snappy = ""
            summary_dl = ""
            best_quote = ""
            best_quote_verified = ""
            summary_long = ""
            theme = ""
            subthemes = ""
            status = "ok"
            error = ""

            if args.require_full_text and paywalled:
                status = "auth_error"
                error = "Possible paywall detected; provide valid authenticated cookies."

            if enable_summaries and status == "ok":
                try:
                    if not args.no_theme:
                        theme = classify_theme_with_openai(
                            requests,
                            openai_api_key,
                            args.openai_model,
                            title,
                            text,
                            timeout=max(60, args.timeout),
                            max_retries=max(0, args.summary_max_retries),
                            backoff_base=max(0.1, args.summary_backoff_base),
                        )
                        theme = theme.strip().lower()
                        time.sleep(max(0.0, args.summary_sleep))
                        if theme and not args.no_subthemes:
                            subtheme_pool = theme_subthemes_map.get(theme, [])
                            if len(subtheme_pool) < 3:
                                try:
                                    generated_subthemes = classify_subthemes_with_openai(
                                        requests,
                                        openai_api_key,
                                        args.openai_model,
                                        theme,
                                        title,
                                        text,
                                        disallowed_words=used_subtheme_words,
                                        timeout=max(60, args.timeout),
                                        max_retries=max(0, args.summary_max_retries),
                                        backoff_base=max(0.1, args.summary_backoff_base),
                                    )
                                    if len(generated_subthemes) >= 3:
                                        subtheme_pool = dedupe_subtheme_words(generated_subthemes[:5])
                                        theme_subthemes_map[theme] = subtheme_pool
                                        used_subtheme_words.update(subtheme_pool)
                                    elif generated_subthemes and subtheme_pool:
                                        merged_pool = dedupe_subtheme_words(subtheme_pool + generated_subthemes)
                                        if merged_pool:
                                            subtheme_pool = merged_pool[:5]
                                            theme_subthemes_map[theme] = subtheme_pool
                                            used_subtheme_words.update(subtheme_pool)
                                except Exception as subtheme_err:  # noqa: BLE001
                                    print(f"  Subtheme pool error: {subtheme_err}", file=sys.stderr)
                                time.sleep(max(0.0, args.summary_sleep))
                            if subtheme_pool:
                                # One assigned subtheme per URL, selected from the theme's 3-5 word pool.
                                subthemes = choose_subtheme_heuristic(subtheme_pool, title, text)
                                time.sleep(max(0.0, args.summary_sleep))
                    if "summary_short_neutral" in selected_variants:
                        summary_short_neutral = summarize_with_openai(
                            requests,
                            openai_api_key,
                            args.openai_model,
                            title,
                            text,
                            word_limit=15,
                            target_words=10,
                            style_instructions=(
                                "Tone: neutral and plain, like answering 'what is this about?' "
                                "No hype or recommendation language. Prioritize one central idea over listing multiple conclusions. "
                                "Do not begin with 'the article' or 'this article'; jump directly into the idea."
                            ),
                            timeout=max(60, args.timeout),
                            max_retries=max(0, args.summary_max_retries),
                            backoff_base=max(0.1, args.summary_backoff_base),
                        )
                        time.sleep(max(0.0, args.summary_sleep))
                    if "summary_short_mimic" in selected_variants:
                        summary_short_mimic = summarize_with_openai(
                            requests,
                            openai_api_key,
                            args.openai_model,
                            title,
                            text,
                            word_limit=15,
                            target_words=10,
                            style_instructions=(
                                "Tone: mimic the article's voice and rhetoric. "
                                "If it is witty or acerbic, reflect that briefly. Prioritize one central idea over listing multiple conclusions. "
                                "Do not begin with 'the article' or 'this article'; jump directly into the idea."
                            ),
                            timeout=max(60, args.timeout),
                            max_retries=max(0, args.summary_max_retries),
                            backoff_base=max(0.1, args.summary_backoff_base),
                        )
                        time.sleep(max(0.0, args.summary_sleep))
                    if "summary_very_short_snappy" in selected_variants:
                        summary_very_short_snappy = summarize_with_openai(
                            requests,
                            openai_api_key,
                            args.openai_model,
                            title,
                            text,
                            word_limit=10,
                            target_words=5,
                            style_instructions=(
                                "Style: very short and snappy. Prioritize punch and clarity. "
                                "Focus on one big idea, not a list. Do not begin with 'the article' or 'this article'."
                            ),
                            timeout=max(60, args.timeout),
                            max_retries=max(0, args.summary_max_retries),
                            backoff_base=max(0.1, args.summary_backoff_base),
                        )
                        time.sleep(max(0.0, args.summary_sleep))
                    if "summary_dl" in selected_variants:
                        summary_dl = summarize_with_openai(
                            requests,
                            openai_api_key,
                            args.openai_model,
                            title,
                            text,
                            word_limit=10,
                            target_words=10,
                            style_instructions=(
                                "Output a ten word summary of the entire essay. In the summary, "
                                "have it be a single sentence with no clauses or commas (no lists). "
                                "Make sure that the summary contains no lists, no subordinate clauses, "
                                "no sentence fragments."
                            ),
                            include_default_content_guidance=False,
                            timeout=max(60, args.timeout),
                            max_retries=max(0, args.summary_max_retries),
                            backoff_base=max(0.1, args.summary_backoff_base),
                        )
                        time.sleep(max(0.0, args.summary_sleep))
                    if "best_quote" in selected_variants:
                        best_quote = best_quote_with_openai(
                            requests,
                            openai_api_key,
                            args.openai_model,
                            title,
                            text,
                            timeout=max(60, args.timeout),
                            max_retries=max(0, args.summary_max_retries),
                            backoff_base=max(0.1, args.summary_backoff_base),
                        )
                        best_quote_verified = "yes" if quote_is_in_article(best_quote, text) else "no"
                        time.sleep(max(0.0, args.summary_sleep))
                    if "summary_long" in selected_variants:
                        summary_long = summarize_with_openai(
                            requests,
                            openai_api_key,
                            args.openai_model,
                            title,
                            text,
                            word_limit=60,
                            target_words=50,
                            style_instructions="",
                            timeout=max(60, args.timeout),
                            max_retries=max(0, args.summary_max_retries),
                            backoff_base=max(0.1, args.summary_backoff_base),
                        )
                except Exception as e:  # noqa: BLE001
                    status = "summary_error"
                    error = f"summary: {e}"
                time.sleep(max(0.0, args.summary_sleep))

            if summary_merge_map:
                merged = summary_merge_map.get(url) or summary_merge_map.get(normalize_post_url(url))
                if merged:
                    if not theme:
                        theme = merged.get("theme", "").strip().lower()
                    if not subthemes:
                        parsed_merged_subthemes = parse_subthemes_cell(merged.get("subthemes", ""))
                        if parsed_merged_subthemes:
                            subthemes = parsed_merged_subthemes[0]
                    if not summary_short_neutral:
                        summary_short_neutral = merged.get("summary_short_neutral", "")
                    if not summary_short_mimic:
                        summary_short_mimic = merged.get("summary_short_mimic", "")
                    if not summary_very_short_snappy:
                        summary_very_short_snappy = merged.get("summary_very_short_snappy", "")
                    if not summary_dl:
                        summary_dl = merged.get("summary_dl", "")
                    if not best_quote:
                        best_quote = merged.get("best_quote", "")
                    if not best_quote_verified:
                        best_quote_verified = merged.get("best_quote_verified", "")
                    if not summary_long:
                        summary_long = merged.get("summary_long", "")
            if theme and subthemes:
                normalized_subtheme = normalize_subtheme_word(subthemes)
                if normalized_subtheme:
                    subthemes = normalized_subtheme
                    if theme not in theme_subthemes_map:
                        theme_subthemes_map[theme] = [normalized_subtheme]
                    elif normalized_subtheme not in theme_subthemes_map[theme] and len(theme_subthemes_map[theme]) < 5:
                        theme_subthemes_map[theme].append(normalized_subtheme)
                    used_subtheme_words.add(normalized_subtheme)
            if theme and not subthemes and theme in theme_subthemes_map and theme_subthemes_map[theme]:
                subthemes = choose_subtheme_heuristic(theme_subthemes_map[theme], title, text)

            # Credit-free verification mode: if we're updating verification and a quote exists,
            # recompute best_quote_verified from existing quote text without calling the LLM.
            if (
                best_quote
                and not best_quote_verified
                and (
                    (update_mode and "best_quote_verified" in update_columns)
                    or ("best_quote_verified" in selected_variants)
                )
            ):
                best_quote_verified = "yes" if quote_is_in_article(best_quote, text) else "no"

            record = PostRecord(
                title=title,
                url=url,
                publication_date=publication_date,
                likes=likes,
                comments=comments,
                restacks=restacks,
                article_word_count=article_word_count,
                main_word_count=main_word_count,
                footnote_word_count=footnote_word_count,
                image_count=image_count,
                theme=theme,
                subthemes=subthemes,
                summary_short_neutral=summary_short_neutral,
                summary_short_mimic=summary_short_mimic,
                summary_very_short_snappy=summary_very_short_snappy,
                summary_dl=summary_dl,
                best_quote=best_quote,
                best_quote_verified=best_quote_verified,
                summary_long=summary_long,
                status=status,
                error=error,
            )
            if update_mode:
                generated_row_map[url] = record_to_row_dict(record)
            else:
                append_record(out_path, record)
            processed += 1
        except Exception as e:  # noqa: BLE001
            failures += 1
            if graph_only:
                print(f"  Error: {e}", file=sys.stderr)
                time.sleep(args.sleep)
                continue
            error_record = PostRecord(
                title="",
                url=url,
                publication_date="",
                likes=None,
                comments=None,
                restacks=None,
                article_word_count=None,
                main_word_count=None,
                footnote_word_count=None,
                image_count=None,
                theme="",
                subthemes="",
                summary_short_neutral="",
                summary_short_mimic="",
                summary_very_short_snappy="",
                summary_dl="",
                best_quote="",
                best_quote_verified="",
                summary_long="",
                status="fetch_error",
                error=str(e),
            )
            if update_mode:
                generated_row_map[url] = record_to_row_dict(error_record)
            else:
                append_record(out_path, error_record)
            print(f"  Error: {e}", file=sys.stderr)

        time.sleep(args.sleep)

    if graph_requested:
        for normalized_url in known_post_urls:
            graph_titles.setdefault(normalized_url, _fallback_post_label(normalized_url))
            graph_dates.setdefault(normalized_url, "")
        if args.link_graph_dot:
            dot_path = Path(args.link_graph_dot)
            write_link_graph_dot(dot_path, graph_titles, graph_edge_details)
            print(f"Wrote link graph DOT: {dot_path}")
        if args.link_graph_json:
            json_path = Path(args.link_graph_json)
            write_link_graph_json(json_path, graph_titles, graph_dates, graph_edge_details)
            print(f"Wrote link graph JSON: {json_path}")
        if args.link_graph_html:
            html_path = Path(args.link_graph_html)
            write_link_graph_html(
                html_path,
                graph_titles,
                graph_dates,
                graph_edge_details,
                standalone=args.link_graph_html_standalone,
            )
            print(f"Wrote link graph HTML: {html_path}")
        print(f"Link graph stats: nodes={len(graph_titles)}, edges={len(graph_edge_details)}")

    if graph_only:
        print(
            f"Done. processed={processed}, skipped={skipped}, failures={failures}, "
            "output=link_graph_only"
        )
        return 0 if failures == 0 else 2

    if update_mode:
        assert update_source_path is not None
        with update_source_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            original_rows = list(reader)
            fieldnames = reader.fieldnames or CSV_COLUMNS
        if "url" not in fieldnames:
            raise SystemExit(f"CSV must include a 'url' column: {update_source_path}")
        for col in update_columns:
            if col not in fieldnames:
                fieldnames.append(col)

        selected_urls = set(post_urls)
        updated_count = 0
        for row_idx, row in enumerate(original_rows, start=1):
            if update_row_range is not None:
                start, end = update_row_range
                if row_idx < start or row_idx > end:
                    continue
            row_url = (row.get("url") or "").strip()
            if row_url not in selected_urls:
                continue
            generated = generated_row_map.get(row_url)
            if not generated or generated.get("status") != "ok":
                continue
            for col in update_columns:
                if col in generated:
                    row[col] = generated[col]
            updated_count += 1

        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(original_rows)
        print(f"Applied updates to {updated_count} row(s) in {out_path}")

    print(
        f"Done. processed={processed}, skipped={skipped}, failures={failures}, "
        f"output={out_path}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(run())
