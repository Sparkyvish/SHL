"""
scraper.py  –  Scrapes the SHL Individual Test Solutions catalog.

Run once before building the FAISS index:
    python scraper.py

Writes: catalog.json
"""

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"

# SHL catalog uses ?start=N&type=1 for individual tests
# type=1  → Individual Test Solutions
# type=2  → Pre-packaged Job Solutions  (OUT OF SCOPE)
PAGE_SIZE = 12  # SHL shows 12 items per page

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# SHL test-type letter codes → human-readable label
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Multimedia",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


@dataclass
class Assessment:
    name: str
    url: str
    description: str
    test_types: List[str]          # e.g. ["A", "K"]
    test_type_labels: List[str]    # e.g. ["Ability & Aptitude", "Knowledge & Skills"]
    remote_testing: bool
    adaptive_irt: bool
    duration_minutes: Optional[int]
    languages: List[str]
    job_levels: List[str]


def _get(url: str, params: dict = None, retries: int = 3) -> Optional[requests.Response]:
    """GET with retry and polite delay."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            time.sleep(1.0)  # polite crawl delay
            return resp
        except requests.RequestException as exc:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, exc)
            time.sleep(2 ** attempt)
    log.error("All retries exhausted for %s", url)
    return None


def _parse_catalog_page(html: str) -> List[dict]:
    """
    Parse one catalog listing page.
    Returns a list of partial assessment dicts (name, url, remote, adaptive, types).
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    # SHL catalog table has rows with class "catalogue__table-row" (or similar)
    # Each row: name link | remote icon | adaptive icon | type letters
    table_rows = soup.select("table.custom-table tr[data-href]")
    if not table_rows:
        # Alternative selector used by some SHL pages
        table_rows = soup.select("tr.catalogue__table-row")

    for row in table_rows:
        try:
            # Name + URL
            name_cell = row.select_one("td.custom-table__title a, a.catalogue__link")
            if not name_cell:
                continue
            name = name_cell.get_text(strip=True)
            href = name_cell.get("href", "")
            url = href if href.startswith("http") else BASE_URL + href

            # Remote testing flag (green circle = yes)
            remote_cells = row.select("td span.catalogue__circle--yes")
            cells = row.select("td")
            # Typically col 2 = remote testing, col 3 = adaptive/IRT
            remote_testing = False
            adaptive_irt = False
            if len(cells) >= 3:
                remote_testing = bool(cells[1].select("span.catalogue__circle--yes, .yes-icon"))
                adaptive_irt = bool(cells[2].select("span.catalogue__circle--yes, .yes-icon"))

            # Test type letters: each letter in its own span/badge
            type_spans = row.select("span.catalogue__tag, span.product__tag, td:last-child span")
            test_types = []
            for span in type_spans:
                letter = span.get_text(strip=True).upper()
                if letter in TEST_TYPE_LABELS:
                    test_types.append(letter)

            # row data-href is sometimes the detail URL
            data_href = row.get("data-href", "")
            if data_href:
                url = data_href if data_href.startswith("http") else BASE_URL + data_href

            results.append({
                "name": name,
                "url": url,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "test_types": list(dict.fromkeys(test_types)),  # dedupe, preserve order
            })
        except Exception as exc:
            log.debug("Row parse error: %s", exc)

    return results


def _parse_detail_page(html: str, base: dict) -> Assessment:
    """
    Enrich a partial assessment dict with data from its detail page.
    Falls back gracefully when elements are missing.
    """
    soup = BeautifulSoup(html, "lxml")

    # Description: the first substantial paragraph on the page
    desc = ""
    for selector in [
        ".product__description p",
        ".product-detail__description",
        "article p",
        "main p",
    ]:
        node = soup.select_one(selector)
        if node:
            desc = node.get_text(separator=" ", strip=True)
            if len(desc) > 40:
                break

    # Duration
    duration = None
    for node in soup.select("*"):
        text = node.get_text(" ", strip=True)
        if "minutes" in text.lower() and any(c.isdigit() for c in text):
            import re
            match = re.search(r"(\d+)\s*(?:–|-|to)?\s*(\d+)?\s*minutes?", text, re.I)
            if match:
                duration = int(match.group(2) or match.group(1))
                break

    # Languages
    langs = []
    for node in soup.select(".product__languages li, .languages-list li"):
        lang = node.get_text(strip=True)
        if lang:
            langs.append(lang)

    # Job levels
    job_levels = []
    for node in soup.select(".product__job-levels li, .job-level li"):
        lvl = node.get_text(strip=True)
        if lvl:
            job_levels.append(lvl)

    # Re-parse test types from the detail page in case listing missed them
    test_types = list(base.get("test_types", []))
    for span in soup.select("span.product__tag, span.catalogue__tag"):
        letter = span.get_text(strip=True).upper()
        if letter in TEST_TYPE_LABELS and letter not in test_types:
            test_types.append(letter)

    return Assessment(
        name=base["name"],
        url=base["url"],
        description=desc or f"SHL assessment: {base['name']}",
        test_types=test_types,
        test_type_labels=[TEST_TYPE_LABELS.get(t, t) for t in test_types],
        remote_testing=base.get("remote_testing", False),
        adaptive_irt=base.get("adaptive_irt", False),
        duration_minutes=duration,
        languages=langs,
        job_levels=job_levels,
    )


def _count_total(html: str) -> int:
    """Extract total number of individual test assessments from catalog page."""
    soup = BeautifulSoup(html, "lxml")
    # SHL typically shows "Showing X–Y of Z results"
    for selector in [".catalogue__count", ".results-count", "p.result-count"]:
        node = soup.select_one(selector)
        if node:
            import re
            m = re.search(r"of\s+(\d+)", node.get_text())
            if m:
                return int(m.group(1))
    # Fallback: count rows on first page and assume more exist
    rows = soup.select("table.custom-table tr[data-href], tr.catalogue__table-row")
    return max(len(rows), PAGE_SIZE)


def scrape_catalog(out_path: str = "catalog.json") -> List[dict]:
    """
    Full scrape: paginate through all Individual Test Solutions, fetch each
    detail page, and write catalog.json.
    """
    log.info("Fetching first catalog page …")
    first_resp = _get(CATALOG_URL, params={"type": "1", "start": "0"})
    if first_resp is None:
        raise RuntimeError("Cannot reach SHL catalog. Check network connectivity.")

    total = _count_total(first_resp.text)
    log.info("Total individual test assessments to scrape: ~%d", total)

    # ── 1. Collect all listing rows ──────────────────────────────────────────
    partials: List[dict] = []
    partials.extend(_parse_catalog_page(first_resp.text))

    start = PAGE_SIZE
    while start < total:
        resp = _get(CATALOG_URL, params={"type": "1", "start": str(start)})
        if resp is None:
            log.warning("Skipping page at start=%d", start)
            start += PAGE_SIZE
            continue
        batch = _parse_catalog_page(resp.text)
        if not batch:
            log.info("Empty page at start=%d – stopping pagination.", start)
            break
        partials.extend(batch)
        log.info("  … collected %d listings so far", len(partials))
        start += PAGE_SIZE

    # Deduplicate by URL
    seen_urls: set = set()
    unique_partials = []
    for p in partials:
        if p["url"] not in seen_urls:
            seen_urls.add(p["url"])
            unique_partials.append(p)
    log.info("Unique listings: %d", len(unique_partials))

    # ── 2. Fetch each detail page ────────────────────────────────────────────
    assessments: List[dict] = []
    for i, partial in enumerate(unique_partials, 1):
        log.info("[%d/%d] Scraping detail: %s", i, len(unique_partials), partial["name"])
        resp = _get(partial["url"])
        if resp is None:
            # Use partial data with a placeholder description
            a = Assessment(
                name=partial["name"],
                url=partial["url"],
                description=f"SHL assessment: {partial['name']}",
                test_types=partial.get("test_types", []),
                test_type_labels=[TEST_TYPE_LABELS.get(t, t) for t in partial.get("test_types", [])],
                remote_testing=partial.get("remote_testing", False),
                adaptive_irt=partial.get("adaptive_irt", False),
                duration_minutes=None,
                languages=[],
                job_levels=[],
            )
        else:
            a = _parse_detail_page(resp.text, partial)
        assessments.append(asdict(a))

    # ── 3. Persist ────────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    log.info("Saved %d assessments → %s", len(assessments), out_path)
    return assessments


if __name__ == "__main__":
    scrape_catalog()
