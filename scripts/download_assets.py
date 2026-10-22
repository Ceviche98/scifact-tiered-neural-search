"""
Asset verification and automatic GitHub Release downloader for SciFact Tiered Neural Search.

Verifies the presence of the BEIR SciFact corpus, pre-computed 256d FAISS indices,
and the 1536d Matryoshka Stage-2 reranking cache (`data_pro_tier/mrl_cache.pkl`, 30.4 MB).
If `mrl_cache.pkl` is missing locally, downloads it directly from the repository's
GitHub Release (`v1.0.0`) with a progress bar.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    CORPUS_FILE,
    FAST_IDMAP,
    FAST_INDEX,
    GITHUB_RELEASE_CACHE_URL,
    HYDE_CACHE,
    PRO_CACHE,
    PRO_IDMAP,
    PRO_INDEX,
    QRELS_FILE,
    QUERIES_FILE,
)


def ensure_pro_cache(url: str = GITHUB_RELEASE_CACHE_URL, quiet: bool = False) -> Path:
    """Ensure `data_pro_tier/mrl_cache.pkl` exists locally, downloading from GitHub Release if missing."""
    if PRO_CACHE.exists() and PRO_CACHE.stat().st_size > 1_000_000:
        if not quiet:
            size_mb = PRO_CACHE.stat().st_size / (1024 * 1024)
            print(f"[OK] Pro Stage-2 MRL cache present: {PRO_CACHE} ({size_mb:.2f} MB)")
        return PRO_CACHE

    PRO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PRO_CACHE.with_suffix(".pkl.part")

    if not quiet:
        print(f"[INFO] Downloading Pro Tier 1536d MRL cache from:\n       {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scifact-tiered-neural-search/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response, open(tmp_path, "wb") as out_file:
            total_size = int(response.info().get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if not quiet and total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r       Progress: {mb_done:.1f} / {mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)
        if not quiet:
            print()
        tmp_path.replace(PRO_CACHE)
        if not quiet:
            print(f"[OK] Saved {PRO_CACHE} ({PRO_CACHE.stat().st_size / (1024 * 1024):.2f} MB)")
        return PRO_CACHE
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {PRO_CACHE.name} from {url} ({exc}).\n"
            "You can also rebuild it locally by running: python pipelines_offline/build_mrl_index.py --tier pro"
        ) from exc


def verify_all_assets() -> bool:
    """Verify all required SciFact datasets, FAISS indices, and Pro Tier caches."""
    required = [
        ("BEIR SciFact Corpus", CORPUS_FILE),
        ("BEIR SciFact Queries", QUERIES_FILE),
        ("BEIR SciFact Qrels", QRELS_FILE),
        ("Fast Tier 256d FAISS Index", FAST_INDEX),
        ("Fast Tier ID Map", FAST_IDMAP),
        ("Pro Tier 256d FAISS Index", PRO_INDEX),
        ("Pro Tier ID Map", PRO_IDMAP),
        ("Pre-computed Qwen3-8B + GOLFer HyDE Cache", HYDE_CACHE),
    ]

    print("=" * 68)
    print("SciFact Tiered Neural Search - Asset Verification")
    print("=" * 68)

    all_ok = True
    for label, path in required:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"[OK] {label:<42} ({size_kb:>8.1f} KB)")
        else:
            print(f"[MISSING] {label:<37} -> {path}")
            all_ok = False

    try:
        ensure_pro_cache(quiet=False)
    except RuntimeError as err:
        print(f"[WARN] {err}")
        all_ok = False

    print("=" * 68)
    if all_ok:
        print("All required indices, caches, and BEIR SciFact files are ready!")
        print("Launch the application with: streamlit run app_ui.py")
    return all_ok


if __name__ == "__main__":
    ok = verify_all_assets()
    sys.exit(0 if ok else 1)
