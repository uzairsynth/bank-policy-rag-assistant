"""Download the SBP regulation PDFs listed in assistant/sources.py into data/raw/."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant.sources import DOCUMENTS, RAW_DIR, pdf_path  # noqa: E402


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for doc in DOCUMENTS:
        path = pdf_path(doc)
        if path.exists():
            print(f"exists   {path.name}")
            continue
        request = urllib.request.Request(doc["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if not data.startswith(b"%PDF"):
            raise RuntimeError(f"{doc['url']} did not return a PDF")
        path.write_bytes(data)
        print(f"saved    {path.name} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
