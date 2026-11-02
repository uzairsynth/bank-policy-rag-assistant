"""Write each PDF's cleaned text, page by page, to data/pages/ for labelling evaluation questions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant.ingest import read_pages  # noqa: E402
from assistant.sources import DOCUMENTS, ROOT, pdf_path  # noqa: E402


def main() -> None:
    out_dir = ROOT / "data" / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in DOCUMENTS:
        pages = read_pages(pdf_path(doc))
        text = "\n\n".join(f"=== PAGE {n} ===\n{page}" for n, page in enumerate(pages, start=1))
        (out_dir / f"{doc['id']}.txt").write_text(text, encoding="utf-8")
        print(f"{doc['id']}: {len(pages)} pages")


if __name__ == "__main__":
    main()
