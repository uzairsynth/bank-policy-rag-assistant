"""Public State Bank of Pakistan (SBP) documents the assistant answers from.

The PDFs are not committed to the repository; scripts/download_docs.py fetches them
from SBP's official archive site (archive.sbp.org.pk).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"

DOCUMENTS = [
    {
        "id": "consumer-financing",
        "title": "Prudential Regulations for Consumer Financing (2016)",
        "url": "https://archive.sbp.org.pk/bprd/2016/C10-Annx-A.pdf",
    },
    {
        "id": "branchless-banking",
        "title": "Branchless Banking Regulations (2019)",
        "url": "https://archive.sbp.org.pk/bprd/2019/C10-Branchless-Banking-Regulations.pdf",
    },
    {
        "id": "payment-card-security",
        "title": "Regulations for Payment Card Security (2016)",
        "url": "https://archive.sbp.org.pk/psd/2016/C5_Annex.pdf",
    },
    {
        "id": "aml-cft-cpf",
        "title": "AML/CFT/CPF Regulations (2020)",
        "url": "https://archive.sbp.org.pk/bprd/2020/C5-Annex.pdf",
    },
]


def pdf_path(doc: dict) -> Path:
    return RAW_DIR / f"{doc['id']}.pdf"
