"""
Loading and cleaning user datasets (CSV / TSV / semicolon files / Excel).

Every dataset is reduced to a two-column frame, ``text`` and (optionally)
``label``, with blanks dropped and duplicate texts removed. De-duplication is
not cosmetic: an identical requirement on both sides of a train/test split leaks
the answer and inflates every score.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_TEXT_HINTS = ("text", "requirement", "sentence", "content", "description", "title", "body")
_LABEL_HINTS = ("label", "class", "category", "target", "y", "tag")


@dataclass
class Dataset:
    """A named list of texts with optional gold labels."""

    name: str
    texts: List[str]
    labels: Optional[List[str]] = None

    def __post_init__(self):
        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError(
                f"{self.name}: {len(self.texts)} texts but {len(self.labels)} labels."
            )

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    def __len__(self) -> int:
        return len(self.texts)


#: Datasets shipped in data/, usable from the UI and the scripts.
SAMPLE_DATASETS = {
    "PROMISE (sec vs nonsec)": {
        "path": DATA_DIR / "PROMISE_exp_clean.csv",
        "about": "969 software requirements from 47 projects; 13% security-related.",
    },
    "SecReq (CPN + GPS + ePurse)": {
        "path": DATA_DIR / "secreq_clean.csv",
        "about": "444 requirements from three security-focused specifications.",
    },
    "NFR-SO (400-row sample)": {
        "path": DATA_DIR / "NFR_SO_clean.csv",
        "sample": 400,
        "about": "Stack Overflow question titles tagged by NFR type; stratified sample.",
    },
}


def read_table(source: Union[str, Path, io.IOBase], name: Optional[str] = None) -> pd.DataFrame:
    """
    Read a CSV/TSV/semicolon-delimited text file or an Excel workbook.

    ``source`` may be a path or a file-like object such as a Streamlit
    ``UploadedFile`` (whose ``.name`` supplies the extension).
    """
    name = name or getattr(source, "name", None) or str(source)
    suffix = Path(name).suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".json" or suffix == ".jsonl":
        return pd.read_json(source, lines=suffix == ".jsonl")

    raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    # sep=None sniffs the delimiter, which covers SecReq's `<text>;<label>` files.
    try:
        df = pd.read_csv(io.StringIO(raw), sep=None, engine="python")
    except pd.errors.ParserError:
        df = pd.read_csv(io.StringIO(raw))
    if df.shape[1] == 1 and ";" in raw.splitlines()[0]:
        df = _read_headerless_semicolon(raw)
    return df


def _read_headerless_semicolon(raw: str) -> pd.DataFrame:
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if ";" in line:
            text, label = line.rsplit(";", 1)
            rows.append({"text": text.strip().strip('"'), "label": label.strip()})
    return pd.DataFrame(rows)


def guess_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """Best guess at (text column, label column) by name, then by content."""
    lowered = {c: str(c).lower() for c in df.columns}

    def by_hint(hints, exclude=None):
        for hint in hints:
            for col, low in lowered.items():
                if col != exclude and (low == hint or low.startswith(hint) or hint in low):
                    return col
        return None

    text_col = by_hint(_TEXT_HINTS)
    if text_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == object]
        text_col = max(object_cols or list(df.columns),
                       key=lambda c: df[c].astype(str).str.len().mean())

    label_col = by_hint(_LABEL_HINTS, exclude=text_col)
    if label_col is None:
        candidates = [c for c in df.columns if c != text_col and 1 < df[c].nunique() <= 50]
        label_col = min(candidates, key=lambda c: df[c].nunique()) if candidates else None
    return text_col, label_col


def clean_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: Optional[str] = None,
    min_chars: int = 3,
) -> pd.DataFrame:
    """Two-column frame (``text``, ``label``?) with blanks and duplicates removed."""
    cols = [text_col] + ([label_col] if label_col else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found. Present: {list(df.columns)}")

    out = df[cols].copy()
    out.columns = ["text"] + (["label"] if label_col else [])
    out = out.dropna(subset=["text"])
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() >= min_chars]
    if label_col:
        out = out.dropna(subset=["label"])
        out["label"] = out["label"].astype(str).str.strip()
        out = out[out["label"].str.len() > 0]
    return out.drop_duplicates(subset=["text"]).reset_index(drop=True)


def to_dataset(df: pd.DataFrame, name: str) -> Dataset:
    """Wrap a cleaned frame from :func:`clean_dataset`."""
    labels = df["label"].tolist() if "label" in df.columns else None
    return Dataset(name=name, texts=df["text"].tolist(), labels=labels)


def load_sample(name: str, random_state: int = 42) -> pd.DataFrame:
    """A shipped sample dataset, already cleaned (``text``, ``label``)."""
    spec = SAMPLE_DATASETS[name]
    df = clean_dataset(read_table(spec["path"]), "text", "label")
    n = spec.get("sample")
    if n and len(df) > n:
        from sklearn.model_selection import train_test_split

        df, _ = train_test_split(df, train_size=n, stratify=df["label"],
                                 random_state=random_state)
        df = df.reset_index(drop=True)
    return df


def class_counts(labels: Sequence[str]) -> dict:
    return pd.Series(list(labels)).value_counts().to_dict()
