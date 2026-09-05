"""Spreadsheet parser (pandas) for xlsx and csv.

Per the slice-2 decision, a sheet is represented for retrieval by **one
``embeddable=True`` summary block**, not by its rows. Windowed rows would be
near-duplicates in embedding space and would crowd out everything else in
top-k. The actual rows are still stored -- as ``embeddable=False`` ``table``
blocks -- so a citation can point at them and a future reading endpoint can page
through them; they are simply never embedded.

Two thin classes (:class:`XlsxParser`, :class:`CsvParser`) share
:class:`_SpreadsheetParser`; they exist separately only so ``file_type`` is
``"xlsx"`` vs ``"csv"`` for the retrieval filter.

Judgment calls recorded in ``extraction_flags``:

* ``sheet_empty``            -- a sheet with no columns, or no data rows
* ``sheet_all_string_columns`` -- every column came back as text (a misread
  header row or a genuinely all-text sheet -- worth surfacing either way)
* ``spreadsheet_no_data``    -- nothing indexable at all -> ``EMPTY_NO_TEXT``
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.ir import Locator
from app.parsers._common import BlockBuilder, PlainText, rows_to_markdown
from app.parsers.base import EMPTY_NO_TEXT, ParseInput, ParseResult, register_parser

_SAMPLE_CHARS = 40
_MAX_SAMPLES = 3
_COLNAME_CHARS = 60

_DTYPE_NAMES = {
    "object": "string",
    "str": "string",
    "string": "string",
    "int8": "integer", "int16": "integer", "int32": "integer", "int64": "integer",
    "Int8": "integer", "Int16": "integer", "Int32": "integer", "Int64": "integer",
    "uint8": "integer", "uint16": "integer", "uint32": "integer", "uint64": "integer",
    "float16": "float", "float32": "float", "float64": "float",
    "Float32": "float", "Float64": "float",
    "bool": "boolean", "boolean": "boolean",
}


class _SpreadsheetParser:
    name: str
    version: str
    file_type: str
    mimetypes: tuple[str, ...]

    def __init__(self, rows_per_block: int | None = None) -> None:
        self._rows_per_block = rows_per_block

    # -- subclass hook --------------------------------------------------
    def _load(self, src: ParseInput, data: bytes) -> list[tuple[str, pd.DataFrame]]:
        raise NotImplementedError

    def parse(self, src: ParseInput) -> ParseResult:
        data = src.open_stream().read()
        rows_per_block = self._rows_per_block or get_settings().sheet_rows_per_block
        builder = BlockBuilder(src=src)
        text = PlainText()

        for sheet, df in self._load(src, data):
            _emit_sheet(sheet, df, text, builder, rows_per_block)

        if not builder.blocks:
            builder.note("spreadsheet_no_data")
            return builder.result(status_hint=EMPTY_NO_TEXT)
        return builder.result()


class XlsxParser(_SpreadsheetParser):
    name = "xlsx-pandas"
    version = "1"
    file_type = "xlsx"
    mimetypes = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",)

    def _load(self, src: ParseInput, data: bytes) -> list[tuple[str, pd.DataFrame]]:
        book = pd.read_excel(io.BytesIO(data), sheet_name=None)  # {name: DataFrame}, header=0
        return list(book.items())


class CsvParser(_SpreadsheetParser):
    name = "csv-pandas"
    version = "1"
    file_type = "csv"
    mimetypes = ("text/csv",)

    def _load(self, src: ParseInput, data: bytes) -> list[tuple[str, pd.DataFrame]]:
        stem = Path(src.filename).stem or "data"
        return [(stem, _read_csv(data))]


register_parser(XlsxParser())
register_parser(CsvParser())


def _read_csv(data: bytes) -> pd.DataFrame:
    """Delimiter-sniffing read, with graceful fallbacks: an empty/blank file and
    a file the sniffer chokes on (e.g. a single column with no delimiter) both
    degrade rather than raise."""
    if not data.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except (csv.Error, pd.errors.ParserError):
        try:
            return pd.read_csv(io.BytesIO(data))
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


# --------------------------------------------------------------------------
# per-sheet emission
# --------------------------------------------------------------------------


def _emit_sheet(
    sheet: str,
    df: pd.DataFrame,
    text: PlainText,
    builder: BlockBuilder,
    rows_per_block: int,
) -> None:
    if df.shape[1] == 0:
        builder.note("sheet_empty")
        return

    summary = _summary_markdown(sheet, df)
    start, end = text.add(summary)
    builder.add(
        "table", summary,
        locator=Locator(sheet=sheet, char_start=start, char_end=end),
        section_path=[sheet],
        embeddable=True,
    )

    if len(df) == 0:
        builder.note("sheet_empty")
        return
    if all(_friendly_dtype(df[c].dtype) == "string" for c in df.columns):
        builder.note("sheet_all_string_columns")

    header = [_fmt(str(c), _COLNAME_CHARS) for c in df.columns]
    for w0 in range(0, len(df), rows_per_block):
        window = df.iloc[w0 : w0 + rows_per_block]
        body = [[_fmt(v) for v in row] for row in window.itertuples(index=False, name=None)]
        label = f"rows {w0 + 1}-{min(w0 + rows_per_block, len(df))}"
        md = rows_to_markdown([header, *body])
        start, end = text.add(md)
        builder.add(
            "table", md,
            locator=Locator(sheet=sheet, char_start=start, char_end=end),
            section_path=[sheet, label],
            embeddable=False,
        )


def _summary_markdown(sheet: str, df: pd.DataFrame) -> str:
    head = f'Sheet "{sheet}" — {len(df):,} rows × {df.shape[1]:,} columns'
    return head + "\n\n" + rows_to_markdown(_profile_rows(df))


def _profile_rows(df: pd.DataFrame) -> list[list[str]]:
    out: list[list[str]] = [
        ["column", "dtype", "non-null", "distinct", "min", "max", "samples"]
    ]
    for col in df.columns:
        s = df[col]
        friendly = _friendly_dtype(s.dtype)
        non_null = int(s.notna().sum())
        distinct = int(s.nunique(dropna=True))

        cmin = cmax = ""
        if non_null and friendly in ("integer", "float", "datetime"):
            try:
                cmin, cmax = _fmt(s.min()), _fmt(s.max())
            except (TypeError, ValueError):
                cmin = cmax = ""

        samples: list[str] = []
        for v in s.dropna().tolist():
            fv = _fmt(v)
            if fv and fv not in samples:
                samples.append(fv)
            if len(samples) == _MAX_SAMPLES:
                break

        out.append([
            _fmt(str(col), _COLNAME_CHARS),
            friendly,
            f"{non_null:,}",
            f"{distinct:,}",
            cmin,
            cmax,
            "; ".join(samples),
        ])
    return out


def _friendly_dtype(dtype: object) -> str:
    s = str(dtype)
    if s in _DTYPE_NAMES:
        return _DTYPE_NAMES[s]
    if s.startswith("datetime64") or s.startswith("datetime"):
        return "datetime"
    if s.startswith("string"):
        return "string"
    return s


def _fmt(v: object, limit: int = _SAMPLE_CHARS) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, float):
        s = f"{v:.6g}"
    else:
        s = str(v)
    s = s.replace("\n", " ").replace("\r", " ").replace("|", r"\|").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"
