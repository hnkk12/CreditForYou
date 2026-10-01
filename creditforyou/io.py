"""Đọc dataset, tự nhận diện cột theo config/schema.json và chuẩn hoá kiểu dữ liệu.

Mục tiêu: khi có dataset thật, chỉ cần chạy `python -m creditforyou inspect <file>` để
xem cột nào đã khớp / còn thiếu, rồi bổ sung alias trong schema.json nếu cần.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import schema_config


# --------------------------------------------------------------------------- đọc file

def read_table(path, nrows=None, usecols=None) -> pd.DataFrame:
    """Đọc CSV / TSV / Excel / Parquet / JSON. CSV tự dò dấu phân cách và encoding."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, nrows=nrows, usecols=usecols)
    if suffix == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
        return df.head(nrows) if nrows else df
    if suffix == ".json":
        df = pd.read_json(path)
        return df.head(nrows) if nrows else df

    # CSV / TXT / TSV
    encoding = _detect_encoding(path)
    with open(path, "r", encoding=encoding, errors="replace") as f:
        first_lines = [f.readline() for _ in range(3)]
    sep = _detect_sep(first_lines)
    # File Lending Club gốc (LoanStats*.csv) có 1 dòng ghi chú trước header
    skiprows = 1 if first_lines[0].count(sep) == 0 and first_lines[1].count(sep) > 0 else 0
    usecols_fn = None
    if usecols is not None:
        wanted = set(usecols)
        usecols_fn = lambda c: c in wanted  # noqa: E731
    return pd.read_csv(path, sep=sep, encoding=encoding, skiprows=skiprows, nrows=nrows,
                       usecols=usecols_fn, low_memory=False)


def read_header(path) -> list:
    return list(read_table(path, nrows=5).columns)


def _detect_encoding(path: Path) -> str:
    raw = path.open("rb").read(200_000)
    for enc in ("utf-8-sig", "utf-8"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "cp1258"  # Windows tiếng Việt


def _detect_sep(lines) -> str:
    lines = [l for l in lines if l.strip()]
    if not lines:
        return ","
    counts = {s: max(l.count(s) for l in lines) for s in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


# --------------------------------------------------------------------------- map cột

def normalize_name(name) -> str:
    """'Ngày Giao Dịch' -> 'ngay_giao_dich'"""
    s = str(name).strip().lower().replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


@dataclass
class ColumnMapping:
    dataset: str
    mapping: dict                      # tên chuẩn -> tên cột gốc
    missing_required: list = field(default_factory=list)
    unmapped_columns: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required


def map_columns(columns, dataset: str) -> ColumnMapping:
    spec = schema_config()[dataset]
    overrides = {k: v for k, v in spec.get("overrides", {}).items() if v}
    norm_to_orig = {}
    for c in columns:
        norm_to_orig.setdefault(normalize_name(c), c)

    mapping, used = {}, set()
    for canon, orig in overrides.items():
        if orig not in columns:
            raise KeyError(f"schema.json overrides[{dataset}][{canon}] = '{orig}' nhưng file không có cột này.")
        mapping[canon] = orig
        used.add(orig)

    for canon, aliases in spec["columns"].items():
        if canon in mapping:
            continue
        for alias in aliases:
            orig = norm_to_orig.get(normalize_name(alias))
            if orig is not None and orig not in used:
                mapping[canon] = orig
                used.add(orig)
                break

    missing = [c for c in spec.get("required", []) if c not in mapping]
    # required_one_of: danh sách các lựa chọn, chỉ cần thoả 1 lựa chọn
    one_of = spec.get("required_one_of", [])
    if one_of and not any(all(c in mapping for c in option) for option in one_of):
        missing.append(" HOẶC ".join("+".join(option) for option in one_of))

    unmapped = [c for c in columns if c not in used]
    return ColumnMapping(dataset, mapping, missing, unmapped)


def apply_mapping(df: pd.DataFrame, cm: ColumnMapping) -> pd.DataFrame:
    """Trả về DataFrame chỉ gồm các cột đã map, đổi sang tên chuẩn."""
    if not cm.ok:
        raise ValueError(
            f"Dataset '{cm.dataset}' thiếu cột bắt buộc: {cm.missing_required}. "
            f"Chạy `python -m creditforyou inspect <file> --as {cm.dataset}` rồi thêm alias/overrides "
            f"trong config/schema.json."
        )
    return df[list(cm.mapping.values())].rename(columns={v: k for k, v in cm.mapping.items()}).copy()


def load_dataset(path, dataset: str, nrows=None, keep_extra=False) -> pd.DataFrame:
    """Đọc file + map cột. Mặc định chỉ đọc những cột cần (quan trọng với Lending Club ~150 cột).
    keep_extra=True: giữ thêm các cột không map (giữ tên gốc) để xuất kèm kết quả."""
    cm = map_columns(read_header(path), dataset)
    if not cm.ok:
        apply_mapping(pd.DataFrame(), cm)  # raise lỗi có hướng dẫn
    df = read_table(path, nrows=nrows, usecols=None if keep_extra else list(cm.mapping.values()))
    out = apply_mapping(df, cm)
    if keep_extra:
        extra = [c for c in cm.unmapped_columns if c not in out.columns]
        out = pd.concat([out, df[extra]], axis=1)
    return out


# --------------------------------------------------------------------------- parse giá trị

def normalize_id(s: pd.Series) -> pd.Series:
    """Mã khách hàng -> chuỗi; '1001.0' (do Excel/NaN biến thành float) -> '1001'."""
    return s.astype(str).str.strip().str.replace(r"^(\d+)\.0+$", r"\1", regex=True)


def parse_amount(s: pd.Series) -> pd.Series:
    """Chuyển '1.234.567', '1,234,567', '1 234 567đ', '(500,000)', '8.2M' -> float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def one(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        t = str(x).strip().lower()
        if not t or t in ("nan", "none", "-"):
            return np.nan
        neg = t.startswith("-") or (t.startswith("(") and t.endswith(")"))
        mult = 1.0
        if re.search(r"(tr|trieu|m)\s*$", t):
            mult = 1e6
        elif re.search(r"(k|nghin|ngan)\s*$", t):
            mult = 1e3
        t = re.sub(r"[^0-9.,]", "", t)
        if not t:
            return np.nan
        if "," in t and "." in t:
            # dấu xuất hiện sau cùng là dấu thập phân
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "").replace(",", ".")
            else:
                t = t.replace(",", "")
        elif "," in t or "." in t:
            sep = "," if "," in t else "."
            parts = t.split(sep)
            if len(parts) > 2 or (len(parts[-1]) == 3 and mult == 1.0):
                t = t.replace(sep, "")            # dấu phân cách hàng nghìn
            else:
                t = t.replace(",", ".")           # dấu thập phân
        try:
            v = round(float(t) * mult, 2)
        except ValueError:
            return np.nan
        return -v if neg else v

    return s.map(one).astype(float)


def parse_dates(s: pd.Series, dayfirst=None) -> pd.Series:
    """dayfirst=None: tự đoán. Nếu có giá trị kiểu 25/03/2024 -> dd/mm; 03/25/2024 -> mm/dd."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    st = s.astype(str).str.strip()
    if dayfirst is None:
        parts = st.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")
        if parts[0].notna().any():
            a = pd.to_numeric(parts[0], errors="coerce")
            b = pd.to_numeric(parts[1], errors="coerce")
            if (b > 12).any() and not (a > 12).any():
                dayfirst = False
            else:
                dayfirst = True   # mặc định kiểu Việt Nam dd/mm/yyyy
        else:
            dayfirst = False      # ISO yyyy-mm-dd
    return pd.to_datetime(st, dayfirst=dayfirst, errors="coerce", format="mixed")


_INFLOW_WORDS = {"credit", "cr", "c", "in", "inflow", "deposit", "income", "receive", "received",
                 "co", "ghi_co", "tien_vao", "thu", "nhan", "+"}
_OUTFLOW_WORDS = {"debit", "dr", "d", "out", "outflow", "withdrawal", "payment", "expense", "spend",
                  "no", "ghi_no", "tien_ra", "chi", "-"}


def parse_direction(s: pd.Series) -> pd.Series:
    """Trả về +1 (tiền vào), -1 (tiền ra), NaN (không rõ)."""
    norm = s.astype(str).map(normalize_name)
    out = pd.Series(np.nan, index=s.index)
    out[norm.isin(_INFLOW_WORDS)] = 1.0
    out[norm.isin(_OUTFLOW_WORDS)] = -1.0
    raw = s.astype(str).str.strip()
    out[raw == "+"] = 1.0
    out[raw == "-"] = -1.0
    return out


def parse_term_months(s: pd.Series) -> pd.Series:
    """' 36 months' -> 36; '3 năm' -> 36; 36 -> 36."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    st = s.astype(str).map(normalize_name)
    num = pd.to_numeric(st.str.extract(r"(\d+(?:_\d+)?)")[0].str.replace("_", "."), errors="coerce")
    years = st.str.contains(r"year|nam", na=False)
    return num.where(~years, num * 12)


def parse_years(s: pd.Series) -> pd.Series:
    """'10+ years' -> 10; '< 1 year' -> 0.5; 'n/a' -> NaN; '18 months' -> 1.5."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    st = s.astype(str).str.lower().str.strip()
    num = pd.to_numeric(st.str.extract(r"(\d+(?:[.,]\d+)?)")[0].str.replace(",", "."), errors="coerce")
    num = num.where(~st.str.contains("<", na=False), 0.5)
    months = st.str.contains(r"month|thang", na=False)
    return num.where(~months, num / 12)


def parse_percent_ratio(s: pd.Series) -> pd.Series:
    """'45.2%' / 45.2 / 0.452 -> 0.452 (tỷ lệ 0-1)."""
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
    else:
        v = pd.to_numeric(s.astype(str).str.replace("%", "").str.replace(",", ".").str.strip(),
                          errors="coerce")
    if v.dropna().median() > 1.5:   # đang ở dạng phần trăm
        v = v / 100.0
    return v


def parse_label(s: pd.Series) -> pd.Series:
    """Nhãn vỡ nợ -> 1 / 0 / NaN (trạng thái chưa kết thúc như 'Current' -> NaN)."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        v = s.astype(float)
        return v.where(v.isin([0.0, 1.0]))
    labels = schema_config()["label_values"]
    bad = {x.lower() for x in labels["bad"]}
    good = {x.lower() for x in labels["good"]}
    st = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index)
    out[st.isin(bad) | st.isin({"1", "1.0"})] = 1.0
    out[st.isin(good) | st.isin({"0", "0.0"})] = 0.0
    return out
