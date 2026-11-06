"""Data science toolkit: the calculations and plots behind main.ipynb.

Every function works on any tabular dataset. After `prepare()`, the working
dataset remembers its target and feature columns (in `df.attrs`), so most
functions accept `columns=None` to mean "all feature columns" and
`color_by="target"` to color by the target column.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

OUTPUT_DIR = Path("output")
TRANSFORMS = ("minmax", "standard", "robust", "log", "log10")
TEXT_TYPES = ["object", "string", "category", "bool"]
COPY_SUFFIX = "_copy"  # changed data is saved next to the original as <name>_copy.csv

plt.style.use("ggplot")


# ---------------------------------------------------------------- helpers

def _features(df):
    """Feature columns remembered by prepare(), else all numeric columns."""
    return list(df.attrs.get("features") or df.select_dtypes("number").columns)


def _target(df):
    return df.attrs.get("target")


def _columns(df, columns, minimum=1):
    """Validate a column selection; None means all feature columns."""
    if columns is None:
        cols = _features(df)
    elif isinstance(columns, str):
        cols = [columns]
    else:
        cols = list(columns)
    unknown = [c for c in cols if c not in df.columns]
    if unknown:
        raise KeyError(f"Unknown column(s) {unknown}. Available: {list(df.columns)}")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Column(s) {non_numeric} are not numeric")
    if len(cols) < minimum:
        raise ValueError(f"Select at least {minimum} column(s), got {len(cols)}")
    return cols


def _column(df, column):
    """Validate one column name of any type."""
    if column not in df.columns:
        raise KeyError(f"Unknown column '{column}'. Available: {list(df.columns)}")
    return column


def _category(df, column=None, exclude=()):
    """A grouping column: the given one, else the text column with the fewest distinct values."""
    if column is not None:
        return _column(df, column)
    candidates = [c for c in df.select_dtypes(include=TEXT_TYPES).columns
                  if c not in exclude and df[c].nunique() > 1]
    if not candidates:
        other = f" other than {list(exclude)}" if exclude else ""
        raise ValueError(f"No text column{other} to group by; pass a column name explicitly")
    return min(candidates, key=lambda c: df[c].nunique())


def _labels(df, *columns):
    """Copy of df with True/False grouping columns turned into text, so their values work as labels."""
    bools = [c for c in columns if pd.api.types.is_bool_dtype(df[c])]
    return df.astype({c: str for c in bools}) if bools else df


def _value(df, value=None):
    """A numeric value column: the given one, else the first feature."""
    return _columns(df, value if value is not None else _features(df)[:1])[0]


def _tidy_index(table):
    """Show whole-number float labels (e.g. years read as 2006.0) as integers."""
    idx = table.index
    if pd.api.types.is_float_dtype(idx) and np.all(np.mod(idx, 1) == 0):
        table.index = idx.astype(int)
    return table


def _bounds(x, method, factor, threshold):
    """Per-column (lower, upper) outlier bounds for the IQR or z-score method."""
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        return q1 - factor * (q3 - q1), q3 + factor * (q3 - q1)
    if method == "zscore":
        return x.mean() - threshold * x.std(), x.mean() + threshold * x.std()
    raise ValueError('method must be "iqr" or "zscore"')


def _color_column(df, color_by):
    """Resolve color_by: "target" -> the target column, None -> no coloring."""
    if color_by == "target":
        return _target(df)
    if color_by is not None and color_by not in df.columns:
        raise KeyError(f"color_by column '{color_by}' not found")
    return color_by


def _groups(df, color_col):
    """Yield (label, row mask) per class of color_col, or one group for all rows."""
    if color_col is None:
        yield None, np.ones(len(df), dtype=bool)
        return
    for label in sorted(df[color_col].dropna().unique(), key=str):
        yield label, (df[color_col] == label).to_numpy()


def _scatter(ax, df, color_col, color, *coords):
    for label, mask in _groups(df, color_col):
        kwargs = {"color": color} if label is None else {"label": str(label)}
        ax.scatter(*(np.asarray(c)[mask] for c in coords), alpha=0.7, s=15, **kwargs)
    if color_col is not None:
        ax.legend(title=color_col)


def _finish(fig, save_as):
    """Lay out, optionally save to OUTPUT_DIR, and show a figure."""
    fig.tight_layout()
    if save_as:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / (save_as if Path(save_as).suffix else f"{save_as}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def _strength(r):
    a = abs(r)
    strength = ("negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5
                else "strong" if a < 0.7 else "very strong")
    return f"{strength} {'positive' if r > 0 else 'negative'}" if a >= 0.1 else strength


# ---------------------------------------------------------------- Part 1: loading

def list_datasets(data_dir="data"):
    """Print the CSV files available in data_dir."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv")) if data_dir.is_dir() else []
    if not files:
        print(f"No CSV files in {data_dir.resolve()}")
    for f in files:
        note = "  (changed copy)" if f.stem.endswith(COPY_SUFFIX) else ""
        print(f"{f.name:45s} {f.stat().st_size / 1e6:8.2f} MB{note}")


def _detect_separator(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        sample = "".join(fh.readline() for _ in range(50))
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def copy_path(source):
    """Where the changed copy of `source` is saved: same folder, name + COPY_SUFFIX."""
    p = Path(source)
    stem = p.stem if p.stem.endswith(COPY_SUFFIX) else p.stem + COPY_SUFFIX
    return p.with_name(stem + p.suffix)


def _changed(out, df, changed):
    """Mark `out` as different from the loaded file when this step (or an earlier one) changed it,
    and then save it as the copy. The original file is never written to."""
    out.attrs["modified"] = bool(df.attrs.get("modified") or changed)
    source = out.attrs.get("source")
    if out.attrs["modified"] and source:
        path = copy_path(source)
        out.to_csv(path, index=False)
        if path == Path(source):
            print(f"Updated the copy {path} (the original data file is untouched)")
        else:
            print(f"Saved the changed data as {path} (the original {Path(source).name} is untouched)")
    return out


def load_csv(data_dir="data", file=None, sep=None, header="auto", index_col=None):
    """Read a CSV from `data_dir`. Separator and header row are detected automatically.

    file: file name inside data_dir; None = the only .csv in data_dir.
    header: "auto", True or False (False -> columns named col_1, col_2, ...).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir.resolve()}. Create it and put your CSV inside.")
    if file:
        path = data_dir / file
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path.resolve()}")
    else:
        csv_files = sorted(data_dir.glob("*.csv"))
        if len(csv_files) != 1:
            raise FileNotFoundError(
                f"Expected one CSV file in {data_dir.resolve()}, found: {[f.name for f in csv_files]}. "
                "Pass file='name.csv'."
            )
        path = csv_files[0]

    read_kwargs = {"sep": sep or _detect_separator(path)}
    if header == "auto":
        # A header exists when some column is text in the first row but numeric below it.
        sample = pd.read_csv(path, header=None, nrows=50, **read_kwargs)
        first_numeric = pd.to_numeric(sample.iloc[0], errors="coerce").notna()
        rest_numeric = sample.iloc[1:].apply(lambda c: pd.to_numeric(c, errors="coerce").notna().all())
        has_header = bool((rest_numeric & ~first_numeric).any() or not rest_numeric.any())
    else:
        has_header = bool(header)

    df = pd.read_csv(path, header=0 if has_header else None, index_col=index_col, **read_kwargs)
    if not has_header:
        df.columns = [f"col_{i + 1}" for i in range(df.shape[1])]
    df.columns = df.columns.astype(str).str.strip()
    # Drop an unnamed leftover index column (written by DataFrame.to_csv), if any.
    leftover = [c for c in df.columns if c.startswith("Unnamed: 0")]
    if leftover and (df[leftover[0]].to_numpy() == np.arange(len(df))).all():
        df = df.drop(columns=leftover[0])
    df.attrs.update(source=str(path), modified=False)
    print(f"Loaded {path} -> {df.shape[0]} rows x {df.shape[1]} columns (header row: {'yes' if has_header else 'no'})")
    return df


# ---------------------------------------------------------------- Part 2: inspection

def shape(df):
    rows, cols = df.shape
    print(f"Rows: {rows}, Columns: {cols}")


def column_info(df):
    """Data type, non-null, missing and unique counts per column."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "unique": df.nunique(),
    })


def summary(df, columns=None):
    """count, mean, std, min, Q1, median, Q3, max per column (all numeric columns when None)."""
    cols = list(df.select_dtypes("number").columns) if columns is None else _columns(df, columns)
    return df[cols].describe().T


def data_issues(df):
    """List common data problems: missing values, duplicates, ID-like, constant and mistyped columns."""
    issues = []
    n_dup = int(df.duplicated().sum())
    if n_dup:
        issues.append(("(all columns)", "duplicate rows", f"{n_dup} fully identical rows"))
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        if n_missing:
            issues.append((c, "missing values", f"{n_missing} ({n_missing / len(df):.1%})"))
        n_unique = s.nunique()
        if n_unique <= 1:
            issues.append((c, "constant", "only one distinct value: carries no information"))
        elif n_unique == len(df) and not pd.api.types.is_float_dtype(s):
            issues.append((c, "ID-like", "a different value in every row: probably an identifier"))
        if pd.api.types.is_float_dtype(s) and n_missing < len(df) and np.all(np.mod(s.dropna(), 1) == 0):
            issues.append((c, "whole numbers stored as decimals",
                           "often caused by missing values; may be a count, code or year"))
        if s.dtype.kind in "OUT" or pd.api.types.is_string_dtype(s):
            as_number = pd.to_numeric(s.dropna(), errors="coerce")
            if len(as_number) and as_number.notna().mean() > 0.9:
                issues.append((c, "numbers stored as text",
                               f"{int(as_number.isna().sum())} value(s) are not numbers"))
    if not issues:
        print("No common data issues found")
    return pd.DataFrame(issues, columns=["column", "issue", "detail"])


def clean(df, duplicates=True, required=None, keep=None, integer=None):
    """Remove problem rows. Each step reports how many rows it dropped.

    duplicates: True = drop fully identical rows; a list of columns = drop rows that repeat
                those columns (e.g. ["Name", "Platform"]); False = keep everything.
    required:   columns that must be filled in; rows missing any of them are dropped.
    keep:       {column: (min, max)} keeps only rows inside the range (inclusive). None = no limit.
                Missing values are left alone here; list the column in `required` to drop them.
    integer:    columns to store as whole numbers (e.g. ["Year"], read as 2006.0 because of
                missing values). Columns that still have missing values become a nullable
                integer type; real decimals (e.g. 2006.5) raise an error instead of being rounded.
    """
    data = df.copy()
    start = len(data)
    if duplicates:
        subset = None if duplicates is True else [_column(data, c) for c in duplicates]
        before = len(data)
        data = data.drop_duplicates(subset=subset)
        print(f"Duplicates ({'all columns' if subset is None else subset}): {before - len(data)} rows dropped")
    for c in [required] if isinstance(required, str) else (required or []):
        before = len(data)
        data = data.dropna(subset=[_column(data, c)])
        print(f"Missing {c}: {before - len(data)} rows dropped")
    for c, (low, high) in (keep or {}).items():
        s = data[_columns(data, [c])[0]]
        inside = s.isna() | ((low is None) | (s >= low)) & ((high is None) | (s <= high))
        print(f"Outside {c} range ({low}, {high}): {int((~inside).sum())} rows dropped")
        data = data[inside]
    converted = []
    for c in [integer] if isinstance(integer, str) else (integer or []):
        s = data[_columns(data, [c])[0]]
        if pd.api.types.is_integer_dtype(s):
            continue
        fractional = s.dropna()[np.mod(s.dropna(), 1) != 0]
        if len(fractional):
            raise ValueError(f"{c} has {len(fractional)} value(s) with decimals, e.g. {fractional.iloc[0]}; "
                             "they cannot be stored as whole numbers")
        data[c] = s.astype("int64" if s.notna().all() else "Int64")
        converted.append(c)
    if converted:
        print(f"Stored as whole numbers: {converted}")
    print(f"Clean dataset: {len(data)} of {start} rows kept")
    data.attrs = dict(df.attrs)
    return _changed(data, df, len(data) < start or bool(converted))


def combine(df, keys, sum=None, lowest=None):
    """Merge rows that are the same item (e.g. versions of one game) into one row.

    keys:   columns that together identify the item, e.g. ["Name", "Platform", "Publisher"].
            Rows with a missing key value are left alone, so unrelated rows are never merged.
    sum:    numeric columns to add up (e.g. sales); None = every numeric column not listed
            in keys or lowest.
    lowest: columns that keep their smallest value (e.g. "Year" = first release, "Rank").
    Every other column keeps the value of the group's first row.
    """
    keys = [_column(df, k) for k in ([keys] if isinstance(keys, str) else keys)]
    lowest = [_column(df, c) for c in ([lowest] if isinstance(lowest, str) else (lowest or []))]
    if sum is None:
        sum = [c for c in df.select_dtypes("number").columns if c not in keys + lowest]
    sum = _columns(df, sum)

    complete = df[keys].notna().all(axis=1)
    sizes = df[complete].groupby(keys, sort=False)[keys[0]].transform("size")
    repeated = df[complete][sizes > 1]
    if repeated.empty:
        print(f"No rows share the same {keys}: nothing to combine")
        return _changed(df.copy(), df, False)

    how = {c: "first" for c in df.columns if c not in keys}
    how.update({c: "sum" for c in sum})
    how.update({c: "min" for c in lowest})
    groups = repeated.groupby(keys, sort=False)
    merged = groups.agg(how).reset_index()[df.columns]
    merged.index = groups.head(1).index  # each group takes the place of its first row
    out = pd.concat([df.drop(index=repeated.index), merged]).sort_index()

    print(f"Combined {len(repeated)} rows into {len(merged)} (same {keys}); "
          f"rows: {len(df)} -> {len(out)}")
    print(f"Added up: {sum}" + (f" | kept the lowest: {lowest}" if lowest else ""))
    shown = merged[keys + lowest + sum].head(10)
    print(shown.to_string(index=False))
    out.attrs = dict(df.attrs)
    return _changed(out, df, True)


def prepare(df, target="auto", features=None, drop=(), missing="drop"):
    """Build the working dataset used by all later steps.

    target:   "auto" (last column if it looks like labels), a column name, or None.
    features: list of columns; None = all numeric columns except the target (constant ones skipped).
    drop:     columns to exclude, e.g. ["id"].
    missing:  "drop" rows, fill with "mean" / "median", or None to leave as-is.
    """
    data = df.drop(columns=[c for c in drop if c in df.columns])

    if target == "auto":
        last = data.columns[-1]
        n_unique = data[last].nunique(dropna=True)
        target = last if 1 < n_unique <= 20 and n_unique < len(data) else None
    elif target is not None and target not in data.columns:
        raise KeyError(f"Target '{target}' not found. Available: {list(data.columns)}")

    if features is None:
        numeric = [c for c in data.select_dtypes("number").columns if c != target]
        constant = [c for c in numeric if data[c].nunique(dropna=True) <= 1]
        if constant:
            print(f"Skipping constant column(s): {constant}")
        features = [c for c in numeric if c not in constant]
    features = _columns(data, features)
    if target in features:
        raise ValueError(f"The target '{target}' cannot also be a feature")

    n_missing = int(data[features].isna().sum().sum())
    if missing == "drop":
        data = data.dropna(subset=features)
    elif missing in ("mean", "median"):
        data[features] = data[features].fillna(data[features].agg(missing))
    elif missing is not None:
        raise ValueError('missing must be "drop", "mean", "median" or None')

    data.attrs = {**df.attrs, "target": target, "features": features}
    print(f"Target: {target}")
    if target is not None:
        print(f"Class counts: {data[target].value_counts().to_dict()}")
    print(f"Features ({len(features)}): {features}")
    print(f"Missing feature values: {n_missing} (handling: {missing})")
    print(f"Working dataset: {data.shape[0]} rows")
    filled = missing in ("mean", "median") and n_missing > 0
    return _changed(data, df, data.shape != df.shape or filled)


# ---------------------------------------------------------------- Part 3: transformation

def transform(df, method, columns=None):
    """Return the selected columns transformed with "minmax", "standard", "robust" or "log"."""
    cols = _columns(df, columns)
    x = df[cols]
    if method == "minmax":    # rescale to [0, 1]
        out = (x - x.min()) / (x.max() - x.min()).replace(0, 1)
    elif method == "standard":  # z-score: mean 0, std 1
        out = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    elif method == "robust":  # center on median, scale by IQR
        out = (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)).replace(0, 1)
    elif method in ("log", "log10"):  # log(1 + x); columns with negatives are shifted to start at 0
        shift = (-x.min()).clip(lower=0)
        if (shift > 0).any():
            print(f"Shifted before log: {list(shift[shift > 0].index)}")
        out = np.log1p(x + shift) if method == "log" else np.log10(1 + x + shift)
    else:
        raise ValueError(f"method must be one of {TRANSFORMS}")
    out.attrs = {**df.attrs, "features": cols}
    target = _target(df)
    if target is not None:  # keep the target so later steps can still color by it
        out[target] = df[target]
    return out


def add_transformed(df, method, columns=None, suffix=None):
    """Return a copy of df with transformed columns added next to the originals.

    New columns are named "<column>_<suffix>" (suffix defaults to the method, e.g. Global_Sales_log)
    and become features, so the later steps include them.
    """
    cols = _columns(df, columns)
    transformed = transform(df, method, cols)
    out = df.copy()
    new = []
    for c in cols:
        name = f"{c}_{suffix or method}"
        out[name] = transformed[c]
        new.append(name)
    out.attrs = {**df.attrs, "features": [f for f in _features(df) if f not in new] + new}
    print(f"Added: {new}")
    return _changed(out, df, True)


# ---------------------------------------------------------------- Part 4: descriptive analysis

def describe(df, columns=None):
    """Central tendency, spread, range and quartiles per column."""
    x = df[_columns(df, columns)]
    q1, q2, q3 = x.quantile(0.25), x.quantile(0.50), x.quantile(0.75)
    return pd.DataFrame({
        "mean": x.mean(), "median": x.median(), "mode": x.mode().iloc[0],
        "std": x.std(), "variance": x.var(),
        "min": x.min(), "max": x.max(), "range": x.max() - x.min(),
        "Q1": q1, "Q2": q2, "Q3": q3, "IQR": q3 - q1,
    })


def top(df, by=None, n=10, columns=None, compare=None):
    """The n rows with the highest value in `by`, and how much they stand out.

    columns: extra columns to show; None = the first 3 text columns (e.g. name, platform, genre).
    compare: a transformation ("log", "log10", ...) to show next to the raw values, to see
             whether the top rows still dominate after transforming.
    Columns added: pct_of_total (share of the column total) and x_median (times the median row).
    """
    by = _value(df, by)
    if columns is None:
        columns = [c for c in df.select_dtypes(include=TEXT_TYPES).columns][:3]
    columns = [_column(df, c) for c in ([columns] if isinstance(columns, str) else columns) if c != by]

    def stand_out(values, label):
        total, median = values.sum(), values.median()
        ranked = values.loc[rows.index]
        rows[label] = ranked
        rows[f"pct_of_total{label[len(by):]}"] = (ranked / total * 100).round(2)
        rows[f"x_median{label[len(by):]}"] = (ranked / median).round(1) if median else np.nan
        times = f", #1 is {ranked.iloc[0] / median:.1f}x the median" if median else ""
        print(f"Top {n} {label}: {ranked.sum() / total:.1%} of the total{times}")

    rows = df.nlargest(n, by)[columns].copy()
    stand_out(df[by], by)
    if compare:
        stand_out(transform(df, compare, [by])[by], f"{by}_{compare}")
    rows.index = pd.RangeIndex(1, len(rows) + 1, name="rank")
    return rows


# ---------------------------------------------------------------- Part 5: relationships

def relationship(df, x=None, y=None, plot=True, color_by="target", save_as=None):
    """Pearson, Spearman and covariance between two columns, plus a scatter plot.

    x, y: column names; None = the first / second feature column.
    """
    features = _features(df)
    if (x is None or y is None) and len(features) < 2:
        raise ValueError("Need at least two feature columns; pass x and y explicitly")
    x, y = _columns(df, [x or features[0], y or features[1]])
    pair = df[[x, y]].dropna()
    pearson = pair[x].corr(pair[y], method="pearson")
    spearman = pair[x].corr(pair[y], method="spearman")
    covariance = pair[x].cov(pair[y])
    print(f"{x} vs {y} ({len(pair)} rows)")
    print(f"  Pearson r     = {pearson:.4f}  ({_strength(pearson)}, linear)")
    print(f"  Spearman rho  = {spearman:.4f}  ({_strength(spearman)}, monotonic)")
    print(f"  Covariance    = {covariance:.4f}")
    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        _scatter(ax, df, _color_column(df, color_by), "tab:blue", df[x], df[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}  (r = {pearson:.2f})")
        _finish(fig, save_as)


# ---------------------------------------------------------------- Part 6: outliers

def outliers(df, columns=None, method="iqr", factor=1.5, threshold=3.0):
    """Detect outliers per column. Returns (summary table, rows containing outliers).

    method="iqr":    outside [Q1 - factor*IQR, Q3 + factor*IQR]
    method="zscore": |z| > threshold
    """
    x = df[_columns(df, columns)]
    lower, upper = _bounds(x, method, factor, threshold)
    mask = x.lt(lower) | x.gt(upper)
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        spread = {"Q1": q1, "Q3": q3, "IQR": q3 - q1}
    else:
        spread = {"mean": x.mean(), "std": x.std()}
    table = pd.DataFrame({
        **spread, "lower_bound": lower, "upper_bound": upper,
        "n_outliers": mask.sum(), "pct_outliers": (mask.mean() * 100).round(2),
    }).sort_values("n_outliers", ascending=False)
    rows = df[mask.any(axis=1)]
    print(f"Method: {method} | rows with at least one outlier: {len(rows)} of {len(df)}")
    return table, rows


def handle_outliers(df, columns=None, action="keep", method="iqr", factor=1.5, threshold=3.0):
    """Return a copy of df with outliers handled.

    action: "keep" (no change), "remove" (drop rows with an outlier in any selected column),
            "median" (replace outlier values by the column median) or
            "clip" (cap outlier values at the lower / upper bound).
    """
    cols = _columns(df, columns)
    x = df[cols]
    lower, upper = _bounds(x, method, factor, threshold)
    mask = x.lt(lower) | x.gt(upper)
    out = df.copy()
    if action == "remove":
        out = out[~mask.any(axis=1)]
    elif action == "median":
        out[cols] = x.mask(mask, x.median(), axis=1)
    elif action == "clip":
        out[cols] = x.clip(lower, upper, axis=1)
    elif action != "keep":
        raise ValueError('action must be "keep", "remove", "median" or "clip"')
    out.attrs = dict(df.attrs)
    print(f"Outlier values ({method}): {int(mask.sum().sum())} in {int(mask.any(axis=1).sum())} rows "
          f"| action: {action} | rows now: {len(out)} of {len(df)}")
    return _changed(out, df, action != "keep" and bool(mask.any().any()))


def plot_outliers(df, columns=None, factor=1.5, row=None, save_as=None):
    """Boxplots on standardized values so all columns share one scale.

    row: index label of one row to highlight (e.g. from the rows returned by outliers()).
         Its values are drawn as stars: red where it is an outlier, black elsewhere.
    """
    cols = _columns(df, columns)
    x = df[cols]
    z = (x - x.mean()) / x.std().replace(0, 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(cols) + 2), 5))
    ax.boxplot(z.to_numpy(), whis=factor, flierprops={"markerfacecolor": "tab:red", "markersize": 4})
    ax.set_xticks(range(1, len(cols) + 1), cols, rotation=90)
    ax.set_ylabel("Standardized value (z)")
    ax.set_title("Outliers per column")
    if row is not None:
        if row not in df.index:
            raise KeyError(f"Row {row!r} not found in the index")
        q1, q3 = z.quantile(0.25), z.quantile(0.75)
        values = z.loc[row]
        is_out = values.lt(q1 - factor * (q3 - q1)) | values.gt(q3 + factor * (q3 - q1))
        positions = np.arange(1, len(cols) + 1)
        ax.scatter(positions[~is_out], values[~is_out], marker="*", s=150, color="black",
                   zorder=3, label=f"row {row}")
        ax.scatter(positions[is_out], values[is_out], marker="*", s=150, color="tab:red",
                   edgecolor="black", zorder=3, label=f"row {row} (outlier)")
        ax.legend()
        ax.set_title(f"Row {row} against the other values")
        print(f"Row {row} is an outlier in: {list(values.index[is_out]) or 'no selected column'}")
    _finish(fig, save_as)


# ---------------------------------------------------------------- Part 7: visualization

def plot(df, columns=None, kind="hist", color="tab:blue", color_by="target", title=None, bins=30, save_as=None):
    """Plot columns as "hist", "box", "violin", "scatter" (exactly 2 columns) or "line"."""
    cols = _columns(df, columns if columns is not None else _features(df)[:4])
    color_col = _color_column(df, color_by)

    if kind == "hist":
        n_cols = min(len(cols), 3)
        n_rows = int(np.ceil(len(cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)
        for ax, col in zip(axes.flat, cols):
            for label, mask in _groups(df, color_col):
                ax.hist(df[col][mask].dropna(), bins=bins, alpha=0.6,
                        color=color if label is None else None,
                        label=None if label is None else str(label))
            ax.set_title(col)
        for ax in axes.flat[len(cols):]:
            ax.set_visible(False)
        if color_col is not None:
            axes.flat[0].legend(title=color_col)

    elif kind in ("box", "violin"):
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cols) + 2), 5))
        values = [df[c].dropna().to_numpy() for c in cols]
        if kind == "box":
            ax.boxplot(values, patch_artist=True, boxprops={"facecolor": color, "alpha": 0.6})
        else:
            for body in ax.violinplot(values, showmedians=True)["bodies"]:
                body.set_facecolor(color)
        ax.set_xticks(range(1, len(cols) + 1), cols, rotation=45, ha="right")

    elif kind == "scatter":
        if len(cols) != 2:
            raise ValueError("A scatter plot needs exactly 2 columns, e.g. ['a', 'b']")
        fig, ax = plt.subplots(figsize=(6, 5))
        _scatter(ax, df, color_col, color, df[cols[0]], df[cols[1]])
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])

    elif kind == "line":
        fig, ax = plt.subplots(figsize=(10, 5))
        df[cols].plot(ax=ax, color=color if len(cols) == 1 else None)
        ax.set_xlabel(df.index.name or "index")

    else:
        raise ValueError('kind must be "hist", "box", "violin", "scatter" or "line"')

    fig.suptitle(title or kind.capitalize())
    _finish(fig, save_as)


# ---------------------------------------------------------------- Part 8: relationship maps

def heatmap(matrix, title="", cmap="coolwarm", vmin=None, vmax=None, fmt=".2f", save_as=None):
    """Lower-triangle heatmap (no mirrored half, no diagonal) with a value in every cell."""
    # Drop the first row and last column: they would only hold the diagonal / mirrored half.
    tri = matrix.iloc[1:, :-1]
    values = np.array(tri, dtype=float)  # a writable copy
    values[np.triu_indices_from(values, k=1)] = np.nan
    n = len(tri)
    if vmin is None or vmax is None:
        vmax = np.nanmax(np.abs(values))
        vmin = -vmax
    size = max(5, min(0.55 * n + 2, 24))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n), tri.columns, rotation=90)
    ax.set_yticks(range(n), tri.index)
    fontsize = max(6, min(11, 220 / n))
    colormap = plt.get_cmap(cmap)
    for i in range(n):
        for j in range(i + 1):
            v = values[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = colormap((v - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
            text_color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=fontsize, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _finish(fig, save_as)


def correlation_heatmap(df, columns=None, method="pearson", cmap="coolwarm", save_as=None):
    """Correlation map ("pearson" or "spearman") across the selected columns."""
    cols = _columns(df, columns, minimum=2)
    heatmap(df[cols].corr(method=method), f"{method.capitalize()} correlation",
            cmap=cmap, vmin=-1, vmax=1, save_as=save_as)


def covariance_heatmap(df, columns=None, cmap="coolwarm", save_as=None):
    """Covariance map (scale dependent: columns with large values dominate the colors)."""
    cols = _columns(df, columns, minimum=2)
    heatmap(df[cols].cov(), "Covariance", cmap=cmap, fmt=".2g", save_as=save_as)


# ---------------------------------------------------------------- Part 9: PCA

def pca(df, columns=None, n_components=3, scale="standard"):
    """Run PCA. Returns (principal components table, loadings table).

    scale: transformation applied first (None, "minmax", "standard", "robust", "log").
    """
    cols = _columns(df, columns, minimum=2)
    if n_components > len(cols):
        raise ValueError(f"n_components ({n_components}) cannot exceed the number of columns ({len(cols)})")
    x = transform(df, scale, cols)[cols] if scale else df[cols]
    if x.isna().any().any():
        raise ValueError("PCA cannot handle missing values; use prepare(..., missing='drop' / 'mean' / 'median')")

    model = PCA(n_components=n_components)
    names = [f"pc{i + 1}" for i in range(n_components)]
    components = pd.DataFrame(model.fit_transform(x.to_numpy()), columns=names, index=df.index)
    target = _target(df)
    if target is not None:
        components[target] = df[target]
    components.attrs = {"target": target, "features": names,
                        "explained": dict(zip(names, model.explained_variance_ratio_))}

    for name, ratio in components.attrs["explained"].items():
        print(f"{name}: {ratio:.2%} of variance explained")
    print(f"Total: {model.explained_variance_ratio_.sum():.2%}")

    loadings = pd.DataFrame(model.components_.T, index=cols, columns=names)
    loadings = loadings.reindex(loadings["pc1"].abs().sort_values(ascending=False).index)
    return components, loadings


def plot_pca(components, color_by="target", save_as=None):
    """2D scatter of the first two components, plus a 3D view when there are 3."""
    names = components.attrs["features"]
    explained = components.attrs["explained"]
    color_col = _color_column(components, color_by)
    labels = [f"{n.upper()} ({explained[n]:.1%})" for n in names]

    if len(names) >= 3:
        fig = plt.figure(figsize=(12, 5))
        ax2d = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
        _scatter(ax3d, components, color_col, "tab:blue", *(components[n] for n in names[:3]))
        ax3d.set_xlabel(labels[0])
        ax3d.set_ylabel(labels[1])
        ax3d.set_zlabel(labels[2])  # pyright: ignore[reportAttributeAccessIssue]
        ax3d.set_title("PCA — 3D")
    else:
        fig, ax2d = plt.subplots(figsize=(6, 5))

    _scatter(ax2d, components, color_col, "tab:blue", components[names[0]], components[names[1]])
    ax2d.set_xlabel(labels[0])
    ax2d.set_ylabel(labels[1])
    ax2d.set_title("PCA — 2D")
    _finish(fig, save_as)


# ---------------------------------------------------------------- group analysis

AGGREGATIONS = ("sum", "mean", "median", "count")


def _bar_colors(n):
    return plt.get_cmap("tab10" if n <= 10 else "tab20")(np.arange(n) % 20)


def _rotate_labels(ax):
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def aggregate(df, by=None, value=None, agg="sum", top=None, plot="auto", save_as=None):
    """Summarize `value` per group of `by`: count, sum, mean, median and share of the total.

    by:   grouping column, e.g. "Genre" or "Year"; None = the text column with the fewest values.
    agg:  what to rank and plot: "sum", "mean", "median" or "count".
    top:  keep only the n largest groups (by `agg`); None = all.
    plot: "bar", "line", None, or "auto" (line for a numeric `by` such as a year, else bar).
    A numeric `by` (e.g. a year) stays in its natural order; other groups are sorted by `agg`.
    """
    by = _category(df, by)
    value = _value(df, value)
    df = _labels(df, by)
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}")
    n_missing = int(df[by].isna().sum())
    if n_missing:
        print(f"{n_missing} rows without {by} are left out")
    g = df.groupby(by, observed=True)[value]
    table = pd.DataFrame({"count": g.count(), "sum": g.sum(), "mean": g.mean(), "median": g.median()})
    table["share_pct"] = (table["sum"] / table["sum"].sum() * 100).round(2)
    ordered = pd.api.types.is_numeric_dtype(df[by])
    if top:
        table = table.loc[table.index.isin(table[agg].nlargest(top).index)]
    if not ordered:
        table = table.sort_values(agg, ascending=False)
    table = _tidy_index(table)

    best = table[agg].nlargest(min(3, len(table)))
    print(f"{agg} of {value} per {by} ({len(table)} groups). Highest: "
          + ", ".join(f"{k} ({v:,.2f})" for k, v in best.items()))

    if plot == "auto":
        plot = "line" if ordered else "bar"
    if plot:
        fig, ax = plt.subplots(figsize=(max(6, min(0.45 * len(table) + 2, 16)), 4.5))
        if plot == "line":
            peak = table[agg].idxmax()
            ax.plot(table.index, table[agg], marker="o", markersize=3, color="tab:blue")
            ax.annotate(f"peak: {peak}", (peak, table[agg].max()), textcoords="offset points", xytext=(5, 5))
        elif plot == "bar":
            ax.bar(table.index.astype(str), table[agg], color="tab:blue")
            _rotate_labels(ax)
        else:
            raise ValueError('plot must be "bar", "line", "auto" or None')
        ax.set_xlabel(by)
        ax.set_ylabel(f"{agg} of {value}")
        ax.set_title(f"{agg.capitalize()} of {value} per {by}")
        _finish(fig, save_as)
    return table


def pivot(df, rows=None, columns=None, value=None, agg="sum", top_rows=None, top_columns=None,
          share=None, plot="stacked", save_as=None):
    """Cross table of `value` for every combination of two grouping columns, plus a chart.

    rows / columns: the two grouping columns (e.g. "Platform" and "Genre"); None = the two
                    text columns with the fewest values.
    top_rows / top_columns: keep only the n largest groups of each; None = all.
    share:  None = the plain values; "rows" = each row as % of its total
            ("which genres make up this platform"); "columns" = each column as %.
    plot:   "stacked" bars, "grouped" bars, "heatmap", "line" (handy when rows are years) or None.
    """
    rows = _category(df, rows)
    columns = _category(df, columns, exclude=[rows])
    value = _value(df, value)
    if rows == columns:
        raise ValueError("rows and columns must be different columns")
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}")
    df = _labels(df, rows, columns)
    data = df.dropna(subset=[rows, columns])
    if top_rows:
        keep = data.groupby(rows, observed=True)[value].agg(agg).nlargest(top_rows).index
        data = data[data[rows].isin(keep)]
    if top_columns:
        keep = data.groupby(columns, observed=True)[value].agg(agg).nlargest(top_columns).index
        data = data[data[columns].isin(keep)]

    table = data.pivot_table(index=rows, columns=columns, values=value, aggfunc=agg, observed=True)
    if agg in ("sum", "count"):
        table = table.fillna(0)
    if not pd.api.types.is_numeric_dtype(df[rows]):
        table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    table = table[table.sum(axis=0).sort_values(ascending=False).index]
    if share == "rows":
        table = table.div(table.sum(axis=1), axis=0) * 100
    elif share == "columns":
        table = table.div(table.sum(axis=0), axis=1) * 100
    elif share is not None:
        raise ValueError('share must be None, "rows" or "columns"')
    table = _tidy_index(table.round(2))
    label = f"{agg} of {value}" + (f" (% of each {rows if share == 'rows' else columns})" if share else "")
    title = f"{label} per {rows} and {columns}"

    if plot in ("stacked", "grouped", "line"):
        per_row = 0.5 if plot != "grouped" else 0.2 * len(table.columns)
        fig, ax = plt.subplots(figsize=(max(7, min(per_row * len(table) + 3, 18)), 5))
        colors = _bar_colors(len(table.columns))
        if plot == "line":
            table.plot(ax=ax, color=colors, marker="o", markersize=2)
        else:
            table.plot(ax=ax, kind="bar", stacked=plot == "stacked", color=colors, width=0.8)
            _rotate_labels(ax)
        ax.set_ylabel(label)
        ax.legend(title=columns, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(title)
        _finish(fig, save_as)
    elif plot == "heatmap":
        values = table.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(table.columns) + 3), max(4, 0.4 * len(table) + 2)))
        im = ax.imshow(values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(table)), table.index)
        vmax = np.nanmax(values) if np.isfinite(values).any() else 1
        for i, j in np.ndindex(values.shape):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3g}", ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel(columns)
        ax.set_ylabel(rows)
        ax.grid(False)
        ax.set_title(title)
        _finish(fig, save_as)
    elif plot is not None:
        raise ValueError('plot must be "stacked", "grouped", "heatmap", "line" or None')
    return table


def simpson(df, group=None, by=None, value=None, agg="mean", min_count=10, min_gap_pct=5, pair=None,
            plot=True, save_as=None):
    """Look for Simpson's paradox: group A beats group B overall, but B beats A inside most subgroups.

    group: the column whose groups are compared (e.g. "Genre").
    by:    the column that splits the data into subgroups (e.g. "Platform").
    agg:   "mean" or "median" of `value` per group.
    min_count: a subgroup only counts when both groups have at least this many rows in it.
    min_gap_pct: A must beat B overall by at least this many percent, so near-ties are skipped.
    pair:  two groups to show in detail, e.g. ("Action", "Puzzle"); None = the strongest reversal.
    Returns one row per reversed pair, strongest first.
    """
    group = _category(df, group)
    by = _category(df, by, exclude=[group])
    value = _value(df, value)
    if agg not in ("mean", "median"):
        raise ValueError('agg must be "mean" or "median"')
    df = _labels(df, group, by)
    data = df.dropna(subset=[group, by, value])
    overall = data.groupby(group, observed=True)[value].agg(agg)
    within = data.pivot_table(index=by, columns=group, values=value, aggfunc=agg, observed=True)
    counts = data.pivot_table(index=by, columns=group, values=value, aggfunc="count", observed=True)
    within = within.where(counts >= min_count)

    found = []
    for a in overall.index:
        for b in overall.index:
            if overall[a] <= overall[b] * (1 + min_gap_pct / 100):
                continue  # each pair once, with a = the clear overall winner
            both = within[[a, b]].dropna()
            if len(both) < 2:
                continue
            loser_wins = int((both[b] > both[a]).sum())
            if loser_wins > len(both) / 2:
                found.append({"overall_winner": a, "overall_loser": b,
                              f"winner_{agg}": overall[a], f"loser_{agg}": overall[b],
                              "overall_gap_pct": round(100 * (overall[a] / overall[b] - 1), 1),
                              f"{by}_compared": len(both), "loser_wins_in": loser_wins,
                              "reversal_pct": round(100 * loser_wins / len(both), 1)})
    table = pd.DataFrame(found)
    print(f"Compared the {agg} {value} of every pair of {group} groups inside each {by} "
          f"(only {by} values with at least {min_count} rows of both, overall gap at least {min_gap_pct}%).")
    if table.empty:
        print("No Simpson's paradox found. Try another `by`, `agg`, or a lower `min_count` / `min_gap_pct`.")
        return table
    table = table.sort_values(["reversal_pct", f"{by}_compared"], ascending=False, ignore_index=True)
    print(f"Reversed pairs found: {len(table)}")

    if pair:
        a, b = (str(p) if isinstance(p, (bool, np.bool_)) else p for p in pair)
    else:
        a, b = table.loc[0, "overall_winner"], table.loc[0, "overall_loser"]
    for g in (a, b):
        if g not in overall.index:
            raise KeyError(f"'{g}' is not a group of {group}")
    detail = pd.DataFrame({a: within[a], b: within[b], f"n_{a}": counts[a], f"n_{b}": counts[b]})
    detail = detail.dropna(subset=[a, b])
    detail.index = detail.index.astype(str)
    detail.loc["Overall"] = [overall[a], overall[b], (data[group] == a).sum(), (data[group] == b).sum()]
    detail[[f"n_{a}", f"n_{b}"]] = detail[[f"n_{a}", f"n_{b}"]].astype(int)
    detail["higher"] = np.where(detail[a] > detail[b], a, b)
    n_sub = len(detail) - 1
    print(f"\n{a} vs {b}: overall, {a} has the higher {agg} ({overall[a]:.3f} vs {overall[b]:.3f}), "
          f"but {b} is higher in {int((detail['higher'].iloc[:-1] == b).sum())} of {n_sub} {by} subgroups.")
    print(f"Look at the counts (n_): the overall {agg} depends on how each group is spread over {by}.")
    if plot:
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(detail) + 2), 4.5))
        x = np.arange(len(detail))
        ax.bar(x - 0.2, detail[a], 0.4, label=a, color="tab:blue")
        ax.bar(x + 0.2, detail[b], 0.4, label=b, color="tab:orange")
        ax.axvline(n_sub - 0.5, color="grey", linestyle="--")
        ax.set_xticks(x, detail.index, rotation=45, ha="right")
        ax.set_ylabel(f"{agg} of {value}")
        ax.set_xlabel(by)
        ax.legend(title=group)
        ax.set_title(f"Simpson's paradox check: {a} vs {b} per {by}, and overall")
        _finish(fig, save_as)
    print(detail.round(3).to_string())
    return table


def concentration(df, group=None, by=None, value=None, top=5):
    """How much of each group's total is held by its `top` largest `by` values.

    E.g. group="Genre", by="Publisher": the share of each genre's sales that its 5 biggest
    publishers own. Low = an open market, high = dominated by a few players.
    """
    group = _category(df, group)
    by = _category(df, by, exclude=[group])
    value = _value(df, value)
    data = _labels(df, group, by).dropna(subset=[group, by])
    share_col = f"top{top}_share_pct"
    rows = []
    for g, part in data.groupby(group, observed=True):
        totals = part.groupby(by, observed=True)[value].sum().sort_values(ascending=False)
        total = totals.sum()
        rows.append({group: g, "count": len(part), f"n_{by}": len(totals),
                     share_col: round(100 * totals.head(top).sum() / total, 1) if total else np.nan,
                     "leaders": ", ".join(map(str, totals.index[:3]))})
    table = pd.DataFrame(rows).set_index(group).sort_values(share_col)
    print(f"Share of each {group}'s {value} held by its {top} largest {by} values "
          "(low = open market, high = dominated by a few).")
    return table
