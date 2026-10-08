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
TRANSFORMS = ("minmax", "standard", "robust", "log")

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
        print(f"{f.name:45s} {f.stat().st_size / 1e6:8.2f} MB")


def _detect_separator(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        sample = "".join(fh.readline() for _ in range(50))
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


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

    data.attrs.update(target=target, features=features)
    print(f"Target: {target}")
    if target is not None:
        print(f"Class counts: {data[target].value_counts().to_dict()}")
    print(f"Features ({len(features)}): {features}")
    print(f"Missing feature values: {n_missing} (handling: {missing})")
    print(f"Working dataset: {data.shape[0]} rows")
    return data


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
    elif method == "log":     # log(1 + x); columns with negatives are shifted to start at 0
        shift = (-x.min()).clip(lower=0)
        if (shift > 0).any():
            print(f"Shifted before log: {list(shift[shift > 0].index)}")
        out = np.log1p(x + shift)
    else:
        raise ValueError(f"method must be one of {TRANSFORMS}")
    out.attrs = {**df.attrs, "features": cols}
    target = _target(df)
    if target is not None:  # keep the target so later steps can still color by it
        out[target] = df[target]
    return out


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
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        lower, upper = q1 - factor * (q3 - q1), q3 + factor * (q3 - q1)
    elif method == "zscore":
        lower, upper = x.mean() - threshold * x.std(), x.mean() + threshold * x.std()
    else:
        raise ValueError('method must be "iqr" or "zscore"')
    mask = x.lt(lower) | x.gt(upper)
    table = pd.DataFrame({
        "lower_bound": lower, "upper_bound": upper,
        "n_outliers": mask.sum(), "pct_outliers": (mask.mean() * 100).round(2),
    }).sort_values("n_outliers", ascending=False)
    rows = df[mask.any(axis=1)]
    print(f"Method: {method} | rows with at least one outlier: {len(rows)} of {len(df)}")
    return table, rows


def plot_outliers(df, columns=None, factor=1.5, save_as=None):
    """Boxplots on standardized values so all columns share one scale."""
    cols = _columns(df, columns)
    x = df[cols]
    z = (x - x.mean()) / x.std().replace(0, 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(cols) + 2), 5))
    ax.boxplot(z.to_numpy(), whis=factor, flierprops={"markerfacecolor": "tab:red", "markersize": 4})
    ax.set_xticks(range(1, len(cols) + 1), cols, rotation=90)
    ax.set_ylabel("Standardized value (z)")
    ax.set_title("Outliers per column")
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
    values = tri.to_numpy(dtype=float)
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
