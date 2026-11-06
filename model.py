"""Regression model behind Step 6 of main.ipynb: fit y = f(x) with polynomial curves.

Works on any dataset: pick one numeric input column (x) and one numeric output column (y).
The flow is clean() -> linearity() -> fit() -> plot_fits() -> predict().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy import stats

import toolkit as tk

MODEL_NAMES = {1: "linear", 2: "quadratic", 3: "cubic"}
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
SUPERSCRIPTS = {2: "²", 3: "³"}


# ---------------------------------------------------------------- helpers

def _name(degree):
    return MODEL_NAMES.get(degree, f"degree {degree}")


def _check(df, x, y):
    for label, col in (("X (input column)", x), ("Y (output column)", y)):
        if col is None or col == "":
            numeric = list(df.select_dtypes("number").columns)
            raise ValueError(f"Set {label} to a column name, e.g. one of {numeric}")
    for col in (x, y):
        if col not in df.columns:
            raise KeyError(f"Unknown column '{col}'. Available: {list(df.columns)}")
    if x == y:
        raise ValueError("x and y must be different columns")


def _xy(df, x, y):
    """Validate the two column names and return x and y as float arrays."""
    _check(df, x, y)
    try:
        return df[x].to_numpy(dtype=float), df[y].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"{x} and {y} must be numeric; run clean() first to drop text values") from None


def _poly(xv, yv, degree):
    """Least-squares polynomial, with coefficients for the original (unscaled) x."""
    if len(xv) < degree + 2:
        raise ValueError(f"A degree-{degree} fit needs at least {degree + 2} rows, got {len(xv)}")
    if np.ptp(xv) == 0:
        raise ValueError("x has only one value, so no curve can be fitted")
    return Polynomial.fit(xv, yv, degree).convert()


def _rss(poly, xv, yv):
    return float(np.sum((yv - poly(xv)) ** 2))


def _f_test(rss_small, rss_big, extra_terms, df_big):
    """F-test: does the bigger (nested) model explain significantly more than the smaller one?"""
    if rss_big == 0:
        return np.inf, 0.0
    f = ((rss_small - rss_big) / extra_terms) / (rss_big / df_big)
    return f, float(stats.f.sf(f, extra_terms, df_big))


def _equation(coefs):
    """y = a + b·x + c·x² ... with 4 significant digits."""
    text = f"y = {coefs[0]:.4g}"
    for power, c in enumerate(coefs[1:], start=1):
        term = "x" if power == 1 else f"x{SUPERSCRIPTS.get(power, f'^{power}')}"
        text += f" {'-' if c < 0 else '+'} {abs(c):.4g}·{term}"
    return text


# ---------------------------------------------------------------- 6.2 clean

def clean(df, x, y, duplicates=True, x_range=None, y_range=None, outliers="iqr",
          factor=1.5, threshold=3.0, plot=True, save_as=None):
    """Keep only the rows that are fit for modelling y from x. Looks at x and y only.

    1. values that are not numbers become missing; rows missing x or y are dropped
    2. duplicates=True drops fully identical rows
    3. x_range / y_range = (min, max) keep values inside the range (inclusive); None = no limit
    4. outliers = "iqr" or "zscore" drops outliers in x or y; None keeps them
    The original df is not changed; a cleaned copy is returned.
    """
    _check(df, x, y)
    data = df.copy()
    for col in (x, y):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().all():
            raise TypeError(f"'{col}' has no numeric values; pick a numeric column")
    start = len(data)
    removed = []

    def drop(keep, label):
        nonlocal data
        removed.append((label, int((~keep).sum())))
        data = data[keep]

    drop(data[[x, y]].notna().all(axis=1).to_numpy(), "missing or not a number")
    if duplicates:
        drop(~data.duplicated().to_numpy(), "duplicate rows")
    for col, rng in ((x, x_range), (y, y_range)):
        if rng is not None:
            low, high = rng
            s = data[col]
            drop((((low is None) | (s >= low)) & ((high is None) | (s <= high))).to_numpy(),
                 f"{col} outside {rng}")
    if outliers:
        _, rows = tk.outliers(data, [x, y], outliers, factor, threshold)
        drop(~data.index.isin(rows.index), f"outliers ({outliers})")

    for label, n in removed:
        print(f"  {label:35s} -{n}")
    print(f"Kept {len(data)} of {start} rows")
    if len(data) < 3:
        raise ValueError("Too few rows left to fit a model; loosen the cleaning settings")

    if plot:
        gone = df.index.difference(data.index)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(data[x], data[y], s=15, alpha=0.5, color="grey", label=f"kept ({len(data)})")
        ax.scatter(pd.to_numeric(df.loc[gone, x], errors="coerce"),
                   pd.to_numeric(df.loc[gone, y], errors="coerce"),
                   s=20, alpha=0.7, marker="x", color="tab:red", label=f"removed ({len(gone)})")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title("Cleaning: kept and removed rows")
        ax.legend()
        tk._finish(fig, save_as)

    # Keep target/features, but not the file source: this model subset is never saved as the copy.
    data.attrs = {k: v for k, v in df.attrs.items() if k not in ("source", "modified")}
    data.attrs.update(x=x, y=y)
    return data


# ---------------------------------------------------------------- 6.3 linearity test

def linearity(df, x, y, alpha=0.05, min_r2=0.01, plot=True, save_as=None):
    """Two questions: does y depend on x at all, and is that dependence linear?

    Both are F-tests against a cubic curve. A p-value below `alpha` is significant, but a
    verdict also needs at least `min_r2` explained variation: with many rows, tiny effects
    become significant without being useful.
    """
    xv, yv = _xy(df, x, y)
    n = len(xv)
    line, cubic = _poly(xv, yv, 1), _poly(xv, yv, 3)
    tss = float(np.sum((yv - yv.mean()) ** 2))
    if tss == 0:
        raise ValueError(f"{y} has only one value, so there is nothing to explain")
    rss_line, rss_cubic = _rss(line, xv, yv), _rss(cubic, xv, yv)
    r2_line, r2_cubic = 1 - rss_line / tss, 1 - rss_cubic / tss
    gain = r2_cubic - r2_line

    _, p_depends = _f_test(tss, rss_cubic, 3, n - 4)
    _, p_curved = _f_test(rss_line, rss_cubic, 2, n - 4)
    r, p_r = stats.pearsonr(xv, yv)

    print(f"{y} vs {x} ({n} rows), significance level {alpha}")
    if p_depends >= alpha:
        verdict = (f"NO RELATIONSHIP: {y} does not depend on {x} (p = {p_depends:.3g}). "
                   f"No model can predict {y} from {x}.")
    elif r2_cubic < min_r2:
        verdict = (f"NEGLIGIBLE: the relationship is significant (p = {p_depends:.3g}) but explains only "
                   f"{r2_cubic:.1%} of the variation. Predictions will be poor.")
    elif p_curved >= alpha:
        verdict = f"LINEAR: no significant curve (p = {p_curved:.3g}). The linear model is fine."
    elif gain < min_r2:
        verdict = (f"ALMOST LINEAR: the curve is significant (p = {p_curved:.3g}) but adds only "
                   f"{gain:.1%} explained variation. The linear model is good enough.")
    else:
        verdict = (f"NOT LINEAR: the curve adds {gain:.1%} explained variation (p = {p_curved:.3g}). "
                   "Use the quadratic or cubic model.")
    print(verdict)

    if plot:
        residuals = yv - line(xv)
        grid = np.linspace(xv.min(), xv.max(), 200)
        trend = _poly(xv, residuals, 3)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(xv, residuals, s=15, alpha=0.5, color="grey", label="residuals of the line")
        ax.plot(grid, trend(grid), color="tab:red", linewidth=2, label="trend left in the residuals")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel(x)
        ax.set_ylabel(f"{y} minus linear prediction")
        ax.set_title("Linearity check: a flat red line means linear is fine")
        ax.legend()
        tk._finish(fig, save_as)

    return pd.DataFrame({
        "test": ["F-test cubic vs mean", "F-test cubic vs line", "Pearson r"],
        "statistic": [f"R² = {r2_cubic:.4f}", f"R² gain = {gain:+.4f}", f"r = {r:.4f}"],
        "p_value": [p_depends, p_curved, p_r],
    }, index=pd.Index(["Does y depend on x?", "Is the dependence linear?", "Linear correlation"],
                      name="question"))


# ---------------------------------------------------------------- 6.4 fit

def fit(df, x, y, degrees=(1, 2, 3)):
    """Least-squares polynomial fits of y on x, one row per degree.

    r2: share of the variation in y explained; adj_r2: r2 penalized for extra terms (use it to
    compare degrees); rmse: typical prediction error in the units of y.
    """
    xv, yv = _xy(df, x, y)
    n = len(xv)
    tss = float(np.sum((yv - yv.mean()) ** 2))
    rows, polys = [], {}
    for d in sorted(set(int(d) for d in degrees)):
        if d < 1:
            raise ValueError("Degrees must be 1 or higher")
        poly = _poly(xv, yv, d)
        rss = _rss(poly, xv, yv)
        r2 = 1 - rss / tss if tss else np.nan
        adj = 1 - (1 - r2) * (n - 1) / (n - d - 1) if n > d + 1 else np.nan
        coefs = np.pad(poly.coef, (0, d + 1 - len(poly.coef)))  # convert() drops zero top terms
        polys[_name(d)] = coefs
        rows.append({"model": _name(d), "degree": d, "equation": _equation(coefs),
                     "r2": r2, "adj_r2": adj, "rmse": np.sqrt(rss / n)})
    results = pd.DataFrame(rows).set_index("model")
    best = results["adj_r2"].idxmax()
    results.attrs = {"x": x, "y": y, "coefs": polys, "best": best,
                     "x_min": float(xv.min()), "x_max": float(xv.max())}
    print(f"{y} as a function of {x} ({n} rows). Best fit by adjusted R²: {best}")
    return results


def _evaluate(coefs, values):
    return Polynomial(coefs)(np.asarray(values, dtype=float))


# ---------------------------------------------------------------- 6.5 plot

def plot_fits(df, results, save_as=None):
    """The data points with each fitted curve on top."""
    x, y = results.attrs["x"], results.attrs["y"]
    xv, yv = _xy(df, x, y)
    grid = np.linspace(xv.min(), xv.max(), 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xv, yv, s=15, alpha=0.5, color="grey", label="data")
    for i, (name, coefs) in enumerate(results.attrs["coefs"].items()):
        ax.plot(grid, _evaluate(coefs, grid), color=COLORS[i % len(COLORS)], linewidth=2,
                label=f"{name} (R² = {results.loc[name, 'r2']:.3f})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.legend()
    tk._finish(fig, save_as)


# ---------------------------------------------------------------- 6.6 predict

def predict(results, values):
    """Predict y for new x values with every fitted model."""
    x, y = results.attrs["x"], results.attrs["y"]
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if values.size == 0:
        print("No values given: set VALUES to a list of x values, e.g. [1, 2, 3]")
    print(f"Predicting {y} from {x}, with each model. Best model (adjusted R²): {results.attrs['best']}")
    low, high = results.attrs["x_min"], results.attrs["x_max"]
    outside = values[(values < low) | (values > high)]
    if len(outside):
        print(f"Careful: {', '.join(f'{v:g}' for v in outside)} lie outside the data range "
              f"({low:g} to {high:g}), so those predictions are extrapolations.")
    table = pd.DataFrame({name: _evaluate(coefs, values) for name, coefs in results.attrs["coefs"].items()},
                         index=pd.Index(values, name=f"given {x}"))
    table.columns = pd.MultiIndex.from_product([[f"predicted {y}"], table.columns])
    return table
