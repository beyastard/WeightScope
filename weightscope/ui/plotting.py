"""
WeightScope - Plotting helpers

All Plotly figure-creation functions live here so that tab modules stay
focused on UI layout and event-wiring only.

---

Copyright (C) 2026 Bryan K Reinhart & BeySoft

This file is part of WeightScope.

WeightScope is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

WeightScope is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public
License along with WeightScope. If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..config import MAX_UNIQUE_FOR_PLOT


# ─── Histogram ────────────────────────────────────────────────────────────────

def create_histogram(
    df: pd.DataFrame,
    value_min: float,
    value_max: float,
    log_counts: bool = True,
    nbins: int = 200,
) -> go.Figure:
    """Return a Plotly histogram of weight-value frequencies."""
    filtered = df[(df["value"] >= value_min) & (df["value"] <= value_max)]

    if len(filtered) == 0:
        return _empty_fig("No data in selected range")

    if len(filtered) > MAX_UNIQUE_FOR_PLOT:
        filtered = filtered.sample(n=MAX_UNIQUE_FOR_PLOT, random_state=42)

    fig = px.histogram(
        filtered,
        x="value",
        y="count",
        nbins=nbins,
        log_y=log_counts,
        title=f"Weight Distribution  [{value_min:.3f}, {value_max:.3f}]",
        labels={"value": "Weight Value", "count": "Frequency"},
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


# ─── Scatter ──────────────────────────────────────────────────────────────────

def create_scatter(
    df: Optional[pd.DataFrame],
    show_singletons: bool = False,
    show_outliers: bool = False,
) -> go.Figure:
    """
    Return a scatter plot of value vs. occurrence count (log-y).

    Applies independent filter masks and stratified sampling to preserve
    low-count values even when the dataset is large.
    """
    if df is None or len(df) == 0:
        return _empty_fig("No data loaded")

    df_plot = df.copy()

    # Build filter mask --------------------------------------------------------
    if show_singletons and not show_outliers:
        mask = df_plot["count"] == 1

    elif show_outliers and not show_singletons:
        Q1  = df_plot["count"].quantile(0.25)
        Q3  = df_plot["count"].quantile(0.75)
        IQR = Q3 - Q1
        lo  = max(0, Q1 - 3 * IQR)
        hi  = Q3 + 3 * IQR
        mask = (df_plot["count"] < lo) | (df_plot["count"] > hi)

    elif show_singletons and show_outliers:
        Q1  = df_plot["count"].quantile(0.25)
        Q3  = df_plot["count"].quantile(0.75)
        IQR = Q3 - Q1
        lo  = max(0, Q1 - 3 * IQR)
        hi  = Q3 + 3 * IQR
        mask = (
            (df_plot["count"] == 1)
            | (df_plot["count"] < lo)
            | (df_plot["count"] > hi)
        )

    else:
        mask = pd.Series([True] * len(df_plot), index=df_plot.index)

    df_plot = df_plot[mask]

    if len(df_plot) == 0:
        return _empty_fig("No data matching current filters")

    # Stratified sampling – keep low-count rows intact -------------------------
    if len(df_plot) > MAX_UNIQUE_FOR_PLOT:
        low   = df_plot[df_plot["count"] <= 10]
        high  = df_plot[df_plot["count"] > 10]
        quota = MAX_UNIQUE_FOR_PLOT - len(low)
        if quota > 0:
            high_s  = high.sample(n=min(quota, len(high)), random_state=42)
            df_plot = pd.concat([low, high_s])
        else:
            df_plot = df_plot.sample(n=MAX_UNIQUE_FOR_PLOT, random_state=42)

    hover = ["bit_pattern"] if "bit_pattern" in df_plot.columns else None

    fig = px.scatter(
        df_plot,
        x="value",
        y="count",
        hover_data=hover,
        log_y=True,
        title="Value vs. Count  (log scale)",
        color="count",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


# ─── Comparison ───────────────────────────────────────────────────────────────

def create_comparison_plot(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "Model A",
    name2: str = "Model B",
) -> go.Figure:
    """Return an overlaid histogram comparing two model weight distributions."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=df1["value"], y=df1["count"], name=name1, opacity=0.6, nbinsx=100)
    )
    fig.add_trace(
        go.Histogram(x=df2["value"], y=df2["count"], name=name2, opacity=0.6, nbinsx=100)
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        title="Model Comparison: Weight Distribution",
        xaxis_title="Weight Value",
        yaxis_title="Frequency",
        barmode="overlay",
    )
    return fig


# ─── Plot saving ──────────────────────────────────────────────────────────────

def save_figure(fig: go.Figure, dest: Path, fmt: str) -> None:
    """
    Save *fig* to *dest* in the requested format.

    Parameters
    ----------
    fig  : Plotly Figure object
    dest : Full output path including filename and extension
    fmt  : ``"png"``, ``"svg"``, or ``"html"``

    Raises
    ------
    ValueError
        If *fmt* is not one of the supported formats.
    ImportError
        If ``kaleido`` is not installed and a raster/vector format is requested.
    """
    fmt = fmt.lower().strip()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        fig.write_html(str(dest), include_plotlyjs="cdn")
        return

    if fmt in ("png", "svg", "pdf", "eps"):
        try:
            fig.write_image(str(dest), format=fmt, scale=2 if fmt == "png" else 1)
        except Exception as exc:
            err = str(exc).lower()
            
            # kaleido ≥ 1.0 requires Chrome; 0.2.1 works headlessly
            if "chrome" in err or "chromium" in err:
                raise ImportError(
                    f"Saving {fmt.upper()} failed: kaleido ≥ 1.0 requires Google Chrome. "
                    "Use kaleido 0.2.1 instead:  pip install 'kaleido==0.2.1'"
                ) from exc
            
            if "kaleido" in err or "orca" in err:
                raise ImportError(
                    f"Saving {fmt.upper()} requires kaleido 0.2.1: "
                    "pip install 'kaleido==0.2.1'"
                ) from exc
            raise
        return

    raise ValueError(f"Unsupported plot format: {fmt!r}.  Choose 'png', 'svg', or 'html'.")


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14))
    fig.update_layout(template="plotly_white", height=400)
    return fig
