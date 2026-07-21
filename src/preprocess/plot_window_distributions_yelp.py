"""
plot_window_distributions_yelp.py

Plots sample counts per time window at 5 resolutions (30, 60, 90, 180, 365 days)
and writes a summary stats CSV, for data/yelp_sentiment.jsonl.
Both outputs go to plots/.

Windows are fixed-width in days starting from the first review. The final window
absorbs any remainder so it may be slightly shorter or longer than the nominal
width.

Usage:
    python plot_window_distributions_yelp.py

Data discovery:
    data/yelp_sentiment.jsonl — one JSON object per line, with fields
    "id", "timestamp", "sentiment", "text"
"""

import os
import sys
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


DATA_PATH = "data/yelp_sentiment.jsonl"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

RESOLUTIONS = [30, 60, 90, 180, 365]   # days
RES_LABELS = {30: "30 days", 60: "60 days", 90: "90 days",
              180: "180 days", 365: "365 days"}

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def load_data() -> pd.DataFrame:
    """Load the Yelp sentiment JSONL into a de-duplicated DataFrame."""
    if not os.path.exists(DATA_PATH):
        sys.exit(
            f"\nCould not find data.\n"
            f"  Expected: {DATA_PATH}\n"
            "Run this script from your project root."
        )

    print(f"Loading from {DATA_PATH}...")
    records = []
    with open(DATA_PATH, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: could not parse line {line_num}: {e}")

    data = pd.DataFrame.from_records(
        records, columns=["id", "timestamp", "sentiment", "text"]
    )
    if "id" in data.columns:
        data = data.drop_duplicates(subset="id")

    return _parse_timestamps(data)


def _parse_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    """Parse review timestamps to datetimes and sort chronologically."""
    data = data.copy()
    data["review_dt"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["review_dt"])
    data = data.sort_values("review_dt").reset_index(drop=True)
    print(f"  {len(data):,} rows | "
          f"{data['review_dt'].min().date()} -> {data['review_dt'].max().date()}")
    return data


def count_windows(data: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Divide the timeline into fixed-width bins of `days` days from the first review.
    """
    ts = data["review_dt"].dt.tz_localize(None)
    t0 = ts.min().normalize()                          # midnight on day of first review
    t1 = ts.max().normalize() + pd.Timedelta(days=1)   # exclusive end

    total_days = (t1 - t0).days
    n_full = total_days // days

    # The last window absorbs the remainder, so append the true end as a final
    # boundary beyond the full-width boundaries.
    boundaries = [t0 + pd.Timedelta(days=i * days) for i in range(n_full + 1)]
    boundaries.append(t1)

    records = []
    for i in range(len(boundaries) - 1):
        w_start = boundaries[i]
        w_end = boundaries[i + 1]
        mask = (ts >= w_start) & (ts < w_end)
        records.append({
            "window_idx":   i,
            "window_start": w_start,
            "window_end":   w_end - pd.Timedelta(days=1),
            "actual_days":  (w_end - w_start).days,
            "count":        int(mask.sum()),
        })

    return pd.DataFrame(records)


def make_plot(data: pd.DataFrame):
    """Render one bar panel per resolution and save the figure."""
    n = len(RESOLUTIONS)
    fig = plt.figure(figsize=(16, 4 * n), facecolor="white")
    gs = GridSpec(n, 1, figure=fig, hspace=0.55)

    all_windows = {}
    for i, days in enumerate(RESOLUTIONS):
        wdf = count_windows(data, days)
        all_windows[days] = wdf
        color = COLORS[i]

        ax = fig.add_subplot(gs[i])

        # Make the bar slightly narrower than the window so gaps stay visible.
        bar_width = pd.Timedelta(days=days * 0.92)
        ax.bar(wdf["window_start"], wdf["count"],
               width=bar_width, align="edge",
               color=color, alpha=0.85, linewidth=0.3, edgecolor="white")

        mean_val = wdf["count"].mean()
        ax.axhline(mean_val, color="black", linewidth=1.0,
                   linestyle="--", alpha=0.6, zorder=5)
        ax.text(wdf["window_start"].iloc[-1], mean_val * 1.05,
                f"mean={mean_val:,.0f}", fontsize=7.5,
                va="bottom", ha="right", color="black", alpha=0.75)

        # Only annotate the last window's width when it differs from the nominal.
        last_days = int(wdf["actual_days"].iloc[-1])
        remainder_note = (f"  [last window = {last_days} days]"
                          if last_days != days else "")

        ax.set_title(
            f"{RES_LABELS[days]}  "
            f"(n_windows={len(wdf)}, total={wdf['count'].sum():,})"
            f"{remainder_note}",
            fontsize=11, fontweight="bold", pad=6
        )
        ax.set_ylabel("Reviews per window", fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlim(data["review_dt"].dt.tz_localize(None).min(),
                    data["review_dt"].dt.tz_localize(None).max())
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    fig.suptitle(
        "Yelp Sentiment — Sample counts per time window at different resolutions",
        fontsize=13, fontweight="bold", y=1.005
    )

    out_png = os.path.join(PLOTS_DIR, "yelp_window_distributions.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot  -> {out_png}")
    return all_windows


def make_table(all_windows: dict):
    """Build and save a per-resolution summary stats table."""
    rows = []
    for days, wdf in all_windows.items():
        c = wdf["count"]
        rows.append({
            "Resolution":      RES_LABELS[days],
            "N windows":       len(wdf),
            "Total reviews":   f"{c.sum():,}",
            "Mean / window":   f"{c.mean():,.0f}",
            "Std":             f"{c.std():,.0f}",
            "Min":             f"{c.min():,}",
            "Max":             f"{c.max():,}",
            "% windows < 1k":  f"{(c < 1000).mean() * 100:.1f}%",
            "% windows < 5k":  f"{(c < 5000).mean() * 100:.1f}%",
        })

    table_df = pd.DataFrame(rows)

    out_csv = os.path.join(PLOTS_DIR, "yelp_window_stats.csv")
    table_df.to_csv(out_csv, index=False)
    print(f"Saved table -> {out_csv}")

    print("\n" + "=" * 75)
    print(table_df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    data = load_data()
    all_windows = make_plot(data)
    make_table(all_windows)
    print("\nDone.")