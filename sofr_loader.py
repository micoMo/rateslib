"""
SOFR curve builder: loads instrument definitions from a CSV and calibrates a Curve.

CSV instrument types:
  ois_m2m  - meeting-to-meeting OIS swap; effective and termination are explicit FOMC dates.
             Each row constrains one inter-meeting segment of the curve.
             NOTE (long-term): replace with serial futures (spec="usd_stir1") for monthly periods
             and quarterly futures (spec="usd_stir") for IMM-date periods once you are ready to
             handle the convexity adjustment (approx. -0.5bp * duration^2 per contract).
  irs      - standard IRS; termination is a tenor string ("2y", "10y", etc.)
  fly      - butterfly of three IRS; ref1=short wing tenor, ref2=belly tenor, ref3=long wing tenor
  spread   - switch of two IRS; ref1=short leg tenor, ref2=long leg tenor
  value    - pseudo-instrument; pins a curve value directly (used for ON fixings or DF constraints)
"""

from __future__ import annotations

from datetime import datetime as dt

import pandas as pd

from rateslib import Curve, IRS, Solver, add_tenor
from rateslib.instruments import Fly, Spread, STIRFuture, Value


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_effective(val: str, today: dt, spot: dt) -> dt:
    val = val.strip()
    if val == "today":
        return today
    if val == "spot":
        return spot
    return dt.fromisoformat(val)


def _is_empty(val) -> bool:
    """True if a CSV cell is blank / NaN."""
    if pd.isna(val):
        return True
    return str(val).strip() == ""


# ---------------------------------------------------------------------------
# Instrument factories
# ---------------------------------------------------------------------------

def _make_irs(spot: dt, tenor: str, spec: str, curve_id: str) -> IRS:
    """Helper: build an IRS leg used inside Fly / Spread."""
    return IRS(spot, tenor.strip(), spec=spec, curves=curve_id)


def _build_instrument(row: pd.Series, today: dt, spot: dt, curve_id: str):
    """
    Construct a rateslib instrument from a single CSV row.

    For ois_m2m both effective and termination are treated as explicit dates
    (ISO strings or the special keywords 'today'/'spot').  This means the
    spec's T+2 spot-start convention is overridden, so the swap starts and
    ends exactly on the supplied FOMC meeting dates.
    """
    kind = str(row["type"]).strip().lower()
    eff  = _parse_effective(str(row["effective"]), today, spot)

    if kind == "ois_m2m":
        # Meeting-to-meeting OIS: both legs anchored to explicit FOMC dates.
        # Passing datetime objects overrides the usd_irs spec's eval=2b convention.
        term = dt.fromisoformat(str(row["termination"]).strip())
        return IRS(eff, term, spec=str(row["spec"]), curves=curve_id)

    elif kind == "irs":
        # Standard outright swap; termination is a tenor string ("2y", "30y", …).
        return IRS(eff, str(row["termination"]).strip(), spec=str(row["spec"]), curves=curve_id)

    elif kind == "stir":
        # Short-term interest rate future.
        # NOTE: if re-adding futures in future, remember convexity adjustment —
        # subtract approx. 0.5bp * duration^2 from each futures rate before
        # passing as the target `s` value to the Solver.
        term = dt.fromisoformat(str(row["termination"]).strip())
        return STIRFuture(eff, term, spec=str(row["spec"]), curves=curve_id)

    elif kind == "fly":
        # Butterfly: rate = (-r_short + 2*r_belly - r_long) * 100, result in bps.
        # ref1=short wing, ref2=belly, ref3=long wing — all IRS with same spec.
        # CSV `rate` must be in bps to match (e.g. 5 for a 5bp butterfly).
        #
        # Workaround: Fly is missing _rate_scalar in rateslib 2.6.0 (Spread has it).
        # Patch it here so the Solver can read it. Value matches Spread._rate_scalar=100.0
        # since both rate() methods apply the same *100 scaling.
        fly = Fly(
            _make_irs(eff, str(row["ref1"]), str(row["spec"]), curve_id),
            _make_irs(eff, str(row["ref2"]), str(row["spec"]), curve_id),
            _make_irs(eff, str(row["ref3"]), str(row["spec"]), curve_id),
        )
        fly._rate_scalar = 100.0  # type: ignore[attr-defined]
        return fly

    elif kind == "spread":
        # Switch: rate = (r_long - r_short) * 100.
        # ref1=short leg tenor, ref2=long leg tenor.
        return Spread(
            _make_irs(eff, str(row["ref1"]), str(row["spec"]), curve_id),
            _make_irs(eff, str(row["ref2"]), str(row["spec"]), curve_id),
        )

    elif kind == "value":
        # Pseudo-instrument that pins the curve value at a single date.
        # metric options: "o/n_rate", "curve_value", "cc_zero_rate"
        return Value(
            effective=today,
            metric=str(row["metric"]).strip(),
            curves=curve_id,
        )

    else:
        raise ValueError(
            f"Unknown instrument type '{kind}' for label '{row['label']}'. "
            f"Expected one of: ois_m2m, irs, stir, fly, spread, value."
        )


# ---------------------------------------------------------------------------
# Node-date derivation
# ---------------------------------------------------------------------------

def _node_date(row: pd.Series, today: dt, spot: dt, cal: str, mod: str) -> dt:
    """
    Return the curve node date that this instrument primarily calibrates.

    One node per instrument is the fundamental requirement of the Solver —
    each instrument must introduce exactly one new degree of freedom.

    ois_m2m  -> termination date (the FOMC meeting date where the rate steps up/down).
                No business-day adjustment: we want the node exactly on the meeting date
                so that log_linear interpolation produces a flat rate within each period.
    irs      -> payment-adjusted maturity: add_tenor(spot, tenor, mod, cal).
                The 2bd payment lag in usd_irs means the last cashflow is discounted
                using a DF slightly beyond the stated tenor; the node goes there.
    stir     -> effective date of the future period (start of the futures window).
    fly      -> payment-adjusted maturity of the belly leg (ref2), the new node.
    spread   -> by default, payment-adjusted maturity of the longer leg (ref2).
                Override with node_tenor when ref2 is already introduced by an outright
                swap (e.g. sw_20y: ref2=30y is taken by irs_30y, so node_tenor=20y).
    value    -> today (pins a value at the curve's initial date).

    node_tenor column (optional): if non-empty, always overrides the auto-derived date.
    Use this to disambiguate any collision without changing the instrument definition.
    """
    # Optional explicit override — takes priority over all auto-derivation logic.
    if "node_tenor" in row.index and not _is_empty(row["node_tenor"]):
        return add_tenor(spot, str(row["node_tenor"]).strip(), mod, cal)

    kind = str(row["type"]).strip().lower()

    if kind == "value":
        return today

    elif kind == "ois_m2m":
        return dt.fromisoformat(str(row["termination"]).strip())

    elif kind == "stir":
        return dt.fromisoformat(str(row["effective"]).strip())

    elif kind == "irs":
        return add_tenor(spot, str(row["termination"]).strip(), mod, cal)

    elif kind == "fly":
        # belly is ref2 — the new node being introduced
        return add_tenor(spot, str(row["ref2"]).strip(), mod, cal)

    elif kind == "spread":
        # Default: ref2 is the longer, new leg. Override with node_tenor when ref2
        # is already covered by an outright swap in the same solver.
        return add_tenor(spot, str(row["ref2"]).strip(), mod, cal)

    raise ValueError(f"Cannot derive node date for type '{kind}'")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_sofr_solver(
    csv_path: str,
    today: dt,
    spot: dt,
    curve_id: str = "sofr",
    interpolation: str = "log_linear",
    calendar: str = "nyc",
    convention: str = "act360",
    modifier: str = "mf",
) -> tuple[Curve, Solver]:
    """
    Read a curve-definition CSV and return a calibrated (Curve, Solver) pair.

    Parameters
    ----------
    csv_path    : path to the instrument CSV file
    today       : valuation date (anchor node, DF=1.0, never calibrated)
    spot        : settlement date for spot-starting instruments (typically T+2)
    curve_id    : id string used to link instruments to the curve
    interpolation: interpolation method for the Curve (default "log_linear")
                  "log_linear" gives flat forward rates between nodes, which is
                  exactly what we want with FOMC-date nodes on the short end.
    calendar    : business day calendar for date adjustments
    convention  : day count convention
    modifier    : business day modifier for add_tenor calls

    Returns
    -------
    curve  : calibrated Curve object
    solver : Solver with convergence info in solver.result
    """
    # Read CSV, skip comment lines (rows where label starts with '#')
    df = pd.read_csv(csv_path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["label"].str.strip().str[0] != "#"].reset_index(drop=True)

    # --- Build curve nodes ---
    # today is the fixed anchor (DF=1.0); one free node per calibrating instrument.
    nodes = {today: 1.0}
    for _, row in df.iterrows():
        nd = _node_date(row, today, spot, calendar, modifier)
        nodes[nd] = 1.0   # initial guess; Solver will overwrite

    # Ensure nodes are in chronological order (dict preserves insertion order in Python 3.7+)
    nodes = dict(sorted(nodes.items()))

    # Guard: duplicate node dates mean the system is under-determined
    n_free = len(nodes) - 1   # exclude the fixed anchor at today
    if n_free != len(df):
        dupes = [nd for nd in nodes if list(nodes.keys()).count(nd) > 1]
        raise ValueError(
            f"Instrument count ({len(df)}) != free node count ({n_free}). "
            f"Likely duplicate node dates — check tenors or FOMC dates. {dupes}"
        )

    curve = Curve(
        nodes=nodes,
        interpolation=interpolation,
        calendar=calendar,
        convention=convention,
        id=curve_id,
    )

    # --- Build instruments and target rates ---
    instruments, rates, labels = [], [], []
    for _, row in df.iterrows():
        instruments.append(_build_instrument(row, today, spot, curve_id))
        rates.append(float(row["rate"]))
        labels.append(str(row["label"]).strip())

    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=rates,
        instrument_labels=labels,
        id=f"{curve_id}_solver",
    )

    return curve, solver


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    today = dt(2025, 3, 1)
    spot  = dt(2025, 3, 5)   # T+2 business days from today

    curve, solver = build_sofr_solver(
        csv_path="sofr_instruments.csv",
        today=today,
        spot=spot,
    )

    print("=== Solver result ===")
    print(solver.result)

    print("\n=== Per-instrument pricing error ===")
    print(solver.error)

    print("\n=== Calibrated nodes (discount factors) ===")
    for date, df_val in curve.nodes.nodes.items():
        print(f"  {date.date()}  DF = {float(df_val):.8f}")
