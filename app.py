"""
Bank Nifty Box Spread Opportunity Finder
========================================
Scans an NSE BANKNIFTY monthly option chain for box-spread arbitrage
(debit box and credit/reverse box) using realistic bid/ask pricing,
a configurable Indian F&O cost model and liquidity filters.

Education / research only. Not investment advice.
Run:  streamlit run app.py
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG (all constants are configurable from the UI sidebar as well)
# ----------------------------------------------------------------------------
LOT_SIZE = 30  # BANKNIFTY lot size (as specified; change here if NSE revises)

REQUIRED_COLS = [
    "strike", "ce_bid", "ce_ask", "ce_oi", "ce_volume",
    "pe_bid", "pe_ask", "pe_oi", "pe_volume", "expiry_date",
]


@dataclass
class CostConfig:
    brokerage_per_order: float = 20.0   # Rs per order (flat), 1 order per leg
    stt_sell_pct: float = 0.0125        # % of premium, sell side only
    exch_pct: float = 0.03503           # % of premium, all legs (NSE)
    sebi_per_crore: float = 10.0        # Rs per crore of premium
    gst_pct: float = 18.0               # % on brokerage + exchange + SEBI
    stamp_buy_pct: float = 0.003        # % of premium, buy side only
    exercise_stt_pct: float = 0.125     # % of intrinsic value on ITM long legs
    #                                     exercised/settled at expiry (see README)
    slippage: float = 0.05              # Rs per leg beyond bid/ask (1 tick)


@dataclass
class FilterConfig:
    min_oi: int = 500
    min_volume: int = 100
    max_spread_pct: float = 1.5         # % of mid, per leg
    min_edge_pct: float = 0.1           # min NET profit as % of strike width
    min_width: float = 200.0
    max_width: float = 1000.0


# ----------------------------------------------------------------------------
# DATA LOADING & VALIDATION
# ----------------------------------------------------------------------------
def load_data(source) -> pd.DataFrame:
    """Load an option chain from a CSV path / file-like object / DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean the raw chain. Returns (clean_df, warnings).
    Raises ValueError (friendly message) if the data is unusable.
    """
    warnings: list[str] = []
    if df is None or df.empty:
        raise ValueError("The option chain is empty. Upload a CSV with at least a few strikes.")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = df[REQUIRED_COLS].copy()
    num_cols = [c for c in REQUIRED_COLS if c != "expiry_date"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date

    before = len(df)
    df = df.dropna(subset=["strike", "expiry_date"])
    if len(df) < before:
        warnings.append(f"Dropped {before - len(df)} row(s) with missing strike/expiry.")

    before = len(df)
    df = df.drop_duplicates(subset=["expiry_date", "strike"], keep="first")
    if len(df) < before:
        warnings.append(f"Removed {before - len(df)} duplicate strike row(s).")

    if df.empty:
        raise ValueError("No usable rows after cleaning. Check strike and expiry_date columns.")
    return df.sort_values(["expiry_date", "strike"]).reset_index(drop=True), warnings


def is_monthly_expiry(d: date) -> bool:
    """True if d is the last Tuesday of its month (Bank Nifty monthly expiry)."""
    last_day = calendar.monthrange(d.year, d.month)[1]
    last = date(d.year, d.month, last_day)
    offset = (last.weekday() - 1) % 7  # Tuesday == 1
    return d == date(d.year, d.month, last_day - offset)


# ----------------------------------------------------------------------------
# COSTS
# ----------------------------------------------------------------------------
def compute_costs(legs: list[dict], width: float, cfg: CostConfig, lot: int = LOT_SIZE) -> dict:
    """
    Fees for the 4 legs (position is held to expiry, so no closing orders).

    legs: [{"side": "BUY"|"SELL", "price": execution price per unit}, ...]
    Returns a dict with a component breakdown and "total".
    """
    brokerage = stt = exch = sebi = stamp = 0.0
    for leg in legs:
        prem = leg["price"] * lot
        brokerage += cfg.brokerage_per_order
        exch += prem * cfg.exch_pct / 100
        sebi += prem * cfg.sebi_per_crore / 1e7
        if leg["side"] == "SELL":
            stt += prem * cfg.stt_sell_pct / 100
        else:
            stamp += prem * cfg.stamp_buy_pct / 100
    gst = (brokerage + exch + sebi) * cfg.gst_pct / 100
    # Long ITM legs settle at expiry: their combined intrinsic ~= strike width
    # for any expiry level inside [L, H] (and >= width outside). STT is levied
    # on the buyer on that intrinsic value.
    exercise_stt = width * lot * cfg.exercise_stt_pct / 100
    total = brokerage + stt + exch + sebi + gst + stamp + exercise_stt
    return {
        "brokerage": brokerage, "stt": stt, "exchange": exch, "sebi": sebi,
        "gst": gst, "stamp": stamp, "exercise_stt": exercise_stt, "total_fees": total,
    }


# ----------------------------------------------------------------------------
# CORE SCAN
# ----------------------------------------------------------------------------
def _leg_ok(bid: float, ask: float, max_spread_pct: float) -> bool:
    if np.isnan(bid) or np.isnan(ask) or bid <= 0 or ask <= 0 or ask < bid:
        return False
    mid = (bid + ask) / 2
    return (ask - bid) / mid * 100 <= max_spread_pct


def _strike_ok(r: pd.Series, f: FilterConfig) -> bool:
    """Strike-level liquidity + price sanity (both CE and PE must pass)."""
    vals = [r.ce_oi, r.pe_oi, r.ce_volume, r.pe_volume]
    if any(pd.isna(v) for v in vals):
        return False
    if min(r.ce_oi, r.pe_oi) < f.min_oi or min(r.ce_volume, r.pe_volume) < f.min_volume:
        return False
    return True


def _liquidity_score(rows: list[pd.Series], f: FilterConfig) -> float:
    """0-100. Half from depth (min OI/volume vs thresholds), half from tight spreads."""
    depth = min(
        min(min(r.ce_oi, r.pe_oi) / (10 * f.min_oi) for r in rows),
        min(min(r.ce_volume, r.pe_volume) / (10 * f.min_volume) for r in rows),
        1.0,
    )
    tight = []
    for r in rows:
        for b, a in ((r.ce_bid, r.ce_ask), (r.pe_bid, r.pe_ask)):
            m = (a + b) / 2
            tight.append(max(0.0, 1 - ((a - b) / m * 100) / f.max_spread_pct))
    return round(50 * depth + 50 * float(np.mean(tight)), 1)


def find_opportunities(
    chain: pd.DataFrame,
    expiry: date,
    f: FilterConfig,
    c: CostConfig,
    valuation_date: date,
    lot: int = LOT_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Evaluate every (L, H) pair for both DEBIT and CREDIT boxes.

    Returns (all_evaluated, viable, stats)
      all_evaluated: every pair that passed liquidity/price filters (with NET profit,
                     possibly negative) - used for "closest candidates".
      viable:        subset with NET profit > 0 and >= min_edge_pct of width.
    """
    d = chain[chain["expiry_date"] == expiry].sort_values("strike")
    days = (expiry - valuation_date).days
    stats = {"strikes_total": len(d), "strikes_liquid": 0, "pairs_checked": 0, "days_to_expiry": days}

    liquid = [r for _, r in d.iterrows() if _strike_ok(r, f)]
    stats["strikes_liquid"] = len(liquid)

    rows: list[dict] = []
    for i, lo in enumerate(liquid):
        for hi in liquid[i + 1:]:
            L, H = float(lo.strike), float(hi.strike)
            width = H - L
            if H <= L or width < f.min_width or width > f.max_width:
                continue
            stats["pairs_checked"] += 1

            for kind in ("DEBIT", "CREDIT"):
                if kind == "DEBIT":
                    # Buy CE@L, Sell CE@H, Buy PE@H, Sell PE@L
                    spec = [
                        ("BUY", "CE", L, lo.ce_bid, lo.ce_ask),
                        ("SELL", "CE", H, hi.ce_bid, hi.ce_ask),
                        ("BUY", "PE", H, hi.pe_bid, hi.pe_ask),
                        ("SELL", "PE", L, lo.pe_bid, lo.pe_ask),
                    ]
                else:
                    # Reverse box: Sell CE@L, Buy CE@H, Sell PE@H, Buy PE@L
                    spec = [
                        ("SELL", "CE", L, lo.ce_bid, lo.ce_ask),
                        ("BUY", "CE", H, hi.ce_bid, hi.ce_ask),
                        ("SELL", "PE", H, hi.pe_bid, hi.pe_ask),
                        ("BUY", "PE", L, lo.pe_bid, lo.pe_ask),
                    ]
                if not all(_leg_ok(b, a, f.max_spread_pct) for _, _, _, b, a in spec):
                    continue

                legs, bidask_cash, mid_cash, slip_cost = [], 0.0, 0.0, 0.0
                for side, typ, k, bid, ask in spec:
                    mid = (bid + ask) / 2
                    base = ask if side == "BUY" else bid          # realistic quote
                    exe = base + c.slippage if side == "BUY" else max(base - c.slippage, 0.0)
                    sign = 1 if side == "BUY" else -1             # + = cash out
                    bidask_cash += sign * base
                    mid_cash += sign * mid
                    slip_cost += c.slippage * lot
                    legs.append({"side": side, "type": typ, "strike": k, "bid": bid, "ask": ask,
                                 "mid": mid, "exec_price": exe})

                # bidask_cash > 0 => net debit ; < 0 => net credit
                if kind == "DEBIT":
                    net_amt = bidask_cash                  # debit paid
                    gross = (width - net_amt) * lot        # per lot, at bid/ask
                    mid_amt = mid_cash
                    mid_gross = (width - mid_amt) * lot
                else:
                    net_amt = -bidask_cash                 # credit received
                    gross = (net_amt - width) * lot
                    mid_amt = -mid_cash
                    mid_gross = (mid_amt - width) * lot

                costs = compute_costs([{"side": l["side"], "price": l["exec_price"]} for l in legs],
                                      width, c, lot)
                fees = costs["total_fees"]
                net_profit = gross - slip_cost - fees

                mid_costs = compute_costs([{"side": l["side"], "price": l["mid"]} for l in legs],
                                          width, c, lot)
                mid_net = mid_gross - mid_costs["total_fees"]

                # Capital base for ROI: debit paid (+fees) for debit box;
                # strike-width notional as a margin proxy for credit box.
                base_cap = (net_amt * lot + fees) if kind == "DEBIT" else width * lot
                roi = net_profit / base_cap * 100 if base_cap > 0 else np.nan
                ann = roi * 365 / days if days > 0 and not np.isnan(roi) else np.nan

                rows.append({
                    "Type": kind, "Expiry": expiry, "L": L, "H": H, "Strike Width": width,
                    "Net Debit/Credit": round(net_amt, 2),
                    "Gross Profit": round(gross, 2),
                    "Total Costs": round(slip_cost + fees, 2),
                    "NET Profit": round(net_profit, 2),
                    "ROI %": round(roi, 3) if not np.isnan(roi) else np.nan,
                    "Annualized %": round(ann, 2) if not np.isnan(ann) else np.nan,
                    "Liquidity Score": _liquidity_score([lo, hi], f),
                    "Mid Gross Profit": round(mid_gross, 2),
                    "Mid NET Profit": round(mid_net, 2),
                    "_legs": legs, "_costs": {**costs, "slippage": slip_cost},
                    "_edge_pct": net_profit / (width * lot) * 100,
                })

    cols = ["Type", "Expiry", "L", "H", "Strike Width", "Net Debit/Credit", "Gross Profit",
            "Total Costs", "NET Profit", "ROI %", "Annualized %", "Liquidity Score",
            "Mid Gross Profit", "Mid NET Profit", "_legs", "_costs", "_edge_pct"]
    all_df = pd.DataFrame(rows, columns=cols)
    all_df = rank(all_df)
    viable = all_df[(all_df["NET Profit"] > 0) & (all_df["_edge_pct"] >= f.min_edge_pct)]
    return all_df, viable.reset_index(drop=True), stats


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by NET profit (desc), ties broken by liquidity score."""
    if df.empty:
        return df
    return df.sort_values(["NET Profit", "Liquidity Score"], ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------------
DISCLAIMER = (
    "This tool is for education/research only. Box spread arbitrage opportunities are rare "
    "and fleeting; real execution may differ due to liquidity, slippage, and margin. "
    "Not investment advice. Consult a SEBI-registered advisor."
)
DISPLAY_COLS = ["Type", "Expiry", "L", "H", "Strike Width", "Net Debit/Credit", "Gross Profit",
                "Total Costs", "NET Profit", "ROI %", "Annualized %", "Liquidity Score"]


def _render_breakdown(row: pd.Series):
    st.markdown(f"**{row['Type']} box  {int(row['L'])} / {int(row['H'])}**  "
                f"(width {int(row['Strike Width'])})")
    legs = pd.DataFrame(row["_legs"])
    legs.columns = ["Side", "Type", "Strike", "Bid", "Ask", "Mid", "Exec price (incl. slippage)"]
    st.dataframe(legs, hide_index=True, width="stretch")
    cost = {k.replace("_", " ").title(): round(v, 2) for k, v in row["_costs"].items()}
    st.dataframe(pd.DataFrame([cost]), hide_index=True, width="stretch")
    st.caption(f"Mid-price view: gross {row['Mid Gross Profit']:.2f}, "
               f"net after fees {row['Mid NET Profit']:.2f} (no slippage, unrealistic).")


def main():
    import streamlit as st
    globals()["st"] = st

    st.set_page_config(page_title="Bank Nifty Box Spread Finder", layout="wide")
    st.title("Bank Nifty Box Spread Opportunity Finder")
    st.warning(DISCLAIMER)

    # ---- sidebar parameters ----
    sb = st.sidebar
    sb.header("Contract")
    lot = sb.number_input("Lot size", 1, 500, LOT_SIZE)
    val_date = sb.date_input("Valuation date", date.today())

    sb.header("Liquidity & safety filters")
    f = FilterConfig(
        min_oi=sb.number_input("Min OI (CE & PE)", 0, 10**7, 500, 100),
        min_volume=sb.number_input("Min volume (CE & PE)", 0, 10**7, 100, 50),
        max_spread_pct=sb.slider("Max bid-ask spread % of mid", 0.1, 10.0, 1.5, 0.1),
        min_edge_pct=sb.slider("Min NET edge % of strike width", 0.0, 5.0, 0.1, 0.05,
                               help="Replaces the 'within 10% of width' rule; see README."),
        min_width=sb.number_input("Min strike width", 0, 5000, 200, 100),
        max_width=sb.number_input("Max strike width", 0, 5000, 1000, 100),
    )

    sb.header("Transaction costs")
    c = CostConfig(
        brokerage_per_order=sb.number_input("Brokerage / order (Rs)", 0.0, 500.0, 20.0, 1.0),
        stt_sell_pct=sb.number_input("STT on sell premium (%)", 0.0, 1.0, 0.0125, 0.0025, format="%.4f"),
        exch_pct=sb.number_input("Exchange charges (% premium)", 0.0, 1.0, 0.03503, 0.0001, format="%.5f"),
        sebi_per_crore=sb.number_input("SEBI fee (Rs / crore)", 0.0, 100.0, 10.0, 1.0),
        gst_pct=sb.number_input("GST (%)", 0.0, 30.0, 18.0, 1.0),
        stamp_buy_pct=sb.number_input("Stamp duty on buy premium (%)", 0.0, 1.0, 0.003, 0.001, format="%.4f"),
        exercise_stt_pct=sb.number_input("STT on ITM exercise at expiry (% intrinsic)", 0.0, 1.0, 0.125,
                                         0.005, format="%.4f",
                                         help="Set 0 to ignore. Often the largest hidden cost of a held-to-expiry box."),
        slippage=sb.number_input("Slippage per leg (Rs)", 0.0, 10.0, 0.05, 0.05),
    )
    allow_non_monthly = sb.checkbox("Allow non-monthly expiries (testing only)", False)

    # ---- data input ----
    mode = st.radio("Data source", ["Upload CSV", "Use sample file", "Manual entry"], horizontal=True)
    raw = None
    try:
        if mode == "Upload CSV":
            up = st.file_uploader("Option chain CSV", type="csv")
            if up is not None:
                raw = load_data(up)
        elif mode == "Use sample file":
            raw = load_data("sample_option_chain.csv")
        else:
            template = pd.DataFrame([{k: None for k in REQUIRED_COLS}] * 6)
            template["expiry_date"] = ""
            raw = st.data_editor(template, num_rows="dynamic", width="stretch")
            raw = load_data(raw.dropna(how="all"))
        if raw is None:
            st.info("Provide an option chain to begin.")
            return
        chain, warns = validate(raw)
    except FileNotFoundError:
        st.error("sample_option_chain.csv not found next to app.py.")
        return
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV is empty.")
        return
    except ValueError as e:
        st.error(str(e))
        return
    for w in warns:
        st.caption(f"Note: {w}")

    expiries = sorted(chain["expiry_date"].unique())
    allowed = expiries if allow_non_monthly else [e for e in expiries if is_monthly_expiry(e)]
    if not allowed:
        st.error("No monthly expiries (last Tuesday of the month) found in the data. "
                 "Weekly expiries are not supported; tick 'Allow non-monthly expiries' to override.")
        return
    expiry = st.selectbox("Expiry", allowed, format_func=lambda d: d.strftime("%d-%b-%Y"))

    all_df, viable, stats = find_opportunities(chain, expiry, f, c, val_date, int(lot))
    st.caption(f"{stats['strikes_liquid']}/{stats['strikes_total']} strikes passed liquidity filters; "
               f"{stats['pairs_checked']} strike pairs checked; {stats['days_to_expiry']} days to expiry.")
    if stats["days_to_expiry"] <= 0:
        st.warning("Expiry is on/before the valuation date; annualized return is not computed.")

    # ---- results ----
    if viable.empty:
        st.info("No arbitrage found after costs and filters.")
        if all_df.empty:
            st.write("No strike pair passed the liquidity/spread filters, so there are no candidates to show.")
        else:
            st.subheader("Closest 3 candidates (not profitable)")
            st.dataframe(all_df.head(3)[DISPLAY_COLS], hide_index=True, width="stretch")
            result = all_df.head(3)
    else:
        st.success(f"{len(viable)} viable opportunity(ies) found.")
        st.subheader("Ranked opportunities (NET profit per lot)")
        st.dataframe(viable[DISPLAY_COLS + ["Mid NET Profit"]], hide_index=True, width="stretch")
        result = viable

    if not all_df.empty:
        st.subheader("Leg-by-leg breakdown")
        for _, r in result.iterrows():
            label = f"{r['Type']} {int(r['L'])}/{int(r['H'])}  |  NET Rs {r['NET Profit']:.2f}"
            with st.expander(label):
                _render_breakdown(r)
        export = result.drop(columns=["_legs", "_costs", "_edge_pct"])
        st.download_button("Export results to CSV", export.to_csv(index=False).encode(),
                           file_name=f"box_spreads_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    import streamlit as st  # noqa: F811
    main()
