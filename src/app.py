"""Streamlit app: pick a symbol, optionally upload a chart screenshot, see the live signal.

The screenshot is shown for reference only; the model always uses real candles from Binance.

    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # `streamlit run` does not add the project root

import streamlit as st  # noqa: E402

from src.chart_images import render_window  # noqa: E402
from src.config import load_config  # noqa: E402
from src.live import format_signal, get_signal  # noqa: E402

CHART_PX = 560


def chart_figure(hourly, cfg):
    """Last N candles in the training image style, larger for display."""
    ic = dict(cfg.images)
    return render_window(hourly.tail(ic["n_candles"]), None, ic, size_px=CHART_PX)


def show_signal(sig) -> None:
    color = {"LONG": "green", "SHORT": "red"}.get(sig.decision, "gray")
    st.markdown(f"### :{color}[{sig.decision}]")
    st.caption(f"Bar {sig.bar_time:%Y-%m-%d %H:%M} UTC, closed {sig.decision_time:%H:%M} UTC, model {sig.model}")
    cols = st.columns(4)
    if sig.decision == "NO TRADE":
        cols[0].metric("Last close", f"{sig.entry:,.2f}")
    else:
        cols[0].metric("Entry (approx.)", f"{sig.entry:,.2f}")
        cols[1].metric("Take-profit", f"{sig.tp:,.2f}")
        cols[2].metric("Stop-loss", f"{sig.sl:,.2f}")
        cols[3].metric("Confidence", f"{sig.confidence:.1%}")
    st.write(f"P(long WIN) {sig.p_long_win:.1%} | P(short WIN) {sig.p_short_win:.1%} | ATR {sig.atr:,.2f}")
    st.code(format_signal(sig), language=None)


def main() -> None:
    cfg = load_config()
    st.set_page_config(page_title="Crypto trade signals", layout="centered")
    st.title("Crypto trade signals")
    st.caption("Research project. Not financial advice. Signals use the latest closed 1h candle.")

    symbol = st.selectbox("Symbol", list(cfg.symbols))
    kind = st.selectbox("Model", ["xgb", "logreg"], index=0 if cfg.live.model_kind == "xgb" else 1)
    shot = st.file_uploader("Chart screenshot (optional, shown for reference only)", type=["png", "jpg", "jpeg"])
    if shot is not None:
        st.image(shot, caption="Your screenshot: not used by the model", use_container_width=True)

    if st.button("Get signal", type="primary"):
        try:
            with st.spinner("Fetching candles from Binance..."):
                sig, hourly = get_signal(cfg, symbol, kind)
        except FileNotFoundError:
            st.error("No final model found. Run `python -m src.evaluate --holdout` first.")
            return
        except Exception as e:                       # network or data problems are shown, not raised
            st.error(f"Could not build a signal: {e}")
            return
        show_signal(sig)
        st.subheader(f"Last {cfg.images.n_candles} hourly candles")
        st.pyplot(chart_figure(hourly, cfg))


if __name__ == "__main__":
    main()
