import json

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils import (
    build_feature_row, to_model_frame,
    predict_with_interval, explain_prediction, deal_score
)

st.set_page_config(
    page_title="SpecWorth | Laptop Valuation",
    page_icon="💻",
    layout="wide",
)

def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

@st.cache_resource
def load_artifacts():
    with open("models/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "lookup_df": joblib.load("models/lookup_df.pkl"),
        "dropdowns": joblib.load("models/dropdowns.pkl"),
        "model": joblib.load("models/price_model.pkl"),
        "conformal_quantile": joblib.load("models/conformal_quantile.pkl"),
        "model_columns": joblib.load("models/model_input_columns.pkl"),
        "reference": joblib.load("models/feature_reference.pkl"),
        "metrics": metrics,
    }


art = load_artifacts()
m = art["metrics"]
lookup_df = art["lookup_df"]
dropdowns = art["dropdowns"]

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="badge">AI-POWERED LAPTOP VALUATION</div>
    <h1>💻 Spec<span>Worth</span></h1>
    <h3>Know what a laptop is really worth.</h3>
    <p>
        Estimate fair market value from hardware specifications,
        understand what drives the price, and check whether a listing is a good deal.
    </p>
    <div class="pills">
        <span>⚡ XGBoost</span>
        <span>🧠 Explainable ML</span>
        <span>🎯 Conformal Uncertainty</span>
    </div>
</div>
""", unsafe_allow_html=True)

m = art["metrics"]
a, b, c, d = st.columns(4)
for col, icon, value, label in [
    (a, "\U0001F4C8", f"{m['test_r2'] * 100:.2f}%", "Test R\u00b2"),
    (b, "\U0001F4B0", f"\u20b9{m['test_mae'] / 1000:.2f}K", "Test MAE"),
    (c, "\U0001F3AF", f"{m['conformal_interval_coverage_pct']:.1f}%", "Interval coverage"),
    (d, "\U0001F4BB", f"{m['n_listings']}", "Indian listings"),
]:
    with col:
        st.markdown(
            f'<div class="stat"><b>{icon} &nbsp; {value}</b><small>{label}</small></div>',
            unsafe_allow_html=True
        )

st.markdown("<div class='gap'></div>", unsafe_allow_html=True)

# ---------------- RESULT ----------------
def render_result(row, art, listed_price=None):
    input_df = to_model_frame(row, art["model_columns"])
    point, lo, hi = predict_with_interval(
        art["model"], art["conformal_quantile"], input_df
    )

    st.markdown("""
    <div class="result-head">
        <div><small>VALUATION RESULT</small><h3>Your laptop's estimated worth</h3></div>
        <span>🎯 80% CALIBRATED RANGE</span>
    </div>
    """, unsafe_allow_html=True)

    if listed_price is not None:
        score = deal_score(point, listed_price)
        tone = score["tone"]
        icons = {"good": "✓", "neutral": "≈", "warn": "!", "bad": "×"}
        r1, r2, r3 = st.columns([1.2, 1, 1])

        with r1:
            st.markdown(
                f'<div class="price primary"><small>ESTIMATED FAIR PRICE</small>'
                f'<strong>₹ {point:,.0f}</strong><p>Model-estimated market value</p></div>',
                unsafe_allow_html=True)
        with r2:
            st.markdown(
                f'<div class="price"><small>LISTED PRICE</small>'
                f'<strong>₹ {listed_price:,.0f}</strong><p>{score["pct_diff"]:+.1f}% vs estimate</p></div>',
                unsafe_allow_html=True)
        with r3:
            st.markdown(
                f'<div class="price"><small>CALIBRATED RANGE</small>'
                f'<strong>₹ {lo:,.0f} – ₹ {hi:,.0f}</strong><p>80% prediction interval</p></div>',
                unsafe_allow_html=True)

        st.markdown(
            f'<div class="deal {tone}"><b>{icons[tone]}</b>'
            f'<div><small>DEAL ASSESSMENT</small><strong>{score["label"]}</strong></div></div>',
            unsafe_allow_html=True)
    else:
        r1, r2 = st.columns([1.2, 1])
        with r1:
            st.markdown(
                f'<div class="price primary"><small>ESTIMATED FAIR PRICE</small>'
                f'<strong>₹ {point:,.0f}</strong><p>Model-estimated market value</p></div>',
                unsafe_allow_html=True)
        with r2:
            st.markdown(
                f'<div class="price"><small>CALIBRATED RANGE</small>'
                f'<strong>₹ {lo:,.0f} – ₹ {hi:,.0f}</strong><p>80% prediction interval</p></div>',
                unsafe_allow_html=True)

    with st.expander("🔎 Why did the model predict this price?", expanded=True):
        contributions = explain_prediction(
            art["model"], input_df, art["reference"]
        )
        if not contributions:
            st.info("This configuration is close to a typical laptop.")
        else:
            for item in contributions:
                positive = item["impact_inr"] >= 0
                arrow = "↑" if positive else "↓"
                cls = "positive" if positive else "negative"
                sign = "+" if positive else "−"
                st.markdown(
                    f'<div class="factor"><b class="{cls}">{arrow}</b>'
                    f'<span>{item["feature"]}</span>'
                    f'<strong class="{cls}">{sign}₹{abs(item["impact_inr"]):,}</strong></div>',
                    unsafe_allow_html=True
                )
        st.caption(
            "Each feature is replaced by a typical training-set value and the change "
            "in prediction is measured."
        )

tab_quick, tab_custom, tab_deal = st.tabs(
    ["🗂️  QUICK PICK", "🛠️  CUSTOM BUILD", "🔍  DEAL CHECKER"]
)

# ---------------- QUICK PICK ----------------
with tab_quick:
    st.markdown("""
    <div class="section">
        <div class="section-icon">🗂️</div>
        <div><h2>Explore real laptop listings</h2>
        <p>Filter the catalog and see what the model thinks each machine is worth.</p></div>
    </div>
    """, unsafe_allow_html=True)

    filtered_df = lookup_df.copy()
    c1, c2, c3 = st.columns(3)

    with c1:
        brand = st.selectbox("Brand", sorted(filtered_df["brand"].unique()), key="qp_brand")
        filtered_df = filtered_df[filtered_df["brand"] == brand]
        processor = st.selectbox("Processor", sorted(filtered_df["processor"].unique()), key="qp_proc")
        filtered_df = filtered_df[filtered_df["processor"] == processor]
        cpu = st.selectbox("CPU Configuration", sorted(filtered_df["CPU"].unique()), key="qp_cpu")
        filtered_df = filtered_df[filtered_df["CPU"] == cpu]

    with c2:
        ram = st.selectbox("RAM (GB)", sorted(filtered_df["Ram"].unique()), key="qp_ram")
        filtered_df = filtered_df[filtered_df["Ram"] == ram]
        ram_type = st.selectbox("RAM Type", sorted(filtered_df["Ram_type"].unique()), key="qp_ramtype")
        filtered_df = filtered_df[filtered_df["Ram_type"] == ram_type]
        rom = st.selectbox("Storage (GB)", sorted(filtered_df["ROM"].unique()), key="qp_rom")
        filtered_df = filtered_df[filtered_df["ROM"] == rom]

    with c3:
        rom_type = st.selectbox("Storage Type", sorted(filtered_df["ROM_type"].unique()), key="qp_romtype")
        filtered_df = filtered_df[filtered_df["ROM_type"] == rom_type]
        gpu = st.selectbox("GPU", sorted(filtered_df["GPU"].unique()), key="qp_gpu")
        filtered_df = filtered_df[filtered_df["GPU"] == gpu]
        screen_size = st.selectbox("Screen Size", sorted(filtered_df["display_size"].unique()), key="qp_screen")
        filtered_df = filtered_df[filtered_df["display_size"] == screen_size]

    if filtered_df.empty:
        st.warning("No exact catalog match — try Custom Build instead.")
    else:
        resolutions = sorted(
            (filtered_df["resolution_width"].astype(int).astype(str) + "x" +
             filtered_df["resolution_height"].astype(int).astype(str)).unique()
        )
        resolution = st.selectbox("Resolution", resolutions, key="qp_res")
        width, height = resolution.split("x")
        filtered_df = filtered_df[
            (filtered_df["resolution_width"] == int(width)) &
            (filtered_df["resolution_height"] == int(height))
        ]
        row_data = filtered_df.iloc[0]

        st.markdown(
            f'<div class="listing"><small>SELECTED LISTING</small>'
            f'<b>{row_data["name"]}</b><span>Catalog price: ₹{row_data["price"]:,.0f}</span></div>',
            unsafe_allow_html=True)

        if st.button("💰  VALUE THIS LAPTOP", use_container_width=True, key="qp_predict"):
            row = build_feature_row(
                brand=row_data["brand"], processor_text=row_data["processor"],
                cpu_text=row_data["CPU"], ram_gb=int(row_data["Ram"]),
                ram_type=row_data["Ram_type"], rom_gb=int(row_data["ROM"]),
                rom_type=row_data["ROM_type"], gpu_text=row_data["GPU"],
                warranty=int(row_data["warranty"]),
                screen_size=float(row_data["display_size"]),
                resolution=resolution, os_text=row_data["OS"])
            render_result(row, art, listed_price=float(row_data["price"]))

# ---------------- CUSTOM BUILD ----------------
with tab_custom:
    st.markdown("""
    <div class="section">
        <div class="section-icon">🛠️</div>
        <div><h2>Build your own configuration</h2>
        <p>Enter a laptop specification — even if it is not in the catalog.</p></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        brand_c = st.selectbox("Brand", dropdowns["brand"], key="cb_brand")
        processor_c = st.text_input("Processor", "13th Gen Intel Core i5 1340P", key="cb_proc")
        cpu_c = st.text_input("CPU configuration", "12 Cores (4P + 8E), 16 Threads", key="cb_cpu")
        os_c = st.selectbox("Operating System", dropdowns["os"], key="cb_os")
    with c2:
        ram_c = st.select_slider("RAM (GB)", [4, 8, 16, 32, 64], value=16, key="cb_ram")
        ram_type_c = st.selectbox("RAM Type", dropdowns["ram_type"], key="cb_ramtype")
        rom_c = st.select_slider("Storage (GB)", [128, 256, 512, 1024, 2048], value=512, key="cb_rom")
        rom_type_c = st.selectbox("Storage Type", ["SSD", "HDD", "eMMC"], key="cb_romtype")
    with c3:
        gpu_c = st.text_input("GPU", "4GB NVIDIA GeForce RTX 3050", key="cb_gpu")
        screen_c = st.select_slider("Screen Size (inches)", [13.3, 14.0, 15.6, 16.0, 17.3], value=15.6, key="cb_screen")
        resolution_c = st.selectbox("Resolution", ["1920x1080", "1920x1200", "2560x1440", "3840x2160", "2160x1350"], key="cb_res")
        warranty_c = st.select_slider("Warranty (years)", [0, 1, 2, 3], value=1, key="cb_warranty")

    if st.button("💰  ESTIMATE FAIR VALUE", use_container_width=True, key="cb_predict"):
        row = build_feature_row(
            brand=brand_c, processor_text=processor_c, cpu_text=cpu_c,
            ram_gb=ram_c, ram_type=ram_type_c, rom_gb=rom_c,
            rom_type=rom_type_c, gpu_text=gpu_c, warranty=warranty_c,
            screen_size=screen_c, resolution=resolution_c, os_text=os_c)
        render_result(row, art)

# ---------------- DEAL CHECKER ----------------
with tab_deal:
    st.markdown("""
    <div class="section">
        <div class="section-icon">🔍</div>
        <div><h2>Check whether the listing is worth it</h2>
        <p>Compare a quoted price against SpecWorth's estimated fair value.</p></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        brand_d = st.selectbox("Brand", dropdowns["brand"], key="dc_brand")
        processor_d = st.text_input("Processor", "12th Gen Intel Core i5 1235U", key="dc_proc")
        cpu_d = st.text_input("CPU configuration", "10 Cores (2P + 8E), 12 Threads", key="dc_cpu")
        os_d = st.selectbox("Operating System", dropdowns["os"], key="dc_os")
    with c2:
        ram_d = st.select_slider("RAM (GB)", [4, 8, 16, 32, 64], value=8, key="dc_ram")
        ram_type_d = st.selectbox("RAM Type", dropdowns["ram_type"], key="dc_ramtype")
        rom_d = st.select_slider("Storage (GB)", [128, 256, 512, 1024, 2048], value=512, key="dc_rom")
        rom_type_d = st.selectbox("Storage Type", ["SSD", "HDD", "eMMC"], key="dc_romtype")
    with c3:
        gpu_d = st.text_input("GPU", "Integrated Intel Iris Xe Graphics", key="dc_gpu")
        screen_d = st.select_slider("Screen Size (inches)", [13.3, 14.0, 15.6, 16.0, 17.3], value=15.6, key="dc_screen")
        resolution_d = st.selectbox("Resolution", ["1920x1080", "1920x1200", "2560x1440", "3840x2160", "2160x1350"], key="dc_res")
        warranty_d = st.select_slider("Warranty (years)", [0, 1, 2, 3], value=1, key="dc_warranty")

    listed_price = st.number_input("Listed / quoted price (₹)", min_value=1000, value=50000, step=500, key="dc_price")

    if st.button("🔍  CHECK THIS DEAL", use_container_width=True, key="dc_predict"):
        row = build_feature_row(
            brand=brand_d, processor_text=processor_d, cpu_text=cpu_d,
            ram_gb=ram_d, ram_type=ram_type_d, rom_gb=rom_d,
            rom_type=rom_type_d, gpu_text=gpu_d, warranty=warranty_d,
            screen_size=screen_d, resolution=resolution_d, os_text=os_d)
        render_result(row, art, listed_price=float(listed_price))

# ---------------- FOOTER ----------------
with st.expander("ℹ️  About SpecWorth"):
    _target = int((1 - m["conformal_alpha"]) * 100)
    st.markdown(
        f"**SpecWorth** estimates laptop fair value from hardware specifications, "
        f"trained on **{m['n_listings']} Indian-market listings**. XGBoost on "
        f"log-transformed prices produces the point estimate; the range comes from "
        f"**split conformal calibration** on a held-out set of {m['n_cal']} listings, "
        f"targeting {_target}% coverage. Observed coverage on the untouched "
        f"{m['n_test']}-listing test set was "
        f"**{m['conformal_interval_coverage_pct']:.1f}%**.\n\n"
        f"The point estimate and the interval come from the *same* model, which is what "
        f"makes that coverage figure meaningful. With only {m['n_cal']} calibration "
        f"points the conformal quantile is itself a noisy estimate, so a gap of a few "
        f"points from the {_target}% target is expected rather than evidence of "
        f"conservative intervals."
    )

st.markdown(
    '<div class="footer">SPECWORTH &nbsp;•&nbsp; EXPLAINABLE LAPTOP VALUATION</div>',
    unsafe_allow_html=True
)
