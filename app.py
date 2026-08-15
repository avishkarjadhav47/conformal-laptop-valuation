import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


from utils import (
    build_feature_row, to_model_frame,
    predict_with_interval, explain_prediction, deal_score,
)

st.set_page_config(page_title="Explainable Laptop Valuation with Calibrated Uncertainty", page_icon="💻", layout="wide")
load_css()

#=======================================================Load artifacts========================================================

@st.cache_resource
def load_artifacts():
    return {
        "lookup_df": joblib.load("models/lookup_df.pkl"),
        "dropdowns": joblib.load("models/dropdowns.pkl"),
        "price_model": joblib.load("models/price_model.pkl"),
        "conformal_model": joblib.load("models/price_model_conformal.pkl"),
        "conformal_quantile": joblib.load("models/conformal_quantile.pkl"),
        "model_columns": joblib.load("models/model_input_columns.pkl"),
        "reference": joblib.load("models/feature_reference.pkl"),
    }

art = load_artifacts()
lookup_df = art["lookup_df"]
dropdowns = art["dropdowns"]

st.title("💻 Explainable Laptop Valuation with Calibrated Uncertainty")
st.write(
    "Estimate a fair price from hardware specs — browse real catalog entries, "
    "build a custom configuration from scratch, or check whether a listing is a good deal."
)

tab_quick, tab_custom, tab_deal = st.tabs(["🗂️ Quick Pick", "🛠️ Custom Build", "🔍 Deal Checker"])


def render_result(row, art, listed_price=None):
    input_df = to_model_frame(row, art["model_columns"])
    point, lo, hi = predict_with_interval(
        art["price_model"], art["conformal_model"], art["conformal_quantile"], input_df
    )

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Estimated Fair Price", f"₹ {point:,.0f}")
        st.caption(f"Likely range: ₹ {lo:,.0f} – ₹ {hi:,.0f} (80% confidence)")

    with c2:
        if listed_price is not None:
            score = deal_score(point, listed_price)
            tone_color = {"good": "🟢", "neutral": "🟡", "warn": "🟠", "bad": "🔴"}[score["tone"]]
            st.metric("Listed Price", f"₹ {listed_price:,.0f}", f"{score['pct_diff']:+.1f}% vs estimate")
            st.write(f"{tone_color} **{score['label']}**")

    with st.expander("Why this price? (top factors)"):
        contributions = explain_prediction(art["price_model"], input_df, art["reference"])
        if not contributions:
            st.write("This configuration is close to a 'typical' laptop across the board.")
        else:
            for c in contributions:
                sign = "+" if c["impact_inr"] >= 0 else "−"
                st.write(f"- **{c['feature']}**: {sign}₹{abs(c['impact_inr']):,} vs. a typical laptop")
        st.caption(
            "Estimated by swapping each spec for a 'typical' value and measuring how much "
            "the price moves — a lightweight, from-scratch alternative to SHAP."
        )


# ======================================TAB 1 — Quick Pick (browse real catalog entries)==========================================


with tab_quick:
    st.subheader("Pick from real listings")
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
        ram_options = sorted(filtered_df["Ram"].unique())
        ram = st.selectbox("RAM (GB)", ram_options, key="qp_ram")
        filtered_df = filtered_df[filtered_df["Ram"] == ram]

        ram_type = st.selectbox("RAM Type", sorted(filtered_df["Ram_type"].unique()), key="qp_ramtype")
        filtered_df = filtered_df[filtered_df["Ram_type"] == ram_type]

        rom_options = sorted(filtered_df["ROM"].unique())
        rom = st.selectbox("Storage (GB)", rom_options, key="qp_rom")
        filtered_df = filtered_df[filtered_df["ROM"] == rom]

    with c3:
        rom_type = st.selectbox("Storage Type", sorted(filtered_df["ROM_type"].unique()), key="qp_romtype")
        filtered_df = filtered_df[filtered_df["ROM_type"] == rom_type]

        gpu = st.selectbox("GPU", sorted(filtered_df["GPU"].unique()), key="qp_gpu")
        filtered_df = filtered_df[filtered_df["GPU"] == gpu]

        screen_size = st.selectbox("Screen Size", sorted(filtered_df["display_size"].unique()), key="qp_screen")
        filtered_df = filtered_df[filtered_df["display_size"] == screen_size]

    if filtered_df.empty:
        st.warning("No exact catalog match for this combination — try Custom Build instead, which handles any configuration.")
    else:
        resolution_options = sorted(
            (filtered_df["resolution_width"].astype(int).astype(str) + "x" + filtered_df["resolution_height"].astype(int).astype(str)).unique()
        )
        resolution = st.selectbox("Resolution", resolution_options, key="qp_res")
        width, height = resolution.split("x")
        filtered_df = filtered_df[
            (filtered_df["resolution_width"] == int(width)) & (filtered_df["resolution_height"] == int(height))
        ]

        row_data = filtered_df.iloc[0]
        st.caption(f"Matched catalog listing: **{row_data['name']}** — actual price ₹{row_data['price']:,.0f}")

        if st.button("💰 Predict Price", use_container_width=True, key="qp_predict"):
            row = build_feature_row(
                brand=row_data["brand"], processor_text=row_data["processor"], cpu_text=row_data["CPU"],
                ram_gb=int(row_data["Ram"]), ram_type=row_data["Ram_type"], rom_gb=int(row_data["ROM"]),
                rom_type=row_data["ROM_type"], gpu_text=row_data["GPU"], warranty=int(row_data["warranty"]),
                screen_size=float(row_data["display_size"]), resolution=resolution, os_text=row_data["OS"],
            )
            render_result(row, art, listed_price=float(row_data["price"]))


# ===============================TAB 2 — Custom Build (any configuration, not limited to the dataset)==========================================

with tab_custom:
    st.subheader("Describe any configuration")
    st.caption("Not limited to existing catalog entries — type specs for a laptop that doesn't exist yet.")

    c1, c2, c3 = st.columns(3)
    with c1:
        brand_c = st.selectbox("Brand", dropdowns["brand"], key="cb_brand")
        processor_c = st.text_input("Processor", "13th Gen Intel Core i5 1340P", key="cb_proc")
        cpu_c = st.text_input("CPU configuration", "12 Cores (4P + 8E), 16 Threads", key="cb_cpu")
        os_c = st.selectbox("Operating System", dropdowns["os"], key="cb_os")

    with c2:
        ram_c = st.select_slider("RAM (GB)", options=[4, 8, 16, 32, 64], value=16, key="cb_ram")
        ram_type_c = st.selectbox("RAM Type", dropdowns["ram_type"], key="cb_ramtype")
        rom_c = st.select_slider("Storage (GB)", options=[128, 256, 512, 1024, 2048], value=512, key="cb_rom")
        rom_type_c = st.selectbox("Storage Type", ["SSD", "HDD", "eMMC"], key="cb_romtype")

    with c3:
        gpu_c = st.text_input("GPU", "4GB NVIDIA GeForce RTX 3050", key="cb_gpu")
        screen_c = st.select_slider("Screen Size (inches)", options=[13.3, 14.0, 15.6, 16.0, 17.3], value=15.6, key="cb_screen")
        resolution_c = st.selectbox("Resolution", ["1920x1080", "1920x1200", "2560x1440", "3840x2160", "2160x1350"], key="cb_res")
        warranty_c = st.select_slider("Warranty (years)", options=[0, 1, 2, 3], value=1, key="cb_warranty")

    if st.button("💰 Predict Price", use_container_width=True, key="cb_predict"):
        row = build_feature_row(
            brand=brand_c, processor_text=processor_c, cpu_text=cpu_c, ram_gb=ram_c, ram_type=ram_type_c,
            rom_gb=rom_c, rom_type=rom_type_c, gpu_text=gpu_c, warranty=warranty_c,
            screen_size=screen_c, resolution=resolution_c, os_text=os_c,
        )
        render_result(row, art)


# ==================================TAB 3 — Deal Checker (compare a real listing price to the model estimate)==================================

with tab_deal:
    st.subheader("Is this listing a good deal?")
    st.caption("Enter the specs and the price you've been quoted — see how it compares to the model's fair-price estimate.")

    c1, c2, c3 = st.columns(3)
    with c1:
        brand_d = st.selectbox("Brand", dropdowns["brand"], key="dc_brand")
        processor_d = st.text_input("Processor", "12th Gen Intel Core i5 1235U", key="dc_proc")
        cpu_d = st.text_input("CPU configuration", "10 Cores (2P + 8E), 12 Threads", key="dc_cpu")
        os_d = st.selectbox("Operating System", dropdowns["os"], key="dc_os")

    with c2:
        ram_d = st.select_slider("RAM (GB)", options=[4, 8, 16, 32, 64], value=8, key="dc_ram")
        ram_type_d = st.selectbox("RAM Type", dropdowns["ram_type"], key="dc_ramtype")
        rom_d = st.select_slider("Storage (GB)", options=[128, 256, 512, 1024, 2048], value=512, key="dc_rom")
        rom_type_d = st.selectbox("Storage Type", ["SSD", "HDD", "eMMC"], key="dc_romtype")

    with c3:
        gpu_d = st.text_input("GPU", "Integrated Intel Iris Xe Graphics", key="dc_gpu")
        screen_d = st.select_slider("Screen Size (inches)", options=[13.3, 14.0, 15.6, 16.0, 17.3], value=15.6, key="dc_screen")
        resolution_d = st.selectbox("Resolution", ["1920x1080", "1920x1200", "2560x1440", "3840x2160", "2160x1350"], key="dc_res")
        warranty_d = st.select_slider("Warranty (years)", options=[0, 1, 2, 3], value=1, key="dc_warranty")

    listed_price = st.number_input("Listed / quoted price (₹)", min_value=1000, value=50000, step=500, key="dc_price")

    if st.button("🔍 Check This Deal", use_container_width=True, key="dc_predict"):
        row = build_feature_row(
            brand=brand_d, processor_text=processor_d, cpu_text=cpu_d, ram_gb=ram_d, ram_type=ram_type_d,
            rom_gb=rom_d, rom_type=rom_type_d, gpu_text=gpu_d, warranty=warranty_d,
            screen_size=screen_d, resolution=resolution_d, os_text=os_d,
        )
        render_result(row, art, listed_price=float(listed_price))

st.divider()
with st.expander("ℹ️ About this model"):
    st.write(
        "Trained on ~893 Indian-market laptop listings (Kaggle). "
        "The point estimate comes from an XGBoost regression model trained "
        "on log-transformed prices. The prediction range is generated using "
        "split conformal calibration, targeting 80% coverage rather than "
        "using a fixed ± margin. On the held-out test set, the calibrated "
        "interval achieved 82.1% empirical coverage."
    )
