"""Streamlit interface for the PVT Studio correlation engine."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pvt_engine import (
    ABS_PRESSURE_UNITS,
    BG_UNITS,
    BO_FUNCTIONS,
    CGR_UNITS,
    COMPRESSIBILITY_UNITS,
    DEAD_VISCOSITY_FUNCTIONS,
    DENSITY_UNITS,
    GOR_UNITS,
    OilOptions,
    GasOptions,
    PRESSURE_UNITS,
    PSEUDOCRITICAL_FUNCTIONS,
    RS_FUNCTIONS,
    SAT_VISCOSITY_FUNCTIONS,
    TEMPERATURE_UNITS,
    VISCOSITY_UNITS,
    Z_FACTOR_CORRELATIONS,
    api_to_oil_input,
    api_to_sg_oil,
    bg_from_ft3_scf,
    calculate_black_oil_table,
    calculate_dead_oil_table,
    calculate_gas_table,
    cgr_from_stb_mmscf,
    cgr_to_stb_mmscf,
    compressibility_from_per_psi,
    compressibility_to_per_psi,
    correlation_range_warnings,
    density_from_lbft3,
    format_output_table,
    gas_input_to_sg,
    gas_sg_to_input,
    gor_from_scfstb,
    gor_to_scfstb,
    oil_input_to_api,
    pressure_from_psia,
    pressure_to_psia,
    temperature_from_f,
    temperature_to_f,
    viscosity_from_cp,
    viscosity_to_cp,
)


st.set_page_config(
    page_title="PVT Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "navy": "#123B67",
    "blue": "#137CBD",
    "cyan": "#00A6A6",
    "green": "#2E8B57",
    "orange": "#F59E0B",
    "red": "#DC3C3C",
    "purple": "#7C3AED",
    "slate": "#64748B",
}


st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg, #f8fbff 0%, #ffffff 32%);}
      [data-testid="stSidebar"] {background: #f3f7fb; border-right: 1px solid #dbe7f3;}
      .hero {
        padding: 1.25rem 1.45rem; border-radius: 18px;
        background: linear-gradient(115deg, #0b2d4d 0%, #125b89 55%, #168ca2 100%);
        color: white; box-shadow: 0 10px 28px rgba(16, 58, 92, 0.16);
        margin-bottom: 1rem;
      }
      .hero h1 {margin: 0; font-size: 2.05rem; letter-spacing: -0.03em;}
      .hero p {margin: .38rem 0 0; opacity: .9; font-size: 1rem;}
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #dbe7f3; padding: .8rem 1rem;
        border-radius: 14px; box-shadow: 0 4px 16px rgba(15, 47, 74, .06);
      }
      .small-note {font-size: .86rem; color: #52677a;}
      .correlation-card {
        background: #f7fafc; border-left: 4px solid #137CBD; border-radius: 8px;
        padding: .7rem .9rem; margin-bottom: .55rem;
      }
      .stDownloadButton button {width: 100%; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _convert_between_units(value: float, old_unit: str, new_unit: str, units: tuple[str, ...]) -> float:
    """Preserve the physical quantity when a displayed unit is changed."""
    if old_unit == new_unit:
        return float(value)
    if units == PRESSURE_UNITS:
        return float(pressure_from_psia(pressure_to_psia(value, old_unit), new_unit))
    if units == TEMPERATURE_UNITS:
        return float(temperature_from_f(temperature_to_f(value, old_unit), new_unit))
    if units == GOR_UNITS:
        return float(gor_from_scfstb(gor_to_scfstb(value, old_unit), new_unit))
    if units == COMPRESSIBILITY_UNITS:
        base_value = compressibility_to_per_psi(value, old_unit)
        return float(compressibility_from_per_psi(base_value, new_unit))
    if units == VISCOSITY_UNITS:
        return float(viscosity_from_cp(viscosity_to_cp(value, old_unit), new_unit))
    if units == CGR_UNITS:
        return float(cgr_from_stb_mmscf(cgr_to_stb_mmscf(value, old_unit), new_unit))
    raise ValueError(f"Automatic conversion is not configured for {old_unit} → {new_unit}")


def _on_numeric_unit_change(
    value_key: str,
    unit_key: str,
    previous_unit_key: str,
    units: tuple[str, ...],
) -> None:
    old_unit = st.session_state[previous_unit_key]
    new_unit = st.session_state[unit_key]
    st.session_state[value_key] = _convert_between_units(
        float(st.session_state[value_key]), old_unit, new_unit, units
    )
    st.session_state[previous_unit_key] = new_unit


def number_with_unit(
    label: str,
    value: float,
    units: tuple[str, ...],
    default_unit: str,
    key: str,
    min_value: float | None = None,
    step: float | None = None,
    help_text: str | None = None,
) -> tuple[float, str]:
    value_key = f"{key}_value"
    unit_key = f"{key}_unit"
    previous_unit_key = f"{key}_previous_unit"
    if value_key not in st.session_state:
        st.session_state[value_key] = float(value)
    if unit_key not in st.session_state or st.session_state[unit_key] not in units:
        st.session_state[unit_key] = default_unit
    if previous_unit_key not in st.session_state:
        st.session_state[previous_unit_key] = st.session_state[unit_key]

    col_value, col_unit = st.columns([2.0, 1.15], vertical_alignment="bottom")
    kwargs: dict[str, object] = {
        "key": value_key,
        "help": help_text,
        "format": "%.8g",
    }
    if min_value is not None:
        kwargs["min_value"] = _convert_between_units(
            float(min_value), default_unit, st.session_state[unit_key], units
        )
    if step is not None:
        converted_zero = _convert_between_units(0.0, default_unit, st.session_state[unit_key], units)
        converted_step = _convert_between_units(float(step), default_unit, st.session_state[unit_key], units)
        kwargs["step"] = abs(converted_step - converted_zero)
    with col_value:
        entered = st.number_input(label, **kwargs)
    with col_unit:
        unit = st.selectbox(
            f"{label} unit",
            units,
            key=unit_key,
            label_visibility="collapsed",
            on_change=_on_numeric_unit_change,
            args=(value_key, unit_key, previous_unit_key, units),
        )
    return float(entered), unit


def _on_oil_basis_change(value_key: str, basis_key: str, previous_basis_key: str) -> None:
    old_basis = st.session_state[previous_basis_key]
    new_basis = st.session_state[basis_key]
    api_value = oil_input_to_api(float(st.session_state[value_key]), old_basis)
    st.session_state[value_key] = api_to_oil_input(api_value, new_basis)
    st.session_state[previous_basis_key] = new_basis


def oil_gravity_input(default_api: float = 35.0) -> tuple[float, str, float]:
    kinds = (
        "API gravity (°API)",
        "Oil specific gravity",
        "Stock-tank density (kg/m³)",
        "Stock-tank density (lb/ft³)",
    )
    value_key = "oil_gravity_value"
    basis_key = "oil_gravity_basis"
    previous_basis_key = "oil_gravity_previous_basis"
    if value_key not in st.session_state:
        st.session_state[value_key] = float(default_api)
    if basis_key not in st.session_state or st.session_state[basis_key] not in kinds:
        st.session_state[basis_key] = kinds[0]
    if previous_basis_key not in st.session_state:
        st.session_state[previous_basis_key] = st.session_state[basis_key]

    kind = st.selectbox(
        "Oil gravity basis",
        kinds,
        key=basis_key,
        on_change=_on_oil_basis_change,
        args=(value_key, basis_key, previous_basis_key),
    )
    value = st.number_input(
        "Oil gravity / density",
        min_value=0.01,
        key=value_key,
        format="%.8g",
    )
    return float(value), kind, oil_input_to_api(float(value), kind)


def _on_gas_basis_change(value_key: str, basis_key: str, previous_basis_key: str) -> None:
    old_basis = st.session_state[previous_basis_key]
    new_basis = st.session_state[basis_key]
    gas_sg_value = gas_input_to_sg(float(st.session_state[value_key]), old_basis)
    st.session_state[value_key] = gas_sg_to_input(gas_sg_value, new_basis)
    st.session_state[previous_basis_key] = new_basis


def gas_gravity_input(default_sg: float = 0.70) -> tuple[float, str, float]:
    kinds = ("Gas specific gravity (air=1)", "Molecular weight (g/mol)")
    value_key = "gas_gravity_value"
    basis_key = "gas_gravity_basis"
    previous_basis_key = "gas_gravity_previous_basis"
    if value_key not in st.session_state:
        st.session_state[value_key] = float(default_sg)
    if basis_key not in st.session_state or st.session_state[basis_key] not in kinds:
        st.session_state[basis_key] = kinds[0]
    if previous_basis_key not in st.session_state:
        st.session_state[previous_basis_key] = st.session_state[basis_key]

    kind = st.selectbox(
        "Gas gravity basis",
        kinds,
        key=basis_key,
        on_change=_on_gas_basis_change,
        args=(value_key, basis_key, previous_basis_key),
    )
    value = st.number_input(
        "Gas gravity / molecular weight",
        min_value=0.01,
        key=value_key,
        format="%.8g",
    )
    return float(value), kind, gas_input_to_sg(float(value), kind)


def chart_layout(fig: go.Figure, title: str, y_title: str, x_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=390,
        margin=dict(l=42, r=28, t=65, b=44),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#183247"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
    )
    fig.update_xaxes(title=x_title, gridcolor="#e8eef4", zeroline=False, showline=True, linecolor="#b8c7d5")
    fig.update_yaxes(title=y_title, gridcolor="#e8eef4", zeroline=False, showline=True, linecolor="#b8c7d5")
    return fig


def add_vertical_marker(fig: go.Figure, x: float | None, label: str, color: str = COLORS["red"]) -> None:
    if x is None:
        return
    fig.add_vline(x=x, line_width=1.6, line_dash="dash", line_color=color)
    fig.add_annotation(
        x=x,
        y=1,
        yref="paper",
        text=label,
        showarrow=False,
        yshift=12,
        font=dict(color=color, size=11),
        bgcolor="rgba(255,255,255,.85)",
    )


def line_chart(
    x: np.ndarray,
    series: list[tuple[str, np.ndarray, str, str]],
    title: str,
    x_title: str,
    y_title: str,
    marker_x: float | None = None,
    marker_label: str = "",
) -> go.Figure:
    fig = go.Figure()
    for name, values, color, dash in series:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                name=name,
                mode="lines",
                line=dict(color=color, width=2.6, dash=dash),
            )
        )
    chart_layout(fig, title, y_title, x_title)
    add_vertical_marker(fig, marker_x, marker_label)
    return fig


def dual_axis_chart(
    x: np.ndarray,
    left: tuple[str, np.ndarray, str],
    right: tuple[str, np.ndarray, str],
    title: str,
    x_title: str,
    left_title: str,
    right_title: str,
    marker_x: float | None = None,
    marker_label: str = "",
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=x, y=left[1], name=left[0], mode="lines", line=dict(color=left[2], width=2.6)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=right[1],
            name=right[0],
            mode="lines",
            line=dict(color=right[2], width=2.2, dash="dash"),
        ),
        secondary_y=True,
    )
    chart_layout(fig, title, left_title, x_title)
    fig.update_yaxes(title_text=left_title, secondary_y=False, color=left[2])
    fig.update_yaxes(title_text=right_title, secondary_y=True, color=right[2], showgrid=False)
    add_vertical_marker(fig, marker_x, marker_label)
    return fig


def add_export_metadata(export_df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    result = export_df.copy()
    # Repeated metadata keeps a plain CSV self-describing in any analysis tool.
    for key, value in reversed(list(metadata.items())):
        result.insert(0, key, value)
    return result


st.markdown(
    """
    <div class="hero">
      <h1>PVT Studio</h1>
      <p>Correlation-based fluid property generation • unit-aware inputs • interactive QA • CSV export</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Model setup")
    fluid = st.selectbox("Fluid system", ("Black Oil", "Dead Oil", "Dry Gas", "Wet Gas"))
    st.caption("All calculations are converted to oilfield units internally, then returned in your selected output units.")
    st.caption("Changing an input unit automatically converts its current numerical value.")

    st.subheader("Pressure grid")
    pressure_unit_default = "psia"
    p_min_value, p_min_unit = number_with_unit(
        "Minimum pressure", 100.0, PRESSURE_UNITS, pressure_unit_default, "p_min", step=25.0
    )
    p_max_value, p_max_unit = number_with_unit(
        "Maximum pressure", 6000.0, PRESSURE_UNITS, pressure_unit_default, "p_max", step=100.0
    )
    points = st.slider("Number of pressure points", 25, 500, 160, 5)

    st.subheader("Reservoir temperature")
    t_value, t_unit = number_with_unit(
        "Temperature", 180.0, TEMPERATURE_UNITS, "°F", "temperature", step=1.0
    )

    with st.expander("Output units", expanded=False):
        output_pressure_unit = st.selectbox("Pressure", ABS_PRESSURE_UNITS, index=0)
        output_gor_unit = st.selectbox("Solution GOR", GOR_UNITS, index=0)
        output_bg_unit = st.selectbox("Gas FVF", BG_UNITS, index=0)
        output_viscosity_unit = st.selectbox("Viscosity", VISCOSITY_UNITS, index=0)
        output_density_unit = st.selectbox("Density", DENSITY_UNITS, index=0)
        output_compressibility_unit = st.selectbox("Compressibility", COMPRESSIBILITY_UNITS, index=0)

    st.divider()
    run_pvt = st.button("▶ Run PVT", type="primary", width="stretch")
    st.caption("Change the inputs freely, then press Run PVT to calculate and refresh all results.")


try:
    p_min_psia = pressure_to_psia(p_min_value, p_min_unit)
    p_max_psia = pressure_to_psia(p_max_value, p_max_unit)
    t_f = temperature_to_f(t_value, t_unit)
    if p_min_psia <= 0.0 or p_max_psia <= 0.0:
        st.error("Pressure inputs must be positive on an absolute-pressure basis.")
        st.stop()
    if p_max_psia <= p_min_psia:
        st.error("Maximum pressure must be greater than minimum pressure after unit conversion.")
        st.stop()
    if t_f <= -459.0:
        st.error("Temperature must be above absolute zero.")
        st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


pressures = np.linspace(p_min_psia, p_max_psia, points)
api: float | None = None
gas_sg: float | None = None
dew_point_psia: float | None = None
cgr_stb_mmscf: float | None = None
correlation_metadata: dict[str, object] = {}
sour_enabled = False
co2_pct = 0.0
h2s_pct = 0.0


input_tab, correlation_tab = st.tabs(["Fluid inputs", "Correlations"])

with input_tab:
    if fluid in ("Black Oil", "Dead Oil"):
        oil_col, gas_col = st.columns(2)
        with oil_col:
            st.subheader("Oil definition")
            _, oil_gravity_kind, api = oil_gravity_input()
            st.caption(f"Converted internally to **{api:.2f} °API**.")

        if fluid == "Black Oil":
            with gas_col:
                st.subheader("Solution gas")
                _, gas_gravity_kind, gas_sg = gas_gravity_input()
                st.caption(f"Converted internally to **gas SG = {gas_sg:.4f}**.")
                with st.expander("Optional acid-gas correction"):
                    sour_enabled = st.toggle(
                        "Apply Wichert–Aziz correction",
                        value=False,
                        key="black_oil_sour_enabled",
                    )
                    if sour_enabled:
                        co2_pct = st.number_input("CO₂ (mol%)", 0.0, 70.0, 2.0, 0.1, key="black_oil_co2")
                        h2s_pct = st.number_input("H₂S (mol%)", 0.0, 70.0, 0.0, 0.1, key="black_oil_h2s")
                        if co2_pct + h2s_pct >= 100.0:
                            st.error("CO₂ + H₂S must be less than 100 mol%.")
                            st.stop()

            anchor_col, comp_col = st.columns(2)
            with anchor_col:
                st.subheader("Saturation anchor")
                anchor_mode = st.radio(
                    "Known laboratory/field input",
                    ("Rs at bubble point", "Bubble-point pressure"),
                    horizontal=True,
                )
                if anchor_mode == "Rs at bubble point":
                    rs_value, rs_unit = number_with_unit(
                        "Rs at Pb", 650.0, GOR_UNITS, "scf/STB", "rs_pb", min_value=0.0, step=10.0
                    )
                    rs_at_pb = gor_to_scfstb(rs_value, rs_unit)
                    pb_input = None
                else:
                    pb_value, pb_unit = number_with_unit(
                        "Bubble-point pressure", 2800.0, PRESSURE_UNITS, "psia", "pb", step=50.0
                    )
                    pb_input = pressure_to_psia(pb_value, pb_unit)
                    rs_at_pb = None
            with comp_col:
                st.subheader("Undersaturated oil")
                co_value, co_unit = number_with_unit(
                    "Oil compressibility",
                    1.2e-5,
                    COMPRESSIBILITY_UNITS,
                    "1/psi",
                    "co",
                    min_value=0.0,
                    step=1e-6,
                    help_text="Used above bubble point for Bo pressure behavior.",
                )
                co_per_psi = compressibility_to_per_psi(co_value, co_unit)
        else:
            with gas_col:
                st.subheader("Reference state")
                ref_p_value, ref_p_unit = number_with_unit(
                    "Reference pressure", 1000.0, PRESSURE_UNITS, "psia", "dead_ref_p", step=50.0
                )
                reference_pressure_psia = pressure_to_psia(ref_p_value, ref_p_unit)
                reference_bo = st.number_input("Reference Bo (rb/STB)", value=1.030, min_value=0.1, step=0.005)

            coeff_col, mu_col = st.columns(2)
            with coeff_col:
                st.subheader("Pressure response")
                co_value, co_unit = number_with_unit(
                    "Oil compressibility", 1.0e-5, COMPRESSIBILITY_UNITS, "1/psi", "dead_co", min_value=0.0, step=1e-6
                )
                co_per_psi = compressibility_to_per_psi(co_value, co_unit)
                alpha_value, alpha_unit = number_with_unit(
                    "Viscosity pressure coefficient",
                    2.0e-5,
                    COMPRESSIBILITY_UNITS,
                    "1/psi",
                    "dead_alpha",
                    min_value=0.0,
                    step=1e-6,
                )
                alpha_mu_per_psi = compressibility_to_per_psi(alpha_value, alpha_unit)
            with mu_col:
                st.subheader("Reference viscosity")
                manual_mu = st.toggle("Enter measured reference viscosity", value=False)
                if manual_mu:
                    mu_value, mu_unit = number_with_unit(
                        "Reference viscosity", 2.0, VISCOSITY_UNITS, "cP", "dead_mu", min_value=0.001, step=0.1
                    )
                    manual_mu_cp = viscosity_to_cp(mu_value, mu_unit)
                else:
                    manual_mu_cp = None
                    st.info("Reference viscosity will be calculated from oil gravity and temperature.")

    else:
        gas_col, composition_col = st.columns(2)
        with gas_col:
            st.subheader("Gas definition")
            _, gas_gravity_kind, gas_sg = gas_gravity_input(default_sg=0.68 if fluid == "Dry Gas" else 0.78)
            st.caption(f"Converted internally to **gas SG = {gas_sg:.4f}**.")
        with composition_col:
            st.subheader("Acid-gas correction")
            sour_enabled = st.toggle("Apply Wichert–Aziz correction", value=False)
            if sour_enabled:
                co2_pct = st.number_input("CO₂ (mol%)", 0.0, 70.0, 2.0, 0.1)
                h2s_pct = st.number_input("H₂S (mol%)", 0.0, 70.0, 0.0, 0.1)
                if co2_pct + h2s_pct >= 100.0:
                    st.error("CO₂ + H₂S must be less than 100 mol%.")
                    st.stop()
            else:
                co2_pct = h2s_pct = 0.0

        if fluid == "Wet Gas":
            wet_col1, wet_col2 = st.columns(2)
            with wet_col1:
                st.subheader("Condensate screening inputs")
                condensate_api = st.number_input("Stock-tank condensate gravity (°API)", 25.0, 80.0, 52.0, 0.5)
                cgr_value, cgr_unit = number_with_unit(
                    "Condensate-gas ratio", 35.0, CGR_UNITS, "STB/MMscf", "cgr", min_value=0.0, step=1.0
                )
                cgr_stb_mmscf = cgr_to_stb_mmscf(cgr_value, cgr_unit)
            with wet_col2:
                st.subheader("Phase marker")
                dew_value, dew_unit = number_with_unit(
                    "Dew-point pressure", 3500.0, PRESSURE_UNITS, "psia", "dew", step=50.0
                )
                dew_point_psia = pressure_to_psia(dew_value, dew_unit)
                st.warning(
                    "The wet-gas mode estimates gas-phase properties and marks dew point. It does not calculate retrograde condensate dropout below dew point."
                )
        else:
            sour_enabled = bool(sour_enabled)

with correlation_tab:
    oil_corr_col, gas_corr_col = st.columns(2)
    if fluid in ("Black Oil", "Dead Oil"):
        with oil_corr_col:
            st.subheader("Oil correlations")
            if fluid == "Black Oil":
                rs_corr = st.selectbox("Solution GOR / bubble point", tuple(RS_FUNCTIONS))
                bo_corr = st.selectbox("Saturated oil FVF", tuple(BO_FUNCTIONS))
                sat_mu_corr = st.selectbox("Saturated oil viscosity", tuple(SAT_VISCOSITY_FUNCTIONS))
            else:
                rs_corr = bo_corr = sat_mu_corr = "Not applicable"
            dead_mu_corr = st.selectbox("Dead-oil viscosity", tuple(DEAD_VISCOSITY_FUNCTIONS))
            if fluid == "Black Oil":
                st.markdown(
                    '<div class="correlation-card">Above Pb, Bo uses the entered oil compressibility and viscosity uses the Vasquez–Beggs pressure extension.</div>',
                    unsafe_allow_html=True,
                )
    if fluid in ("Black Oil", "Dry Gas", "Wet Gas"):
        with gas_corr_col:
            st.subheader("Gas correlations")
            ppc_corr = st.selectbox("Pseudo-critical properties", tuple(PSEUDOCRITICAL_FUNCTIONS))
            z_corr = st.selectbox("Gas z-factor", Z_FACTOR_CORRELATIONS)
            st.selectbox("Gas viscosity", ("Lee–Gonzalez–Eakin (1966)",), disabled=True)
            compare_z = st.toggle("Compare both z-factor correlations", value=False)
    else:
        ppc_corr = z_corr = "Not applicable"
        compare_z = False


if not run_pvt:
    st.info("Set or change the parameters, then click **Run PVT** in the sidebar to generate the curves and table.")
    st.stop()


try:
    if fluid == "Black Oil":
        oil_options = OilOptions(rs_corr, bo_corr, dead_mu_corr, sat_mu_corr)
        gas_options = GasOptions(
            ppc_corr,
            z_corr,
            co2_pct / 100.0,
            h2s_pct / 100.0,
            sour_enabled,
        )
        raw_table, model_meta = calculate_black_oil_table(
            pressures,
            t_f,
            api,
            gas_sg,
            rs_at_pb,
            pb_input,
            co_per_psi,
            oil_options,
            gas_options,
        )
        correlation_metadata = {
            "Rs_Pb_Correlation": rs_corr,
            "Bo_Correlation": bo_corr,
            "Dead_Oil_Viscosity_Correlation": dead_mu_corr,
            "Saturated_Oil_Viscosity_Correlation": sat_mu_corr,
            "Undersaturated_Viscosity_Correlation": "Vasquez–Beggs",
            "Gas_Pseudocritical_Correlation": ppc_corr,
            "Z_Factor_Correlation": z_corr,
            "Gas_Viscosity_Correlation": "Lee–Gonzalez–Eakin",
            "Sour_Gas_Correction": "Wichert–Aziz" if sour_enabled else "None",
        }
    elif fluid == "Dead Oil":
        raw_table, model_meta = calculate_dead_oil_table(
            pressures,
            t_f,
            api,
            reference_pressure_psia,
            reference_bo,
            co_per_psi,
            dead_mu_corr,
            manual_mu_cp,
            alpha_mu_per_psi,
        )
        correlation_metadata = {
            "Dead_Oil_Viscosity_Correlation": dead_mu_corr if manual_mu_cp is None else "Measured input",
            "Bo_Pressure_Model": "Constant compressibility exponential",
            "Viscosity_Pressure_Model": "Exponential coefficient",
        }
    else:
        gas_options = GasOptions(
            ppc_corr,
            z_corr,
            co2_pct / 100.0,
            h2s_pct / 100.0,
            sour_enabled,
        )
        raw_table, model_meta = calculate_gas_table(pressures, t_f, gas_sg, gas_options)
        if fluid == "Wet Gas":
            raw_table["CGR_stb_mmscf"] = cgr_stb_mmscf
            raw_table["condensate_density_lbft3"] = 62.366 * api_to_sg_oil(condensate_api)
            raw_table["phase_screening"] = np.where(
                raw_table["P_psia"] >= dew_point_psia,
                "Above dew point",
                "Below dew point — EOS required",
            )
            raw_table.attrs["dew_point_psia"] = dew_point_psia
        correlation_metadata = {
            "Gas_Pseudocritical_Correlation": ppc_corr,
            "Z_Factor_Correlation": z_corr,
            "Gas_Viscosity_Correlation": "Lee–Gonzalez–Eakin",
            "Sour_Gas_Correction": "Wichert–Aziz" if sour_enabled else "None",
        }
except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
    st.error(f"The selected inputs produced an invalid calculation: {exc}")
    st.stop()


output_table = format_output_table(
    raw_table,
    output_pressure_unit,
    output_gor_unit,
    output_bg_unit,
    output_viscosity_unit,
    output_density_unit,
    output_compressibility_unit,
)
if fluid == "Wet Gas":
    output_table[f"CGR [{cgr_unit}]"] = cgr_from_stb_mmscf(raw_table["CGR_stb_mmscf"].to_numpy(), cgr_unit)
    output_table[f"Condensate density [{output_density_unit}]"] = density_from_lbft3(
        raw_table["condensate_density_lbft3"].to_numpy(), output_density_unit
    )
    output_table["phase_screening"] = raw_table["phase_screening"]


pressure_col = f"Pressure [{output_pressure_unit}]"
x = output_table[pressure_col].to_numpy()
x_title = pressure_col
marker_x = None
marker_label = ""
if fluid == "Black Oil":
    marker_x = pressure_from_psia(model_meta["Pb_psia"], output_pressure_unit)
    marker_label = f"Pb = {marker_x:,.1f} {output_pressure_unit}"
elif fluid == "Wet Gas":
    marker_x = pressure_from_psia(dew_point_psia, output_pressure_unit)
    marker_label = f"Pdew = {marker_x:,.1f} {output_pressure_unit}"


st.subheader("Key results")
if fluid == "Black Oil":
    metric_cols = st.columns(5)
    metric_cols[0].metric("Bubble point", f"{marker_x:,.1f} {output_pressure_unit}")
    metric_cols[1].metric("Rs at Pb", f"{gor_from_scfstb(model_meta['Rsb_scfstb'], output_gor_unit):,.1f} {output_gor_unit}")
    metric_cols[2].metric("Bo at Pb", f"{model_meta['Bo_at_Pb']:.4f} rb/STB")
    metric_cols[3].metric("μo at Pb", f"{viscosity_from_cp(model_meta['mu_at_Pb_cp'], output_viscosity_unit):.4g} {output_viscosity_unit}")
    metric_cols[4].metric("Oil SG", f"{model_meta['oil_sg']:.4f}")
elif fluid == "Dead Oil":
    metric_cols = st.columns(4)
    metric_cols[0].metric("Reference Bo", f"{model_meta['reference_bo']:.4f}")
    metric_cols[1].metric("Reference μo", f"{viscosity_from_cp(model_meta['reference_mu_cp'], output_viscosity_unit):.4g} {output_viscosity_unit}")
    metric_cols[2].metric("Oil gravity", f"{api:.2f} °API")
    metric_cols[3].metric("Oil SG", f"{model_meta['oil_sg']:.4f}")
else:
    metric_cols = st.columns(5 if fluid == "Wet Gas" else 4)
    metric_cols[0].metric("Pseudo-critical P", f"{pressure_from_psia(model_meta['Ppc_psia'], output_pressure_unit):,.1f} {output_pressure_unit}")
    metric_cols[1].metric("Pseudo-critical T", f"{temperature_from_f(model_meta['Tpc_R'] - 459.67, t_unit):,.1f} {t_unit}")
    metric_cols[2].metric("Reduced temperature", f"{model_meta['Tpr']:.3f}")
    metric_cols[3].metric("Molecular weight", f"{model_meta['molecular_weight']:.3f} g/mol")
    if fluid == "Wet Gas":
        metric_cols[4].metric("Dew point", f"{marker_x:,.1f} {output_pressure_unit}")


plot_tab, data_tab, qa_tab = st.tabs(["Interactive curves", "PVT table & CSV", "Engineering QA"])

with plot_tab:
    if fluid == "Black Oil":
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                line_chart(
                    x,
                    [
                        ("Bo", output_table["Bo [rb/STB]"].to_numpy(), COLORS["blue"], "solid"),
                        ("Bt", output_table["Bt [rb/STB]"].to_numpy(), COLORS["cyan"], "dash"),
                    ],
                    "Oil formation volume factors",
                    x_title,
                    "FVF [rb/STB]",
                    marker_x,
                    marker_label,
                ),
                width="stretch",
            )
        with col2:
            rs_column = f"Rs [{output_gor_unit}]"
            st.plotly_chart(
                line_chart(
                    x,
                    [("Rs", output_table[rs_column].to_numpy(), COLORS["purple"], "solid")],
                    "Solution gas–oil ratio",
                    x_title,
                    rs_column,
                    marker_x,
                    marker_label,
                ),
                width="stretch",
            )
        col3, col4 = st.columns(2)
        with col3:
            oil_mu_col = f"Oil viscosity [{output_viscosity_unit}]"
            gas_mu_col = f"Gas viscosity [{output_viscosity_unit}]"
            st.plotly_chart(
                dual_axis_chart(
                    x,
                    ("Oil viscosity", output_table[oil_mu_col].to_numpy(), COLORS["navy"]),
                    ("Gas viscosity", output_table[gas_mu_col].to_numpy(), COLORS["orange"]),
                    "Oil and gas viscosity",
                    x_title,
                    oil_mu_col,
                    gas_mu_col,
                    marker_x,
                    marker_label,
                ),
                width="stretch",
            )
        with col4:
            bg_col = f"Bg [{output_bg_unit}]"
            st.plotly_chart(
                dual_axis_chart(
                    x,
                    ("Bg", output_table[bg_col].to_numpy(), COLORS["green"]),
                    ("z-factor", output_table["z"].to_numpy(), COLORS["red"]),
                    "Gas FVF and z-factor",
                    x_title,
                    bg_col,
                    "z [–]",
                    marker_x,
                    marker_label,
                ),
                width="stretch",
            )
    elif fluid == "Dead Oil":
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                line_chart(
                    x,
                    [("Bo", output_table["Bo [rb/STB]"].to_numpy(), COLORS["blue"], "solid")],
                    "Dead-oil formation volume factor",
                    x_title,
                    "Bo [rb/STB]",
                ),
                width="stretch",
            )
        with col2:
            mu_col_name = f"Oil viscosity [{output_viscosity_unit}]"
            st.plotly_chart(
                line_chart(
                    x,
                    [("Oil viscosity", output_table[mu_col_name].to_numpy(), COLORS["navy"], "solid")],
                    "Dead-oil viscosity",
                    x_title,
                    mu_col_name,
                ),
                width="stretch",
            )
        density_col_name = f"Oil density [{output_density_unit}]"
        st.plotly_chart(
            line_chart(
                x,
                [("Oil density", output_table[density_col_name].to_numpy(), COLORS["green"], "solid")],
                "Dead-oil density",
                x_title,
                density_col_name,
            ),
            width="stretch",
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            z_series = [(z_corr, output_table["z"].to_numpy(), COLORS["blue"], "solid")]
            if compare_z:
                other_z_corr = next(item for item in Z_FACTOR_CORRELATIONS if item != z_corr)
                comparison_options = GasOptions(
                    ppc_corr,
                    other_z_corr,
                    co2_pct / 100.0,
                    h2s_pct / 100.0,
                    sour_enabled,
                )
                comparison_raw, _ = calculate_gas_table(pressures, t_f, gas_sg, comparison_options)
                z_series.append((other_z_corr, comparison_raw["z"].to_numpy(), COLORS["orange"], "dash"))
            st.plotly_chart(
                line_chart(x, z_series, "Gas compressibility factor", x_title, "z [–]", marker_x, marker_label),
                width="stretch",
            )
        with col2:
            bg_col = f"Bg [{output_bg_unit}]"
            st.plotly_chart(
                line_chart(
                    x,
                    [("Bg", output_table[bg_col].to_numpy(), COLORS["green"], "solid")],
                    "Gas formation volume factor",
                    x_title,
                    bg_col,
                    marker_x,
                    marker_label,
                ),
                width="stretch",
            )
        col3, col4 = st.columns(2)
        with col3:
            mu_col_name = f"Gas viscosity [{output_viscosity_unit}]"
            st.plotly_chart(
                line_chart(
                    x,
                    [("Gas viscosity", output_table[mu_col_name].to_numpy(), COLORS["orange"], "solid")],
                    "Gas viscosity",
                    x_title,
                    mu_col_name,
                    marker_x,
                    marker_label,
                ),
                width="stretch",
            )
        with col4:
            rho_col_name = f"Gas density [{output_density_unit}]"
            st.plotly_chart(
                line_chart(
                    x,
                    [("Gas density", output_table[rho_col_name].to_numpy(), COLORS["purple"], "solid")],
                    "Gas density",
                    x_title,
                    rho_col_name,
                    marker_x,
                    marker_label,
                ),
                width="stretch",
            )
        cg_col_name = f"Gas compressibility [{output_compressibility_unit}]"
        st.plotly_chart(
            line_chart(
                x,
                [("Gas compressibility", output_table[cg_col_name].to_numpy(), COLORS["red"], "solid")],
                "Isothermal gas compressibility",
                x_title,
                cg_col_name,
                marker_x,
                marker_label,
            ),
            width="stretch",
        )

with data_tab:
    st.dataframe(output_table, width="stretch", hide_index=True, height=480)
    export_metadata: dict[str, object] = {
        "Fluid_System": fluid,
        "Reservoir_Temperature_Input": f"{t_value:g} {t_unit}",
        **correlation_metadata,
    }
    export_df = add_export_metadata(output_table, export_metadata)
    csv_bytes = export_df.to_csv(index=False, float_format="%.8g").encode("utf-8-sig")
    filename = f"pvt_{fluid.lower().replace(' ', '_')}_{datetime.now(timezone.utc):%Y%m%d_%H%M}.csv"
    st.download_button(
        "Download PVT table as CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        type="primary",
        on_click="ignore",
    )
    st.caption("The export uses UTF-8 with BOM for clean opening in Excel and includes fluid type and correlation names on every row.")

with qa_tab:
    warnings = correlation_range_warnings(fluid, raw_table, t_f, api, gas_sg)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No broad correlation-range warning was triggered for the selected inputs.")

    st.markdown("#### Active model")
    active_model = pd.DataFrame(
        {"Setting": list(correlation_metadata.keys()), "Selection": list(correlation_metadata.values())}
    )
    st.dataframe(active_model, width="stretch", hide_index=True)

    qa1, qa2, qa3, qa4 = st.columns(4)
    qa1.metric("Minimum absolute P", f"{p_min_psia:,.1f} psia")
    qa2.metric("Maximum absolute P", f"{p_max_psia:,.1f} psia")
    qa3.metric("Temperature", f"{t_f:,.1f} °F")
    qa4.metric("Rows generated", f"{len(raw_table):,}")

    st.info(
        "Correlation results are screening estimates. Calibrate against representative laboratory PVT data before reserves, simulation, facility-design, or operating decisions."
    )
    with st.expander("Important modeling notes"):
        st.markdown(
            """
            - Inputs are converted to psia, °F, scf/STB, cP and lb/ft³ before calculation.
            - `Bg` is calculated first in ft³/scf. The rb/Mscf output includes the required 5.6146 ft³/bbl conversion.
            - Above bubble point, `Rs` is held constant and `Bo` uses the entered constant oil compressibility.
            - Wet-gas mode does not represent retrograde liquid dropout; a tuned Peng–Robinson or SRK EOS is needed below dew point.
            - Sour-gas correction uses CO₂ and H₂S. Nitrogen is reflected only indirectly through the entered gas gravity.
            """
        )

st.caption("PVT Studio • correlation-based engineering screening tool")
