import re

import numpy as np
import pytest

from pvt_engine import (
    GasOptions,
    OilOptions,
    calculate_black_oil_table,
    calculate_dead_oil_table,
    calculate_gas_table,
    format_output_table,
)
from pvt_export import build_ascii_export, build_eclipse_include


@pytest.fixture(scope="module")
def sample_tables():
    pressure = np.linspace(100.0, 6000.0, 40)
    gas, gas_meta = calculate_gas_table(pressure, 180.0, 0.70, GasOptions())
    black, black_meta = calculate_black_oil_table(
        pressure,
        180.0,
        35.0,
        0.70,
        650.0,
        None,
        1.2e-5,
        OilOptions(),
        GasOptions(),
    )
    dead, dead_meta = calculate_dead_oil_table(
        pressure,
        180.0,
        35.0,
        1000.0,
        1.03,
        1.0e-5,
        "Beggs–Robinson (1975)",
        None,
        2.0e-5,
    )
    return gas, gas_meta, black, black_meta, dead, dead_meta


def test_ascii_export_is_seven_bit_and_self_describing(sample_tables):
    gas, _, *_ = sample_tables
    formatted = format_output_table(
        gas, "bar(a)", "Sm³/Sm³", "rm³/Sm³", "mPa·s", "kg/m³", "1/bar"
    )
    payload = build_ascii_export(formatted, {"Fluid_System": "Dry Gas", "Symbol": "μ"})
    text = payload.decode("ascii")
    assert max(payload) < 128
    assert "# Fluid_System: Dry Gas" in text
    assert "Gas_viscosity_[mPa.s]" in text
    assert "\t" in text


def test_black_oil_field_export_has_live_oil_and_gas_keywords(sample_tables):
    _, _, black, black_meta, *_ = sample_tables
    text = build_eclipse_include(
        "Black Oil", black, black_meta, {"Rs": "Standing (1947)"}, "FIELD"
    ).decode("ascii")
    assert "\nPVTO\n" in text
    assert "\nPVDG\n" in text
    assert "MSCF/STB" in text
    assert re.search(r"\s0\.65\s+2790\.\d+", text)
    assert "-- Suggested TABDIMS capacity:" in text
    assert text.rstrip().endswith("/")


def test_dead_oil_metric_export_converts_pressure(sample_tables):
    *_, dead, dead_meta = sample_tables
    text = build_eclipse_include("Dead Oil", dead, dead_meta, unit_system="METRIC").decode("ascii")
    assert "\nPVDO\n" in text
    assert "BARSA" in text
    assert "6.894757293" in text  # 100 psia in bar(a)
    assert "\nPVDG\n" not in text


def test_dry_and_wet_gas_keywords_and_rv_conversion(sample_tables):
    gas, gas_meta, *_ = sample_tables
    dry = build_eclipse_include("Dry Gas", gas, gas_meta, unit_system="FIELD").decode("ascii")
    wet = build_eclipse_include(
        "Wet Gas",
        gas,
        gas_meta,
        unit_system="FIELD",
        cgr_stb_mmscf=35.0,
        dew_point_psia=3500.0,
    ).decode("ascii")
    assert "\nPVDG\n" in dry and "\nPVTG\n" not in dry
    assert "\nPVTG\n" in wet and "constant Rv" in wet
    assert re.search(r"\s100\s+0\.035\s+", wet)  # 35 STB/MMscf = 0.035 STB/Mscf


def test_pvto_explains_missing_saturated_range(sample_tables):
    _, _, black, black_meta, *_ = sample_tables
    above_pb = black[black["P_psia"] > black_meta["Pb_psia"]]
    with pytest.raises(ValueError, match="below bubble point"):
        build_eclipse_include("Black Oil", above_pb, black_meta, unit_system="FIELD")

