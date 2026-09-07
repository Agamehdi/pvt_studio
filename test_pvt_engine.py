import numpy as np

from pvt_engine import (
    GasOptions,
    OilOptions,
    api_to_oil_input,
    bg_from_ft3_scf,
    calculate_black_oil_table,
    calculate_gas_table,
    gor_from_scfstb,
    gor_to_scfstb,
    gas_input_to_sg,
    gas_sg_to_input,
    oil_input_to_api,
    pressure_from_psia,
    pressure_to_psia,
)


def test_unit_round_trips():
    pressures = np.array([14.6959488, 1000.0, 6000.0])
    for unit in ("psia", "psig", "bar(a)", "bar(g)", "kPa(a)", "MPa(a)"):
        converted = pressure_from_psia(pressures, unit)
        assert np.allclose(pressure_to_psia(converted, unit), pressures)

    gor = np.array([0.0, 650.0, 1200.0])
    assert np.allclose(gor_to_scfstb(gor_from_scfstb(gor, "Sm³/Sm³"), "Sm³/Sm³"), gor)


def test_gravity_basis_round_trips():
    api = 35.0
    for basis in (
        "API gravity (°API)",
        "Oil specific gravity",
        "Stock-tank density (kg/m³)",
        "Stock-tank density (lb/ft³)",
    ):
        assert np.isclose(oil_input_to_api(api_to_oil_input(api, basis), basis), api)

    gas_sg = 0.70
    for basis in ("Gas specific gravity (air=1)", "Molecular weight (g/mol)"):
        assert np.isclose(gas_input_to_sg(gas_sg_to_input(gas_sg, basis), basis), gas_sg)


def test_bg_conversion_uses_barrels():
    assert np.isclose(bg_from_ft3_scf(5.6145833333, "rb/Mscf"), 1000.0)


def test_black_oil_has_physical_shape():
    pressure = np.linspace(100.0, 6000.0, 160)
    table, metadata = calculate_black_oil_table(
        pressure,
        t_f=180.0,
        api=35.0,
        gas_sg=0.70,
        rs_at_pb_scfstb=650.0,
        pb_psia=None,
        co_per_psi=1.2e-5,
        oil_options=OilOptions(),
        gas_options=GasOptions(),
    )
    pb = metadata["Pb_psia"]
    below = table[table.P_psia <= pb]
    above = table[table.P_psia >= pb]
    assert 2000.0 < pb < 4000.0
    assert below.Rs_scfstb.is_monotonic_increasing
    assert below.Bo_rb_stb.is_monotonic_increasing
    assert above.Bo_rb_stb.is_monotonic_decreasing
    assert np.isfinite(table.select_dtypes(include=[float])).all().all()
    assert (table.Bo_rb_stb > 0.0).all()
    assert (table.z > 0.0).all()


def test_dry_gas_has_physical_outputs():
    pressure = np.linspace(100.0, 6000.0, 100)
    table, _ = calculate_gas_table(pressure, 180.0, 0.70, GasOptions())
    assert table.Bg_ft3_scf.iloc[0] > table.Bg_ft3_scf.iloc[-1]
    assert table.rho_g_lbft3.iloc[0] < table.rho_g_lbft3.iloc[-1]
    assert (table.mu_g_cp > 0.0).all()
    assert (table.cg_per_psi > 0.0).all()
