"""Correlation-based petroleum PVT calculations for the Streamlit app.

All public calculation functions use oilfield units internally:
pressure in psia, temperature in degF, solution GOR in scf/STB,
viscosity in cP, and density in lb/ft3.

The module intentionally keeps unit conversion separate from correlations so
that calculation logic remains auditable and easy to validate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


ATM_PSI = 14.6959488
PSI_PER_BAR = 14.503773773
PSI_PER_KPA = 0.1450377377
PSI_PER_MPA = 145.0377377
AIR_MW = 28.967
LBFT3_PER_KGM3 = 0.0624279606
FT3_PER_BBL = 5.6145833333
SCFSTB_TO_SM3SM3 = 0.1781076067


PRESSURE_UNITS = ("psia", "psig", "bar(a)", "bar(g)", "kPa(a)", "MPa(a)")
ABS_PRESSURE_UNITS = ("psia", "bar(a)", "kPa(a)", "MPa(a)")
TEMPERATURE_UNITS = ("°F", "°C", "K", "°R")
GOR_UNITS = ("scf/STB", "Sm³/Sm³")
COMPRESSIBILITY_UNITS = ("1/psi", "1/bar", "1/kPa", "1/MPa")
VISCOSITY_UNITS = ("cP", "mPa·s", "Pa·s")
DENSITY_UNITS = ("lb/ft³", "kg/m³")
BG_UNITS = ("rb/Mscf", "ft³/scf", "rm³/Sm³")
CGR_UNITS = ("STB/MMscf", "m³/MMsm³")


def pressure_to_psia(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    if unit == "psia":
        result = value
    elif unit == "psig":
        result = value + ATM_PSI
    elif unit == "bar(a)":
        result = value * PSI_PER_BAR
    elif unit == "bar(g)":
        result = (value + 1.01325) * PSI_PER_BAR
    elif unit == "kPa(a)":
        result = value * PSI_PER_KPA
    elif unit == "MPa(a)":
        result = value * PSI_PER_MPA
    else:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return float(result) if result.ndim == 0 else result


def pressure_from_psia(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    if unit == "psia":
        result = value
    elif unit == "psig":
        result = value - ATM_PSI
    elif unit == "bar(a)":
        result = value / PSI_PER_BAR
    elif unit == "bar(g)":
        result = value / PSI_PER_BAR - 1.01325
    elif unit == "kPa(a)":
        result = value / PSI_PER_KPA
    elif unit == "MPa(a)":
        result = value / PSI_PER_MPA
    else:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return float(result) if result.ndim == 0 else result


def temperature_to_f(value: float, unit: str) -> float:
    if unit == "°F":
        return float(value)
    if unit == "°C":
        return float(value * 9.0 / 5.0 + 32.0)
    if unit == "K":
        return float((value - 273.15) * 9.0 / 5.0 + 32.0)
    if unit == "°R":
        return float(value - 459.67)
    raise ValueError(f"Unsupported temperature unit: {unit}")


def temperature_from_f(value: float, unit: str) -> float:
    if unit == "°F":
        return float(value)
    if unit == "°C":
        return float((value - 32.0) * 5.0 / 9.0)
    if unit == "K":
        return float((value - 32.0) * 5.0 / 9.0 + 273.15)
    if unit == "°R":
        return float(value + 459.67)
    raise ValueError(f"Unsupported temperature unit: {unit}")


def gor_to_scfstb(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    if unit == "scf/STB":
        result = value
    elif unit == "Sm³/Sm³":
        result = value / SCFSTB_TO_SM3SM3
    else:
        raise ValueError(f"Unsupported GOR unit: {unit}")
    return float(result) if result.ndim == 0 else result


def gor_from_scfstb(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    if unit == "scf/STB":
        result = value
    elif unit == "Sm³/Sm³":
        result = value * SCFSTB_TO_SM3SM3
    else:
        raise ValueError(f"Unsupported GOR unit: {unit}")
    return float(result) if result.ndim == 0 else result


def compressibility_to_per_psi(value: float, unit: str) -> float:
    pressure_factor = {
        "1/psi": 1.0,
        "1/bar": PSI_PER_BAR,
        "1/kPa": PSI_PER_KPA,
        "1/MPa": PSI_PER_MPA,
    }.get(unit)
    if pressure_factor is None:
        raise ValueError(f"Unsupported compressibility unit: {unit}")
    return float(value / pressure_factor)


def compressibility_from_per_psi(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    pressure_factor = {
        "1/psi": 1.0,
        "1/bar": PSI_PER_BAR,
        "1/kPa": PSI_PER_KPA,
        "1/MPa": PSI_PER_MPA,
    }.get(unit)
    if pressure_factor is None:
        raise ValueError(f"Unsupported compressibility unit: {unit}")
    result = value * pressure_factor
    return float(result) if result.ndim == 0 else result


def viscosity_to_cp(value: float, unit: str) -> float:
    if unit in ("cP", "mPa·s"):
        return float(value)
    if unit == "Pa·s":
        return float(value * 1000.0)
    raise ValueError(f"Unsupported viscosity unit: {unit}")


def viscosity_from_cp(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    if unit in ("cP", "mPa·s"):
        result = value
    elif unit == "Pa·s":
        result = value / 1000.0
    else:
        raise ValueError(f"Unsupported viscosity unit: {unit}")
    return float(result) if result.ndim == 0 else result


def density_from_lbft3(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    if unit == "lb/ft³":
        result = value
    elif unit == "kg/m³":
        result = value / LBFT3_PER_KGM3
    else:
        raise ValueError(f"Unsupported density unit: {unit}")
    return float(result) if result.ndim == 0 else result


def bg_from_ft3_scf(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    if unit == "ft³/scf" or unit == "rm³/Sm³":
        result = value
    elif unit == "rb/Mscf":
        result = value * 1000.0 / FT3_PER_BBL
    else:
        raise ValueError(f"Unsupported Bg unit: {unit}")
    return float(result) if result.ndim == 0 else result


def cgr_to_stb_mmscf(value: float, unit: str) -> float:
    if unit == "STB/MMscf":
        return float(value)
    if unit == "m³/MMsm³":
        return float(value / FT3_PER_BBL)
    raise ValueError(f"Unsupported CGR unit: {unit}")


def cgr_from_stb_mmscf(value: float | np.ndarray, unit: str) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    if unit == "STB/MMscf":
        result = value
    elif unit == "m³/MMsm³":
        result = value * FT3_PER_BBL
    else:
        raise ValueError(f"Unsupported CGR unit: {unit}")
    return float(result) if result.ndim == 0 else result


def api_to_sg_oil(api: float) -> float:
    return float(141.5 / (api + 131.5))


def sg_oil_to_api(sg: float) -> float:
    return float(141.5 / sg - 131.5)


def oil_input_to_api(value: float, kind: str) -> float:
    if kind == "API gravity (°API)":
        return float(value)
    if kind == "Oil specific gravity":
        return sg_oil_to_api(value)
    if kind == "Stock-tank density (kg/m³)":
        return sg_oil_to_api(value / 999.016)
    if kind == "Stock-tank density (lb/ft³)":
        return sg_oil_to_api(value / 62.366)
    raise ValueError(f"Unsupported oil gravity input: {kind}")


def api_to_oil_input(api: float, kind: str) -> float:
    if kind == "API gravity (°API)":
        return float(api)
    oil_sg = api_to_sg_oil(api)
    if kind == "Oil specific gravity":
        return oil_sg
    if kind == "Stock-tank density (kg/m³)":
        return float(oil_sg * 999.016)
    if kind == "Stock-tank density (lb/ft³)":
        return float(oil_sg * 62.366)
    raise ValueError(f"Unsupported oil gravity input: {kind}")


def gas_input_to_sg(value: float, kind: str) -> float:
    if kind == "Gas specific gravity (air=1)":
        return float(value)
    if kind == "Molecular weight (g/mol)":
        return float(value / AIR_MW)
    raise ValueError(f"Unsupported gas gravity input: {kind}")


def gas_sg_to_input(gas_sg: float, kind: str) -> float:
    if kind == "Gas specific gravity (air=1)":
        return float(gas_sg)
    if kind == "Molecular weight (g/mol)":
        return float(gas_sg * AIR_MW)
    raise ValueError(f"Unsupported gas gravity input: {kind}")


# ---------------------------------------------------------------------------
# Oil correlations
# ---------------------------------------------------------------------------


def standing_rs(p_psia: float | np.ndarray, t_f: float, api: float, gas_sg: float) -> np.ndarray:
    p = np.asarray(p_psia, dtype=float)
    x = 0.0125 * api - 0.00091 * t_f
    return gas_sg * (((p / 18.2) + 1.4) * 10.0**x) ** 1.2048


def standing_pb(rs_scfstb: float, t_f: float, api: float, gas_sg: float) -> float:
    x = 0.0125 * api - 0.00091 * t_f
    a = (rs_scfstb / gas_sg) ** (1.0 / 1.2048)
    return float(18.2 * (a / 10.0**x - 1.4))


def vasquez_beggs_coefficients(api: float) -> tuple[float, float, float]:
    if api <= 30.0:
        return 0.0362, 1.0937, 25.7240
    return 0.0178, 1.1870, 23.9310


def vasquez_beggs_rs(p_psia: float | np.ndarray, t_f: float, api: float, gas_sg: float) -> np.ndarray:
    p = np.asarray(p_psia, dtype=float)
    c1, c2, c3 = vasquez_beggs_coefficients(api)
    return c1 * gas_sg * p**c2 * np.exp(c3 * api / (t_f + 460.0))


def vasquez_beggs_pb(rs_scfstb: float, t_f: float, api: float, gas_sg: float) -> float:
    c1, c2, c3 = vasquez_beggs_coefficients(api)
    base = rs_scfstb / (c1 * gas_sg * np.exp(c3 * api / (t_f + 460.0)))
    return float(base ** (1.0 / c2))


RS_FUNCTIONS: dict[str, Callable[[float | np.ndarray, float, float, float], np.ndarray]] = {
    "Standing (1947)": standing_rs,
    "Vasquez–Beggs (1980)": vasquez_beggs_rs,
}

PB_FUNCTIONS: dict[str, Callable[[float, float, float, float], float]] = {
    "Standing (1947)": standing_pb,
    "Vasquez–Beggs (1980)": vasquez_beggs_pb,
}


def standing_bo(rs_scfstb: float | np.ndarray, t_f: float, api: float, gas_sg: float) -> np.ndarray:
    rs = np.asarray(rs_scfstb, dtype=float)
    oil_sg = api_to_sg_oil(api)
    f = rs * np.sqrt(gas_sg / oil_sg) + 1.25 * t_f
    return 0.972 + 0.000147 * np.maximum(f, 0.0) ** 1.175


def vasquez_beggs_bo(rs_scfstb: float | np.ndarray, t_f: float, api: float, gas_sg: float) -> np.ndarray:
    rs = np.asarray(rs_scfstb, dtype=float)
    if api <= 30.0:
        c1, c2, c3 = 4.677e-4, 1.751e-5, -1.811e-8
    else:
        c1, c2, c3 = 4.670e-4, 1.100e-5, 1.337e-9
    thermal = (t_f - 60.0) * (api / gas_sg)
    return 1.0 + c1 * rs + c2 * thermal + c3 * rs * thermal


BO_FUNCTIONS: dict[str, Callable[[float | np.ndarray, float, float, float], np.ndarray]] = {
    "Standing (1947)": standing_bo,
    "Vasquez–Beggs (1980)": vasquez_beggs_bo,
}


def beggs_robinson_dead_viscosity(t_f: float, api: float) -> float:
    y = 10.0 ** (3.0324 - 0.02023 * api)
    x = y * t_f**-1.163
    return float(10.0**x - 1.0)


def beal_dead_viscosity(t_f: float, api: float) -> float:
    exponent = 10.0 ** (0.43 + 8.33 / api)
    return float((0.32 + 1.8e7 / api**4.53) * (360.0 / (t_f + 200.0)) ** exponent)


DEAD_VISCOSITY_FUNCTIONS: dict[str, Callable[[float, float], float]] = {
    "Beggs–Robinson (1975)": beggs_robinson_dead_viscosity,
    "Beal (1946)": beal_dead_viscosity,
}


def beggs_robinson_saturated_viscosity(mu_dead_cp: float, rs_scfstb: float | np.ndarray) -> np.ndarray:
    rs = np.asarray(rs_scfstb, dtype=float)
    a = 10.715 * (rs + 100.0) ** -0.515
    b = 5.44 * (rs + 150.0) ** -0.338
    return a * mu_dead_cp**b


def chew_connally_saturated_viscosity(mu_dead_cp: float, rs_scfstb: float | np.ndarray) -> np.ndarray:
    rs = np.asarray(rs_scfstb, dtype=float)
    a = rs * (2.2e-7 * rs - 7.4e-4)
    b = (
        0.68 / 10.0 ** (8.62e-5 * rs)
        + 0.25 / 10.0 ** (1.10e-3 * rs)
        + 0.062 / 10.0 ** (3.74e-3 * rs)
    )
    return 10.0**a * mu_dead_cp**b


SAT_VISCOSITY_FUNCTIONS: dict[str, Callable[[float, float | np.ndarray], np.ndarray]] = {
    "Beggs–Robinson (1975)": beggs_robinson_saturated_viscosity,
    "Chew–Connally (1959)": chew_connally_saturated_viscosity,
}


def vasquez_beggs_undersat_viscosity(
    mu_pb_cp: float, p_psia: float | np.ndarray, pb_psia: float
) -> np.ndarray:
    p = np.asarray(p_psia, dtype=float)
    m = 2.6 * p**1.187 * np.exp(-11.513 - 8.98e-5 * p)
    return mu_pb_cp * (p / pb_psia) ** m


# ---------------------------------------------------------------------------
# Gas correlations
# ---------------------------------------------------------------------------


def sutton_pseudocritical(gas_sg: float) -> tuple[float, float]:
    tpc_r = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg**2
    ppc_psia = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg**2
    return float(tpc_r), float(ppc_psia)


def standing_pseudocritical(gas_sg: float) -> tuple[float, float]:
    tpc_r = 168.0 + 325.0 * gas_sg - 12.5 * gas_sg**2
    ppc_psia = 677.0 + 15.0 * gas_sg - 37.5 * gas_sg**2
    return float(tpc_r), float(ppc_psia)


PSEUDOCRITICAL_FUNCTIONS: dict[str, Callable[[float], tuple[float, float]]] = {
    "Sutton (1985)": sutton_pseudocritical,
    "Standing (1977)": standing_pseudocritical,
}


def wichert_aziz_correction(
    tpc_r: float, ppc_psia: float, y_co2: float, y_h2s: float
) -> tuple[float, float]:
    acid = max(0.0, y_co2 + y_h2s)
    h2s = max(0.0, y_h2s)
    epsilon = 120.0 * (acid**0.9 - acid**1.6) + 15.0 * (h2s**0.5 - h2s**4)
    corrected_tpc = tpc_r - epsilon
    denominator = tpc_r + h2s * (1.0 - h2s) * epsilon
    corrected_ppc = ppc_psia * corrected_tpc / denominator
    return float(corrected_tpc), float(corrected_ppc)


DAK_COEFFICIENTS = (
    0.3265,
    -1.0700,
    -0.5339,
    0.01569,
    -0.05165,
    0.5475,
    -0.7361,
    0.1844,
    0.1056,
    0.6134,
    0.7210,
)


def _dak_z_from_density(rho_r: float | np.ndarray, tpr: float) -> np.ndarray:
    rho = np.asarray(rho_r, dtype=float)
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = DAK_COEFFICIENTS
    c1 = a1 + a2 / tpr + a3 / tpr**3 + a4 / tpr**4 + a5 / tpr**5
    c2 = a6 + a7 / tpr + a8 / tpr**2
    c3 = -a9 * (a7 / tpr + a8 / tpr**2)
    c4 = a10 * (1.0 + a11 * rho**2) * rho**2 / tpr**3 * np.exp(-a11 * rho**2)
    return 1.0 + c1 * rho + c2 * rho**2 + c3 * rho**5 + c4


def _bisect_root(func: Callable[[float], float], low: float, high: float, iterations: int = 80) -> float:
    f_low = func(low)
    f_high = func(high)
    if not np.isfinite(f_low) or not np.isfinite(f_high) or f_low * f_high > 0.0:
        raise ValueError("Root is not bracketed")
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        f_mid = func(mid)
        if abs(f_mid) < 1e-11:
            return mid
        if f_low * f_mid <= 0.0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return 0.5 * (low + high)


def z_factor_dak(ppr: float, tpr: float) -> float:
    if ppr <= 0.0 or tpr <= 0.0:
        raise ValueError("Reduced pressure and temperature must be positive")

    def residual(rho: float) -> float:
        return float(0.27 * ppr / (rho * tpr) - _dak_z_from_density(rho, tpr))

    # The smallest positive root is the gas-phase root when multiple roots exist.
    grid = np.geomspace(1e-7, 8.0, 900)
    previous_x = float(grid[0])
    previous_f = residual(previous_x)
    for current_x in grid[1:]:
        current_x = float(current_x)
        current_f = residual(current_x)
        if np.isfinite(previous_f) and np.isfinite(current_f) and previous_f * current_f <= 0.0:
            rho = _bisect_root(residual, previous_x, current_x)
            z = 0.27 * ppr / (rho * tpr)
            if np.isfinite(z) and 0.05 < z < 5.0:
                return float(z)
        previous_x, previous_f = current_x, current_f
    raise RuntimeError("DAK solver did not find a physical gas root")


def z_factor_papay(ppr: float, tpr: float) -> float:
    z = 1.0 - 3.53 * ppr / 10.0 ** (0.9813 * tpr) + 0.274 * ppr**2 / 10.0 ** (0.8157 * tpr)
    return float(z)


def gas_z_factor(ppr: float, tpr: float, correlation: str) -> float:
    if correlation == "Dranchuk–Abou-Kassem (1975)":
        z = z_factor_dak(ppr, tpr)
    elif correlation == "Papay (1968)":
        z = z_factor_papay(ppr, tpr)
    else:
        raise ValueError(f"Unsupported z-factor correlation: {correlation}")
    if not np.isfinite(z) or z <= 0.0:
        raise RuntimeError("The selected z-factor correlation returned a non-physical value")
    return float(z)


Z_FACTOR_CORRELATIONS = ("Dranchuk–Abou-Kassem (1975)", "Papay (1968)")


def lee_gonzalez_eakin_gas_viscosity(p_psia: float, t_f: float, gas_sg: float, z: float) -> float:
    t_r = t_f + 459.67
    molecular_weight = AIR_MW * gas_sg
    rho_gcc = p_psia * molecular_weight / (z * 10.7316 * t_r) * 0.016018463
    k = (9.379 + 0.01607 * molecular_weight) * t_r**1.5 / (
        209.2 + 19.26 * molecular_weight + t_r
    )
    x = 3.448 + 986.4 / t_r + 0.01009 * molecular_weight
    y = 2.447 - 0.2224 * x
    return float(1.0e-4 * k * np.exp(x * rho_gcc**y))


@dataclass(frozen=True)
class GasOptions:
    pseudocritical_correlation: str = "Sutton (1985)"
    z_correlation: str = "Dranchuk–Abou-Kassem (1975)"
    y_co2: float = 0.0
    y_h2s: float = 0.0
    apply_sour_correction: bool = False


@dataclass(frozen=True)
class OilOptions:
    rs_pb_correlation: str = "Standing (1947)"
    bo_correlation: str = "Standing (1947)"
    dead_viscosity_correlation: str = "Beggs–Robinson (1975)"
    saturated_viscosity_correlation: str = "Beggs–Robinson (1975)"


def calculate_gas_table(
    pressures_psia: Iterable[float],
    t_f: float,
    gas_sg: float,
    options: GasOptions,
) -> tuple[pd.DataFrame, dict[str, float]]:
    p = np.asarray(list(pressures_psia), dtype=float)
    if np.any(p <= 0.0):
        raise ValueError("All absolute pressures must be greater than zero")
    if t_f <= -459.0:
        raise ValueError("Temperature must be above absolute zero")
    if gas_sg <= 0.0:
        raise ValueError("Gas gravity must be positive")

    tpc_r, ppc_psia = PSEUDOCRITICAL_FUNCTIONS[options.pseudocritical_correlation](gas_sg)
    if options.apply_sour_correction and options.y_co2 + options.y_h2s > 0.0:
        tpc_r, ppc_psia = wichert_aziz_correction(
            tpc_r, ppc_psia, options.y_co2, options.y_h2s
        )

    t_r = t_f + 459.67
    tpr = t_r / tpc_r
    ppr = p / ppc_psia
    z = np.array([gas_z_factor(float(pr), tpr, options.z_correlation) for pr in ppr])
    bg_ft3_scf = 0.0282793 * z * t_r / p
    molecular_weight = AIR_MW * gas_sg
    rho_g_lbft3 = p * molecular_weight / (z * 10.7316 * t_r)
    mu_g_cp = np.array(
        [lee_gonzalez_eakin_gas_viscosity(float(pi), t_f, gas_sg, float(zi)) for pi, zi in zip(p, z)]
    )

    # Real-gas compressibility: cg = 1/P - (1/z)(dz/dP).
    edge_order = 2 if len(p) >= 3 else 1
    dz_dp = np.gradient(z, p, edge_order=edge_order)
    cg_per_psi = 1.0 / p - dz_dp / z

    table = pd.DataFrame(
        {
            "P_psia": p,
            "Ppr": ppr,
            "Tpr": np.full_like(p, tpr),
            "z": z,
            "Bg_ft3_scf": bg_ft3_scf,
            "mu_g_cp": mu_g_cp,
            "rho_g_lbft3": rho_g_lbft3,
            "cg_per_psi": cg_per_psi,
        }
    )
    metadata = {
        "Tpc_R": tpc_r,
        "Ppc_psia": ppc_psia,
        "Tpr": tpr,
        "molecular_weight": molecular_weight,
    }
    return table, metadata


def calculate_black_oil_table(
    pressures_psia: Iterable[float],
    t_f: float,
    api: float,
    gas_sg: float,
    rs_at_pb_scfstb: float | None,
    pb_psia: float | None,
    co_per_psi: float,
    oil_options: OilOptions,
    gas_options: GasOptions,
) -> tuple[pd.DataFrame, dict[str, float]]:
    p = np.asarray(list(pressures_psia), dtype=float)
    if np.any(p <= 0.0):
        raise ValueError("All absolute pressures must be greater than zero")
    if not 5.0 <= api <= 100.0:
        raise ValueError("Oil gravity must be between 5 and 100 °API")
    if not 0.3 <= gas_sg <= 2.0:
        raise ValueError("Gas specific gravity must be between 0.3 and 2.0")
    if co_per_psi < 0.0:
        raise ValueError("Oil compressibility cannot be negative")

    rs_fn = RS_FUNCTIONS[oil_options.rs_pb_correlation]
    pb_fn = PB_FUNCTIONS[oil_options.rs_pb_correlation]
    bo_fn = BO_FUNCTIONS[oil_options.bo_correlation]
    dead_mu_fn = DEAD_VISCOSITY_FUNCTIONS[oil_options.dead_viscosity_correlation]
    sat_mu_fn = SAT_VISCOSITY_FUNCTIONS[oil_options.saturated_viscosity_correlation]

    if rs_at_pb_scfstb is not None:
        rsb = float(rs_at_pb_scfstb)
        pb = pb_fn(rsb, t_f, api, gas_sg)
    elif pb_psia is not None:
        pb = float(pb_psia)
        rsb = float(np.asarray(rs_fn(pb, t_f, api, gas_sg)))
    else:
        raise ValueError("Provide either Rs at bubble point or bubble-point pressure")

    if pb <= 0.0 or rsb < 0.0:
        raise ValueError("The selected inputs produced a non-physical saturation anchor")

    rs_saturated = rs_fn(np.minimum(p, pb), t_f, api, gas_sg)
    rs = np.where(p <= pb, np.minimum(rs_saturated, rsb), rsb)
    rs = np.maximum(rs, 0.0)

    bo_saturated = bo_fn(rs, t_f, api, gas_sg)
    bo_pb = float(np.asarray(bo_fn(rsb, t_f, api, gas_sg)))
    bo = np.where(p <= pb, bo_saturated, bo_pb * np.exp(-co_per_psi * (p - pb)))

    mu_dead = dead_mu_fn(t_f, api)
    mu_saturated = sat_mu_fn(mu_dead, rs)
    mu_pb = float(np.asarray(sat_mu_fn(mu_dead, rsb)))
    mu_undersat = vasquez_beggs_undersat_viscosity(mu_pb, np.maximum(p, pb), pb)
    mu_o = np.where(p <= pb, mu_saturated, mu_undersat)

    gas_table, gas_metadata = calculate_gas_table(p, t_f, gas_sg, gas_options)
    bg_rb_scf = gas_table["Bg_ft3_scf"].to_numpy() / FT3_PER_BBL
    bt = bo + np.maximum(rsb - rs, 0.0) * bg_rb_scf

    oil_sg = api_to_sg_oil(api)
    rho_o_lbft3 = (62.4 * oil_sg + 0.0136 * rs * gas_sg) / bo
    phase_region = np.where(p <= pb, "Saturated oil", "Undersaturated oil")

    table = gas_table.copy()
    table.insert(1, "Rs_scfstb", rs)
    table.insert(2, "Pb_psia", np.full_like(p, pb))
    table.insert(3, "Bo_rb_stb", bo)
    table.insert(4, "Bt_rb_stb", bt)
    table.insert(5, "mu_o_cp", mu_o)
    table.insert(6, "rho_o_lbft3", rho_o_lbft3)
    table["phase_region"] = phase_region

    metadata = {
        **gas_metadata,
        "Pb_psia": pb,
        "Rsb_scfstb": rsb,
        "Bo_at_Pb": bo_pb,
        "mu_dead_cp": mu_dead,
        "mu_at_Pb_cp": mu_pb,
        "oil_sg": oil_sg,
    }
    return table, metadata


def calculate_dead_oil_table(
    pressures_psia: Iterable[float],
    t_f: float,
    api: float,
    reference_pressure_psia: float,
    reference_bo: float,
    co_per_psi: float,
    dead_viscosity_correlation: str,
    manual_reference_mu_cp: float | None,
    viscosity_pressure_coefficient_per_psi: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    p = np.asarray(list(pressures_psia), dtype=float)
    if np.any(p <= 0.0) or reference_pressure_psia <= 0.0:
        raise ValueError("Absolute pressures must be greater than zero")
    if reference_bo <= 0.0:
        raise ValueError("Reference Bo must be positive")
    if co_per_psi < 0.0 or viscosity_pressure_coefficient_per_psi < 0.0:
        raise ValueError("Pressure coefficients cannot be negative")

    mu_correlation = DEAD_VISCOSITY_FUNCTIONS[dead_viscosity_correlation](t_f, api)
    mu_ref = float(manual_reference_mu_cp) if manual_reference_mu_cp is not None else mu_correlation
    bo = reference_bo * np.exp(-co_per_psi * (p - reference_pressure_psia))
    mu_o = mu_ref * np.exp(viscosity_pressure_coefficient_per_psi * (p - reference_pressure_psia))
    oil_sg = api_to_sg_oil(api)
    rho_o_lbft3 = 62.4 * oil_sg / bo

    table = pd.DataFrame(
        {
            "P_psia": p,
            "Bo_rb_stb": bo,
            "mu_o_cp": mu_o,
            "rho_o_lbft3": rho_o_lbft3,
            "Rs_scfstb": np.zeros_like(p),
            "phase_region": np.full(p.shape, "Single-phase dead oil", dtype=object),
        }
    )
    metadata = {
        "reference_pressure_psia": reference_pressure_psia,
        "reference_bo": reference_bo,
        "reference_mu_cp": mu_ref,
        "correlated_dead_mu_cp": mu_correlation,
        "oil_sg": oil_sg,
    }
    return table, metadata


def format_output_table(
    table: pd.DataFrame,
    pressure_unit: str,
    gor_unit: str,
    bg_unit: str,
    viscosity_unit: str,
    density_unit: str,
    compressibility_unit: str,
) -> pd.DataFrame:
    """Return a presentation/export table in the selected output units."""
    result = pd.DataFrame(index=table.index)
    if "P_psia" in table:
        result[f"Pressure [{pressure_unit}]"] = pressure_from_psia(table["P_psia"].to_numpy(), pressure_unit)
    if "Rs_scfstb" in table:
        result[f"Rs [{gor_unit}]"] = gor_from_scfstb(table["Rs_scfstb"].to_numpy(), gor_unit)
    if "Pb_psia" in table:
        result[f"Pb [{pressure_unit}]"] = pressure_from_psia(table["Pb_psia"].to_numpy(), pressure_unit)
    if "Bo_rb_stb" in table:
        result["Bo [rb/STB]"] = table["Bo_rb_stb"].to_numpy()
    if "Bt_rb_stb" in table:
        result["Bt [rb/STB]"] = table["Bt_rb_stb"].to_numpy()
    if "Bg_ft3_scf" in table:
        result[f"Bg [{bg_unit}]"] = bg_from_ft3_scf(table["Bg_ft3_scf"].to_numpy(), bg_unit)
    if "mu_o_cp" in table:
        result[f"Oil viscosity [{viscosity_unit}]"] = viscosity_from_cp(
            table["mu_o_cp"].to_numpy(), viscosity_unit
        )
    if "mu_g_cp" in table:
        result[f"Gas viscosity [{viscosity_unit}]"] = viscosity_from_cp(
            table["mu_g_cp"].to_numpy(), viscosity_unit
        )
    if "rho_o_lbft3" in table:
        result[f"Oil density [{density_unit}]"] = density_from_lbft3(
            table["rho_o_lbft3"].to_numpy(), density_unit
        )
    if "rho_g_lbft3" in table:
        result[f"Gas density [{density_unit}]"] = density_from_lbft3(
            table["rho_g_lbft3"].to_numpy(), density_unit
        )
    if "cg_per_psi" in table:
        result[f"Gas compressibility [{compressibility_unit}]"] = compressibility_from_per_psi(
            table["cg_per_psi"].to_numpy(), compressibility_unit
        )
    for column in ("z", "Ppr", "Tpr", "phase_region"):
        if column in table:
            result[column] = table[column].to_numpy()
    return result


def correlation_range_warnings(
    fluid: str,
    table: pd.DataFrame,
    t_f: float,
    api: float | None,
    gas_sg: float | None,
) -> list[str]:
    warnings: list[str] = []
    if api is not None and not 15.0 <= api <= 60.0:
        warnings.append("Oil gravity is outside the broad calibration envelope of the selected legacy oil correlations.")
    if not 70.0 <= t_f <= 300.0:
        warnings.append("Temperature is outside the usual calibration range of several selected empirical correlations.")
    if gas_sg is not None and not 0.55 <= gas_sg <= 1.5:
        warnings.append("Gas gravity is outside the usual range for the selected natural-gas correlations.")
    if "Ppr" in table and (table["Ppr"].min() < 0.2 or table["Ppr"].max() > 30.0):
        warnings.append("Part of the pressure range falls outside the usual DAK reduced-pressure range (0.2–30).")
    if "Tpr" in table and (table["Tpr"].min() < 1.0 or table["Tpr"].max() > 3.0):
        warnings.append("Reduced temperature falls outside the usual DAK range (1.0–3.0).")
    if fluid == "Wet Gas" and "dew_point_psia" in table.attrs:
        dew = float(table.attrs["dew_point_psia"])
        if table["P_psia"].min() < dew:
            warnings.append(
                "The pressure grid extends below dew point. Retrograde liquid dropout is not modeled; use a compositional EOS for reserves or development decisions."
            )
    return warnings
