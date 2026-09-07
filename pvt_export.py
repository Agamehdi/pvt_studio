"""Text and Eclipse-style exports for PVT Studio.

The calculation engine stores values in oilfield units.  This module converts
those values into either FIELD or METRIC Eclipse deck units and writes one PVT
region per include file.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
import unicodedata

import numpy as np
import pandas as pd

from pvt_engine import FT3_PER_BBL, PSI_PER_BAR, SCFSTB_TO_SM3SM3


ECLIPSE_UNIT_SYSTEMS = ("FIELD", "METRIC")


def _ascii_text(value: object) -> str:
    """Return readable seven-bit text for ASCII and deck comments."""
    text = str(value).translate(
        str.maketrans(
            {
                "°": "deg",
                "³": "3",
                "²": "2",
                "¹": "1",
                "μ": "mu",
                "·": ".",
                "–": "-",
                "—": "-",
                "−": "-",
                "₂": "2",
            }
        )
    )
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _safe_comment(value: object) -> str:
    return re.sub(r"\s+", " ", _ascii_text(value)).replace("--", "-").strip()


def build_ascii_export(table: pd.DataFrame, metadata: Mapping[str, object]) -> bytes:
    """Build a self-describing, tab-delimited, seven-bit ASCII table."""
    ascii_table = table.copy()
    ascii_table.columns = [_ascii_text(column).replace(" ", "_") for column in table.columns]
    for column in ascii_table.select_dtypes(include=["object", "string"]).columns:
        ascii_table[column] = ascii_table[column].map(_ascii_text)

    lines = ["# PVT Studio ASCII export"]
    lines.extend(f"# {_ascii_text(key)}: {_ascii_text(value)}" for key, value in metadata.items())
    lines.append("# Columns are tab-delimited; units are included in column names.")
    lines.append(ascii_table.to_csv(index=False, sep="\t", float_format="%.10g", lineterminator="\n").rstrip())
    return ("\n".join(lines) + "\n").encode("ascii")


def _sorted_finite_table(table: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"Missing columns required for Eclipse export: {', '.join(missing)}")
    result = table.loc[:, columns].copy()
    numeric = result.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Eclipse export cannot contain NaN or infinite values")
    result = result.sort_values("P_psia").drop_duplicates("P_psia", keep="last").reset_index(drop=True)
    if len(result) < 2:
        raise ValueError("Eclipse PVT tables require at least two distinct pressure rows")
    if (result["P_psia"] <= 0.0).any():
        raise ValueError("Eclipse PVT pressure values must be positive and absolute")
    return result


def _number(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("Eclipse export cannot contain a non-finite value")
    return f"{float(value):.10g}"


def _deck_row(values: tuple[float | None, ...], terminate: bool = False) -> str:
    cells = ["" if value is None else _number(value) for value in values]
    line = "  " + "  ".join(f"{cell:>15}" for cell in cells).rstrip()
    return f"{line}  /" if terminate else line


def _unit_values(unit_system: str) -> dict[str, object]:
    unit_system = unit_system.upper()
    if unit_system not in ECLIPSE_UNIT_SYSTEMS:
        raise ValueError(f"Unsupported Eclipse unit system: {unit_system}")
    if unit_system == "FIELD":
        return {
            "pressure": lambda value: np.asarray(value, dtype=float),
            "rs": lambda value: np.asarray(value, dtype=float) / 1000.0,
            "rv": lambda value: np.asarray(value, dtype=float) / 1000.0,
            "bg": lambda value: np.asarray(value, dtype=float) * 1000.0 / FT3_PER_BBL,
            "pressure_label": "PSIA",
            "rs_label": "MSCF/STB",
            "rv_label": "STB/MSCF",
            "bo_label": "RB/STB",
            "bg_label": "RB/MSCF",
        }
    return {
        "pressure": lambda value: np.asarray(value, dtype=float) / PSI_PER_BAR,
        "rs": lambda value: np.asarray(value, dtype=float) * SCFSTB_TO_SM3SM3,
        "rv": lambda value: np.asarray(value, dtype=float) * FT3_PER_BBL / 1_000_000.0,
        "bg": lambda value: np.asarray(value, dtype=float),
        "pressure_label": "BARSA",
        "rs_label": "SM3/SM3",
        "rv_label": "SM3/SM3",
        "bo_label": "RM3/SM3",
        "bg_label": "RM3/SM3",
    }


def _header(
    fluid: str,
    unit_system: str,
    correlations: Mapping[str, object] | None,
) -> list[str]:
    lines = [
        "-- PVT Studio Eclipse-compatible include",
        f"-- Fluid system: {_safe_comment(fluid)}",
        f"-- Deck unit system: {unit_system}",
        "-- Insert the generated PVT keyword(s) in the PROPS section.",
        "-- The parent deck must declare the same unit system and suitable TABDIMS limits.",
    ]
    if correlations:
        lines.append("-- Correlations:")
        for name, selection in correlations.items():
            lines.append(f"--   {_safe_comment(name)} = {_safe_comment(selection)}")
    lines.append("")
    return lines


def _pvdg_lines(table: pd.DataFrame, units: Mapping[str, object]) -> list[str]:
    gas = _sorted_finite_table(table, ("P_psia", "Bg_ft3_scf", "mu_g_cp"))
    pressure = units["pressure"](gas["P_psia"].to_numpy())
    bg = units["bg"](gas["Bg_ft3_scf"].to_numpy())
    viscosity = gas["mu_g_cp"].to_numpy()
    lines = [
        "PVDG",
        "--       PRESSURE               BG          GAS_MU",
        f"--       {units['pressure_label']:<15} {units['bg_label']:<15} CP",
    ]
    for index, values in enumerate(zip(pressure, bg, viscosity)):
        lines.append(_deck_row(tuple(float(value) for value in values), terminate=index == len(gas) - 1))
    return lines


def _pvdo_lines(table: pd.DataFrame, units: Mapping[str, object]) -> list[str]:
    oil = _sorted_finite_table(table, ("P_psia", "Bo_rb_stb", "mu_o_cp"))
    pressure = units["pressure"](oil["P_psia"].to_numpy())
    lines = [
        "PVDO",
        "--       PRESSURE               BO          OIL_MU",
        f"--       {units['pressure_label']:<15} {units['bo_label']:<15} CP",
    ]
    rows = zip(pressure, oil["Bo_rb_stb"].to_numpy(), oil["mu_o_cp"].to_numpy())
    for index, values in enumerate(rows):
        lines.append(_deck_row(tuple(float(value) for value in values), terminate=index == len(oil) - 1))
    return lines


def _pvto_lines(
    table: pd.DataFrame,
    metadata: Mapping[str, object],
    units: Mapping[str, object],
) -> tuple[list[str], int]:
    oil = _sorted_finite_table(table, ("P_psia", "Rs_scfstb", "Bo_rb_stb", "mu_o_cp"))
    required_metadata = ("Pb_psia", "Rsb_scfstb", "Bo_at_Pb", "mu_at_Pb_cp")
    missing = [name for name in required_metadata if name not in metadata]
    if missing:
        raise ValueError(f"Missing black-oil metadata for PVTO: {', '.join(missing)}")

    pb = float(metadata["Pb_psia"])
    rsb = float(metadata["Rsb_scfstb"])
    bo_pb = float(metadata["Bo_at_Pb"])
    mu_pb = float(metadata["mu_at_Pb_cp"])
    tolerance = max(1.0e-7 * pb, 1.0e-7)
    saturated = oil[oil["P_psia"] < pb - tolerance].copy()
    if saturated.empty:
        raise ValueError(
            "PVTO export needs at least one pressure point below bubble point; lower the minimum pressure and run PVT again"
        )

    # Numerical clipping can create repeated Rs values close to Pb.  Eclipse
    # requires the outer Rs column to increase, so retain one row per Rs value.
    saturated = saturated.sort_values(["Rs_scfstb", "P_psia"]).drop_duplicates("Rs_scfstb", keep="last")
    saturated = saturated[saturated["Rs_scfstb"] < rsb - max(1.0e-9 * max(rsb, 1.0), 1.0e-10)]
    if saturated.empty:
        raise ValueError("PVTO export could not form two distinct saturated Rs rows")

    pressure_convert = units["pressure"]
    rs_convert = units["rs"]
    lines = [
        "PVTO",
        "--            RS         PRESSURE               BO          OIL_MU",
        f"--       {units['rs_label']:<15} {units['pressure_label']:<15} {units['bo_label']:<15} CP",
    ]
    for row in saturated.itertuples(index=False):
        lines.append(
            _deck_row(
                (
                    float(rs_convert(row.Rs_scfstb)),
                    float(pressure_convert(row.P_psia)),
                    float(row.Bo_rb_stb),
                    float(row.mu_o_cp),
                ),
                terminate=True,
            )
        )

    undersaturated = oil[oil["P_psia"] > pb + tolerance]
    lines.append(
        _deck_row(
            (
                float(rs_convert(rsb)),
                float(pressure_convert(pb)),
                bo_pb,
                mu_pb,
            ),
            terminate=undersaturated.empty,
        )
    )
    for index, row in enumerate(undersaturated.itertuples(index=False)):
        lines.append(
            _deck_row(
                (
                    None,
                    float(pressure_convert(row.P_psia)),
                    float(row.Bo_rb_stb),
                    float(row.mu_o_cp),
                ),
                terminate=index == len(undersaturated) - 1,
            )
        )
    lines.append("/")
    return lines, len(saturated) + 1


def _pvtg_lines(
    table: pd.DataFrame,
    units: Mapping[str, object],
    cgr_stb_mmscf: float,
) -> list[str]:
    gas = _sorted_finite_table(table, ("P_psia", "Bg_ft3_scf", "mu_g_cp"))
    if not np.isfinite(cgr_stb_mmscf) or cgr_stb_mmscf < 0.0:
        raise ValueError("Wet-gas CGR must be finite and non-negative")
    pressure = units["pressure"](gas["P_psia"].to_numpy())
    rv = float(units["rv"](cgr_stb_mmscf))
    bg = units["bg"](gas["Bg_ft3_scf"].to_numpy())
    viscosity = gas["mu_g_cp"].to_numpy()
    lines = [
        "-- WARNING: screening PVTG with constant Rv.",
        "-- It does not model retrograde condensate dropout or property dependence on Rv.",
        "-- Replace/calibrate with laboratory CVD/CCE data or a tuned EOS for simulation studies.",
        "PVTG",
        "--       PRESSURE               RV               BG          GAS_MU",
        f"--       {units['pressure_label']:<15} {units['rv_label']:<15} {units['bg_label']:<15} CP",
    ]
    for p_value, bg_value, mu_value in zip(pressure, bg, viscosity):
        lines.append(_deck_row((float(p_value), rv, float(bg_value), float(mu_value)), terminate=True))
    lines.append("/")
    return lines


def build_eclipse_include(
    fluid: str,
    table: pd.DataFrame,
    metadata: Mapping[str, object],
    correlations: Mapping[str, object] | None = None,
    unit_system: str = "FIELD",
    cgr_stb_mmscf: float | None = None,
    dew_point_psia: float | None = None,
) -> bytes:
    """Build one-region Eclipse/OPM Flow PVT include data as ASCII bytes.

    The include intentionally contains only PROPS-section PVT keywords.  The
    parent deck remains responsible for RUNSPEC phase flags and TABDIMS.
    """
    unit_system = unit_system.upper()
    units = _unit_values(unit_system)
    lines = _header(fluid, unit_system, correlations)
    row_count = len(table.drop_duplicates("P_psia")) if "P_psia" in table else len(table)

    if fluid == "Black Oil":
        pvto, rs_count = _pvto_lines(table, metadata, units)
        lines.insert(
            5,
            f"-- Suggested TABDIMS capacity: NPPVT >= {max(row_count, 2)}, NRPVT >= {max(rs_count, 2)}.",
        )
        lines.extend(pvto)
        lines.append("")
        lines.extend(_pvdg_lines(table, units))
    elif fluid == "Dead Oil":
        lines.insert(5, f"-- Suggested TABDIMS capacity: NPPVT >= {max(row_count, 2)}.")
        lines.extend(_pvdo_lines(table, units))
    elif fluid == "Dry Gas":
        lines.insert(5, f"-- Suggested TABDIMS capacity: NPPVT >= {max(row_count, 2)}.")
        lines.extend(_pvdg_lines(table, units))
    elif fluid == "Wet Gas":
        if cgr_stb_mmscf is None:
            raise ValueError("Wet-gas PVTG export requires a condensate-gas ratio")
        lines.insert(5, f"-- Suggested TABDIMS capacity: NPPVT >= {max(row_count, 2)}, NRPVT >= 1.")
        if dew_point_psia is not None:
            dew_value = float(units["pressure"](dew_point_psia))
            lines.insert(6, f"-- User-entered dew point: {_number(dew_value)} {units['pressure_label']}.")
        lines.extend(_pvtg_lines(table, units, cgr_stb_mmscf))
    else:
        raise ValueError(f"Unsupported fluid system for Eclipse export: {fluid}")

    text = "\n".join(lines).rstrip() + "\n"
    return text.encode("ascii")

