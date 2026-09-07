# PVT Studio

A Streamlit application for generating correlation-based PVT curves and CSV tables for:

- Black oil
- Dead oil
- Dry gas
- Wet gas / gas-condensate screening

## Features

- Independent unit selectors for pressure, temperature, GOR, compressibility, viscosity, density and gas FVF
- Oil input as API gravity, specific gravity, kg/m³ or lb/ft³
- Gas input as specific gravity or molecular weight
- Selectable oil correlations: Standing, Vasquez–Beggs, Beggs–Robinson, Beal and Chew–Connally
- Selectable gas correlations: Sutton or Standing pseudo-critical properties; Dranchuk–Abou-Kassem or Papay z-factor
- Wichert–Aziz sour-gas correction
- Interactive Plotly charts, engineering QA messages and UTF-8 CSV download
- Optional z-factor correlation comparison

## Install and run

### Fastest option on Windows

Double-click `run_windows.bat`. It creates a local environment, installs the required packages and opens the app.

### Manual setup

Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Correlation scope

This is an engineering screening tool. Empirical correlations must be checked against representative laboratory PVT data before they are used for reserves, simulation, facility design or operating decisions. The wet-gas mode calculates gas-phase properties and marks the entered dew point; it does not calculate retrograde condensate dropout below dew point.

The implementation also corrects two issues in the original prototype:

1. Standing oil FVF uses `Rs × sqrt(gas SG / oil SG)`.
2. Gas FVF is calculated in ft³/scf and converted to rb/Mscf using 5.6146 ft³/bbl.

## References

- Standing, M. B. (1947), *A Pressure-Volume-Temperature Correlation for Mixtures of California Oils and Gases*.
- Vasquez, M. and Beggs, H. D. (1980), *Correlations for Fluid Physical Property Prediction*, SPE 6719-PA.
- Beggs, H. D. and Robinson, J. R. (1975), *Estimating the Viscosity of Crude Oil Systems*, SPE 5434-PA.
- Dranchuk, P. M. and Abou-Kassem, J. H. (1975), *Calculation of Z Factors for Natural Gases Using Equations of State*, DOI: 10.2118/75-03-03.
- Lee, A. L., Gonzalez, M. H. and Eakin, B. E. (1966), *The Viscosity of Natural Gases*, SPE 1340-PA.
