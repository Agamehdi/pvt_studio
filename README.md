# PVT Studio

A Streamlit application for generating correlation-based PVT curves and export tables for:

- Black oil
- Dead oil
- Dry gas
- Wet gas / gas-condensate screening

## Features

- Independent unit selectors for pressure, temperature, GOR, compressibility, viscosity, density and gas FVF
- Automatic value conversion when an input unit or gravity basis is changed, preserving the same physical quantity
- Manual `Run PVT` workflow: edit any number of inputs first, then calculate once on demand
- Oil input as API gravity, specific gravity, kg/m³ or lb/ft³
- Gas input as specific gravity or molecular weight
- Selectable oil correlations: Standing, Vasquez–Beggs, Beggs–Robinson, Beal and Chew–Connally
- Selectable gas correlations: Sutton or Standing pseudo-critical properties; Dranchuk–Abou-Kassem or Papay z-factor
- Wichert–Aziz sour-gas correction
- Interactive Plotly charts and engineering QA messages
- UTF-8 CSV, tab-delimited seven-bit ASCII and Eclipse/OPM Flow `.INC` downloads
- Eclipse keywords by fluid: `PVTO + PVDG`, `PVDO`, `PVDG` or `PVTG`
- Selectable FIELD or METRIC Eclipse deck units
- Optional z-factor correlation comparison

## Install and run

### Fastest option on Windows

1. **Extract the complete ZIP first. Do not run the launcher inside the ZIP preview.**
2. Open the extracted `pvt_studio` folder.
3. Double-click `START_PVT_STUDIO.cmd`.

The launcher finds Python from the Windows `py` launcher, PATH, Anaconda or Miniconda; creates a local environment; installs missing packages; starts Streamlit; checks its health endpoint; and opens the browser. Startup logs are written to the `logs` folder only if/when the launcher is used.

`run_windows.bat` remains as a compatibility shortcut and calls the same launcher.

Inside the app, set the inputs and correlations and then click **Run PVT** in the left sidebar. Changing a unit automatically converts the current numerical value; it does not change the underlying physical input.

If Windows does not open the browser automatically, use the local URL printed by the launcher, normally `http://127.0.0.1:8501`.

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

## Export notes

- CSV repeats fluid and correlation metadata on every row and opens cleanly in Excel.
- ASCII is seven-bit, tab-delimited text with metadata comment lines and units in the headers.
- Eclipse `.INC` output contains PROPS-section PVT keywords for one PVT region. The parent deck must use the selected FIELD or METRIC unit system and provide sufficient `TABDIMS` capacity; suggested limits are written in the file comments.
- Black-oil PVTO needs at least one calculated pressure below bubble point. If the app warns that PVTO cannot be created, lower the minimum pressure and click **Run PVT** again.
- Wet-gas `PVTG` is a constant-Rv screening table because the correlation workflow does not calculate retrograde liquid dropout or Rv-dependent properties. Calibrate/replace it with CVD/CCE laboratory data or a tuned EOS before simulation work.

## References

- Standing, M. B. (1947), *A Pressure-Volume-Temperature Correlation for Mixtures of California Oils and Gases*.
- Vasquez, M. and Beggs, H. D. (1980), *Correlations for Fluid Physical Property Prediction*, SPE 6719-PA.
- Beggs, H. D. and Robinson, J. R. (1975), *Estimating the Viscosity of Crude Oil Systems*, SPE 5434-PA.
- Dranchuk, P. M. and Abou-Kassem, J. H. (1975), *Calculation of Z Factors for Natural Gases Using Equations of State*, DOI: 10.2118/75-03-03.
- Lee, A. L., Gonzalez, M. H. and Eakin, B. E. (1966), *The Viscosity of Natural Gases*, SPE 1340-PA.
- OPM Flow Reference Manual, PVT keyword definitions: https://opm-project.org/wp-content/uploads/2023/06/OPM_Flow_Reference_Manual_2023-04_Rev-0_Reduced.pdf
- OPM common Eclipse deck parser and keyword definitions: https://github.com/OPM/opm-common
