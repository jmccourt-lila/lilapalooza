# Materials Project Lilapalooza

A comprehensive tool for analyzing magnetic moments, metallic nature, and thermodynamic stability of materials using the Materials Project database.

## Features

- 🔍 **Search by Elements**: Find compounds containing specific elements
- 🧲 **Magnetic Moment Analysis**: Compare magnetic moments per atom for compounds and related structures
- ⚡ **Metallic Nature Analysis**: Determine if compounds are metallic or non-metallic based on element types and space group
- 🌡️ **Thermodynamic Stability Analysis**: Determine if compounds are stable, near-stable, metastable, or unstable
- 🔬 **Comprehensive Analysis**: Combined analysis of magnetic, metallic, and stability properties

## Installation

1. Install dependencies:
```bash
pip install -e .
```

Or install manually:
```bash
pip install mp-api numpy matplotlib streamlit ipykernel
```

2. Get a Materials Project API key:
   - Sign up at https://materialsproject.org
   - Get your API key from https://materialsproject.org/api

## Usage

### Command Line / Python Script

Run the main script:
```bash
python main.py
```

Or use in Jupyter notebooks - the file contains cell markers (`# %%`) for interactive use.

### Streamlit Web Application

Run the Streamlit GUI:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

#### Streamlit App Features:

1. **Search by Elements Tab**: 
   - Enter comma-separated elements (e.g., "Fe, Ni, O")
   - Filter by metallic nature, energy above hull, etc.
   - View results in an interactive table

2. **Magnetic Moments Tab**:
   - Enter a compound formula (e.g., "LuMnBRu")
   - Find related structures and compare magnetic moments
   - View statistics (average, max, min)
   - Exact matches are shown first

3. **Metallic Nature Tab**:
   - Enter a compound formula
   - Classifies elements as transition metals, alkali/alkaline earth, or lanthanides
   - Ranks by space group (Fm-3m > Im-3m > P6_3/mmc > others)
   - View band gap statistics

4. **Thermodynamic Stability Tab**:
   - Enter a compound formula
   - Analyze thermodynamic stability based on energy above hull
   - View stability classification (Stable/Near-stable/Metastable/Unstable)
   - See formation energy and decomposition information

5. **Compound Analysis Tab**:
   - Complete analysis combining magnetic, metallic, and stability properties
   - View all results in one place

## API Key Setup

Set your API key as an environment variable:
```bash
export MP_API_KEY="your_api_key_here"
```

Or enter it directly in the Streamlit app sidebar.

## Examples

### Search for compounds with specific elements:
```python
from main import list_compounds_by_elements

docs = list_compounds_by_elements(
    elements=["Fe", "Ni"],
    api_key=API_KEY,
    metallic_only=True,
    max_results=20
)
```

### Compare magnetic moments:
```python
from main import compare_magnetic_moments_for_compound

mag_data = compare_magnetic_moments_for_compound(
    target_formula="LuMnBRu",
    api_key=API_KEY
)
```

### Compare metallic nature:
```python
from main import compare_metallic_nature_for_compound

metallic_data = compare_metallic_nature_for_compound(
    target_formula="MnTe",
    api_key=API_KEY
)
```

### Analyze thermodynamic stability:
```python
from main import compare_thermodynamic_stability_for_compound

stability_data = compare_thermodynamic_stability_for_compound(
    target_formula="Fe2O3",
    api_key=API_KEY,
    show_formula_units=True
)
```

## Thermodynamic Stability Analysis

### Overview

The thermodynamic stability analysis determines whether a compound is stable, near-stable, metastable, or unstable based on its energy above the convex hull.

### Key Concepts

#### Energy Above Hull (E_above_hull)
- **Definition**: The energy difference between the compound and the lowest energy combination of phases at the same composition (the convex hull)
- **Units**: millielectronvolts (meV)
- **Interpretation**:
  - **0 meV**: Compound is on the convex hull → **Stable**
  - **< 50 meV**: Very close to stability → **Near-stable**
  - **50 - 200 meV**: May be synthesizable but unstable → **Metastable**
  - **> 200 meV**: Likely unstable → **Unstable**

#### Formation Energy Per Atom
- The energy required to form the compound from its constituent elements
- Units: eV/atom
- Lower (more negative) values indicate more stable compounds

#### Stability Classification
1. **Stable (on hull)**: E_above_hull = 0 meV
2. **Near-stable**: 0 < E_above_hull < 50 meV
3. **Metastable**: 50 ≤ E_above_hull < 200 meV
4. **Unstable**: E_above_hull ≥ 200 meV

### Workflow Steps

1. **Input Compound Formula**
   - Enter the chemical formula (e.g., "Fe2O3", "LuMnBRu", "GdN")
   - The system searches for:
     - Exact formula matches
     - Structures with all the same elements
     - Structures with subsets of elements (for comparison)

2. **Data Retrieval**
   The system retrieves from Materials Project:
   - `energy_above_hull`: Energy above convex hull (in eV, displayed as meV)
   - `formation_energy_per_atom`: Formation energy per atom (eV/atom)
   - `is_stable`: Boolean indicating if on convex hull
   - `decomposition`: Decomposition pathway (if available)

3. **Analysis & Ranking**
   Results are sorted by:
   1. **Exact matches first** (if found)
   2. **Stability priority**:
      - Stable (on hull) → highest priority
      - Near-stable → high priority
      - Metastable → medium priority
      - Unstable → low priority
   3. **Energy above hull** (lower is better, for tie-breaking)

4. **Output & Interpretation**
   - Material ID, Formula, Status with visual indicators
   - E_above_hull in meV
   - Formation energy per atom in eV/atom
   - Statistics: counts by category, energy ranges, stable materials list

### Interpretation Guide

**✅ Stable (E_above_hull = 0 meV)**
- Compound is thermodynamically stable
- Should be synthesizable
- On the convex hull of the phase diagram

**⚠️ Near-stable (0 < E_above_hull < 50 meV)**
- Very close to stability
- May be synthesizable under specific conditions
- Small energy barrier to decomposition

**⚡ Metastable (50 ≤ E_above_hull < 200 meV)**
- Not thermodynamically stable
- May be kinetically stable
- Could be synthesizable but will decompose over time
- Useful for applications requiring metastable phases

**❌ Unstable (E_above_hull ≥ 200 meV)**
- Thermodynamically unstable
- Likely to decompose
- May require special synthesis conditions
- Higher energy barrier to formation

### Best Practices

1. **Check Exact Matches First**: If an exact match is found, prioritize those results
2. **Compare Related Structures**: Look at structures with the same elements to understand stability trends
3. **Consider Synthesis Conditions**: Even unstable compounds may be synthesizable under non-equilibrium conditions
4. **Use Formation Energy**: Lower (more negative) formation energy indicates greater stability
5. **Check Multiple Sources**: Compare with experimental data when available

### Limitations

- Results are based on DFT calculations (may differ from experiment)
- Stability is at 0 K (temperature effects not included)
- Pressure effects not considered
- Kinetic stability not directly assessed
- Some compounds may not be in the database

## Metallic Nature Analysis

### Element Classification

The system classifies elements into:
- **Transition Metals**: Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, La, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, etc.
- **Alkali/Alkaline Earth Metals**: Li, Na, K, Rb, Cs, Fr, Be, Mg, Ca, Sr, Ba, Ra
- **Lanthanides**: Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu
- **Other**: All other elements

If the majority of elements in a compound are transition metals, alkali/alkaline earth, or lanthanides, the compound is classified as metallic.

### Space Group Priority

Materials are ranked by space group structure:
1. **Fm-3m** (highest priority, most metallic, cubic)
2. **Im-3m** (second priority)
3. **P6_3/mmc** (third priority)
4. **Others** (lowest priority)

## File Structure

- `main.py`: Core functions and command-line interface
- `streamlit_app.py`: Streamlit web application
- `magnetic_elements_reference.txt`: Reference list of magnetic elements
- `pyproject.toml`: Project dependencies

## Notes

- Magnetic moments are given in Bohr magnetons (μB)
- Band gaps are in electron volts (eV)
- Energy above hull is displayed in millielectronvolts (meV)
- Formation energy is in eV/atom
- The app searches for related structures when exact matches aren't found
- Results are sorted by relevance (exact matches first, then by property values)

## Related Metrics

- **Band Gap**: Electronic properties (affects applications)
- **Magnetic Moment**: Magnetic properties
- **Space Group**: Crystal structure (affects properties)
- **Formation Energy**: Thermodynamic driving force
- **Energy Above Hull**: Stability metric

## References

- Materials Project: https://materialsproject.org
- Energy Above Hull: Energy difference from convex hull
- Formation Energy: Energy of compound formation from elements
