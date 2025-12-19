# Materials Project Lilapalooza

A tool for retrieving and comparing information about magnetic moments, metallic nature, and thermodynamic stability of materials from the Materials Project database.

## Installation

1. Install dependencies using uv:
```bash
uv sync
```

2. Get a Materials Project API key:
   - Sign up at https://materialsproject.org
   - Get your API key from https://materialsproject.org/api

## Usage

### Streamlit Web Application

Launch the Streamlit GUI:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

Enter your Materials Project API key in the sidebar, or set it as an environment variable:
```bash
export MP_API_KEY="your_api_key_here"
```

### App Features

The Streamlit app provides five tabs for retrieving materials data:

1. **🔍 Search by Elements**: 
   - Search for compounds containing specific elements (e.g., "Fe, Ni, O")
   - Filter by metallic nature, energy above hull, number of elements
   - View results in an interactive table

2. **🧲 Magnetic Moments**:
   - Enter a compound formula (e.g., "LuMnBRu")
   - Retrieve magnetic moments per atom from Materials Project
   - Compare with related structures
   - View statistics (average, max, min)

3. **⚡ Metallic Nature**:
   - Enter a compound formula
   - Retrieve band gap and metallic/non-metallic classification
   - View element classifications and space group information

4. **🌡️ Thermodynamic Stability**:
   - Enter a compound formula
   - Retrieve energy above hull and formation energy data
   - View stability classification (Stable/Near-stable/Metastable/Unstable)

5. **🔬 Compound Analysis**:
   - Comprehensive data retrieval combining all properties
   - View magnetic, metallic, and stability information together

### Command Line / Python Script

You can also use the functions programmatically:
```bash
python main.py
```

Or import functions in Python/Jupyter notebooks:
```python
from main import list_compounds_by_elements, compare_magnetic_moments_for_compound

# Search for compounds
docs = list_compounds_by_elements(
    elements=["Fe", "Ni"],
    api_key=API_KEY,
    max_results=20
)

# Compare magnetic moments
mag_data = compare_magnetic_moments_for_compound(
    target_formula="LuMnBRu",
    api_key=API_KEY
)
```

## Notes

- All data is retrieved from the Materials Project database
- Magnetic moments are in Bohr magnetons (μB)
- Energy above hull is displayed in millielectronvolts (meV)
- The app searches for related structures when exact matches aren't found
- Results are sorted by relevance (exact matches first)

## References

- Materials Project: https://materialsproject.org
