import streamlit as st
from typing import List, Optional
from mp_api.client import MPRester
import os
import re
import sys
from io import StringIO
from contextlib import contextmanager

try:
    from pymatgen.core import Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

@contextmanager
def suppress_stdout():
    """Context manager to suppress print statements"""
    with StringIO() as buf:
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Import functions from main.py
sys.path.insert(0, os.path.dirname(__file__))
from main import (
    list_compounds_by_elements,
    get_magnetization_data,
    get_metallic_nature_data,
    get_thermodynamic_stability_data,
    search_related_structures,
    parse_formula_to_elements,
    normalize_formula,
    get_magnetization_per_atom,
    get_total_magnetization,
    filter_metallic,
    get_n_atoms_in_unit_cell,
    get_n_atoms_per_formula_unit
)

# Page configuration
st.set_page_config(
    page_title="Materials Project Lilapalooza",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🧲 Materials Project Lilapalooza</h1>', unsafe_allow_html=True)
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "Materials Project API Key",
            value=os.environ.get("MP_API_KEY", ""),
            type="password",
            key="api_key_input",
            help="Enter your Materials Project API key. Get one at https://materialsproject.org/api"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your API key to use the application")
            st.stop()
        
        st.divider()
        st.header("📚 About")
        st.markdown("""
        This app helps you:
        - Search for compounds by elements
        - Compare magnetic moments
        - Analyze metallic nature
        - Find related structures
        """)
        
        st.divider()
        st.markdown("**Made with:**")
        st.markdown("- Materials Project API")
        st.markdown("- Streamlit")
        st.markdown("- pymatgen")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Search by Elements",
        "🧲 Magnetic Moments",
        "⚡ Metallic Nature",
        "🌡️ Thermodynamic Stability",
        "🔬 Compound Analysis"
    ])
    
    with tab1:
        search_by_elements_tab(api_key)
    
    with tab2:
        magnetic_moments_tab(api_key)
    
    with tab3:
        metallic_nature_tab(api_key)
    
    with tab4:
        thermodynamic_stability_tab(api_key)
    
    with tab5:
        compound_analysis_tab(api_key)


def search_by_elements_tab(api_key: str):
    st.markdown('<h2 class="sub-header">Search Compounds by Elements</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        elements_input = st.text_input(
            "Elements (comma-separated)",
            value="Fe, Ni",
            key="search_elements_input",
            help="Enter element symbols separated by commas (e.g., Fe, Ni, O)"
        )
        elements = [e.strip() for e in elements_input.split(",") if e.strip()]
    
    with col2:
        num_elements = st.number_input(
            "Number of Elements",
            min_value=1,
            max_value=10,
            value=None,
            key="search_num_elements",
            help="Leave as 'None' to search for any number of elements"
        )
        if num_elements == 0:
            num_elements = None
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        energy_above_hull_max = st.number_input(
            "Max Energy Above Hull (eV)",
            min_value=0.0,
            max_value=5.0,
            value=0.05,
            step=0.01,
            key="search_energy_hull"
        )
    
    with col4:
        metallic_filter = st.selectbox(
            "Metallic Filter",
            ["All", "Metallic Only", "Non-metallic Only"],
            key="search_metallic_filter",
            help="Filter by metallic nature"
        )
        metallic_only = None if metallic_filter == "All" else (metallic_filter == "Metallic Only")
    
    with col5:
        theoretical_filter = st.selectbox(
            "Theoretical/Experimental",
            ["Both", "Experimental Only", "Theoretical Only"],
            key="search_theoretical_filter"
        )
        theoretical = None if theoretical_filter == "Both" else (theoretical_filter == "Experimental Only")
    
    max_results = st.number_input(
        "Max Results",
        min_value=1,
        max_value=1000,
        value=50,
        key="search_max_results",
        help="Maximum number of results to return"
    )
    
    if st.button("🔍 Search", type="primary", use_container_width=True, key="search_button"):
        if not elements:
            st.error("Please enter at least one element")
        else:
            with st.spinner(f"Searching for compounds containing {', '.join(elements)}..."):
                try:
                    docs = list_compounds_by_elements(
                        elements=elements,
                        api_key=api_key,
                        num_elements=num_elements,
                        energy_above_hull=(0, energy_above_hull_max),
                        theoretical=theoretical,
                        metallic_only=metallic_only,
                        max_results=max_results
                    )
                    
                    if docs:
                        st.success(f"Found {len(docs)} materials")
                        
                        # Display results
                        results_data = []
                        for doc in docs:
                            mag_per_atom = get_magnetization_per_atom(doc)
                            total_mag = get_total_magnetization(doc)
                            
                            metallic_status = "Unknown"
                            if hasattr(doc, 'is_metal') and doc.is_metal is not None:
                                metallic_status = "Metallic" if doc.is_metal else "Non-metallic"
                            elif hasattr(doc, 'band_gap') and doc.band_gap is not None:
                                metallic_status = "Metallic" if doc.band_gap == 0 else f"Non-metallic ({doc.band_gap:.3f} eV)"
                            
                            results_data.append({
                                "Material ID": doc.material_id,
                                "Formula": doc.formula_pretty,
                                "Metallic": metallic_status,
                                "μ/atom (μB)": f"{mag_per_atom:.4f}" if mag_per_atom is not None else "N/A",
                                "Total μ (μB)": f"{total_mag:.4f}" if total_mag is not None else "N/A"
                            })
                        
                        st.dataframe(results_data, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No materials found matching the criteria")
                        
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")


def magnetic_moments_tab(api_key: str):
    st.markdown('<h2 class="sub-header">Magnetic Moment Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Compare magnetic moments</strong> for a compound and related structures.
    Useful when the exact compound might not be in the database.
    </div>
    """, unsafe_allow_html=True)
    
    target_formula = st.text_input(
        "Target Compound Formula",
        value="LuMnBRu",
        key="mag_target_formula",
        help="Enter a chemical formula (e.g., LuMnBRu, Fe2O3, GdN)"
    )
    
    show_formula_units = st.checkbox("Show Formula Unit Information", value=True, key="mag_show_formula_units")
    
    if st.button("🔍 Analyze Magnetic Moments", type="primary", use_container_width=True, key="mag_analyze_button"):
        if not target_formula:
            st.error("Please enter a compound formula")
        else:
            with st.spinner(f"Searching for {target_formula} and related structures..."):
                try:
                    # Use search_related_structures and get_magnetization_data directly
                    # Suppress print statements for cleaner UI
                    with suppress_stdout():
                        docs = search_related_structures(
                            target_formula=target_formula,
                            api_key=api_key,
                            search_exact=True,
                            search_all_elements=True,
                            search_subsets=True,
                            energy_above_hull=(0, 1.0),
                            theoretical=None,
                            max_results_per_search=30
                        )
                    
                    if not docs:
                        st.warning("No related structures found")
                        mag_data = []
                    else:
                        mag_data = get_magnetization_data(docs)
                        # Normalize target formula for comparison (element order doesn't matter)
                        target_formula_norm = normalize_formula(target_formula)
                        # Sort by: 1) exact match first, 2) then by magnetization per atom (descending)
                        mag_data = sorted(
                            mag_data,
                            key=lambda x: (
                                0 if normalize_formula(x['formula']) == target_formula_norm else 1,  # Exact matches first
                                -(x['magnetization_per_atom'] if x['magnetization_per_atom'] is not None else -999)  # Then by magnetization (descending)
                            )
                        )
                    
                    if mag_data:
                        # Check for exact matches
                        exact_matches = [d for d in mag_data if normalize_formula(d['formula']) == target_formula_norm]
                        
                        if exact_matches:
                            st.success(f"✓ Found {len(exact_matches)} exact match(es) for {target_formula} (shown first)")
                        else:
                            st.info(f"Found {len(mag_data)} related structures (no exact match for {target_formula})")
                        
                        # Display results in a table
                        display_data = []
                        for d in mag_data:
                            is_exact = normalize_formula(d['formula']) == target_formula_norm
                            formula_display = f"⭐ {d['formula']}" if is_exact else d['formula']
                            display_data.append({
                                "Material ID": d['material_id'],
                                "Formula": formula_display,
                                "μ/atom (μB)": f"{d['magnetization_per_atom']:.4f}" if d['magnetization_per_atom'] is not None else "N/A",
                                "Total μ (μB)": f"{d['total_magnetization']:.4f}" if d['total_magnetization'] is not None else "N/A",
                                "N (unit cell)": str(d['n_atoms']) if d['n_atoms'] is not None else "N/A",
                                "N (formula)": str(d['n_atoms_per_formula']) if d.get('n_atoms_per_formula') is not None else "N/A"
                            })
                        
                        st.dataframe(display_data, use_container_width=True, hide_index=True)
                        
                        # Statistics
                        magnetic_materials = [d for d in mag_data if d['magnetization_per_atom'] is not None and d['magnetization_per_atom'] > 0.01]
                        if magnetic_materials:
                            moments = [d['magnetization_per_atom'] for d in magnetic_materials]
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Magnetic Materials", len(magnetic_materials))
                            with col2:
                                st.metric("Average μ/atom", f"{sum(moments)/len(moments):.4f} μB")
                            with col3:
                                st.metric("Maximum μ/atom", f"{max(moments):.4f} μB")
                            with col4:
                                st.metric("Minimum μ/atom", f"{min(moments):.4f} μB")
                    else:
                        st.warning("No related structures found")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")


def metallic_nature_tab(api_key: str):
    st.markdown('<h2 class="sub-header">Metallic Nature Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Compare metallic nature</strong> for a compound and related structures.
    Materials are ranked: metallic first, then non-metallic by band gap.
    </div>
    """, unsafe_allow_html=True)
    
    target_formula = st.text_input(
        "Target Compound Formula",
        value="MnTe",
        key="metallic_target_formula",
        help="Enter a chemical formula (e.g., MnTe, Fe2O3, CuAg)"
    )
    
    show_formula_units = st.checkbox("Show Formula Unit Information", value=True, key="metallic_show_formula_units")
    
    if st.button("🔍 Analyze Metallic Nature", type="primary", use_container_width=True, key="metallic_analyze_button"):
        if not target_formula:
            st.error("Please enter a compound formula")
        else:
            with st.spinner(f"Searching for {target_formula} and related structures..."):
                try:
                    # Use search_related_structures and get_metallic_nature_data directly
                    with suppress_stdout():
                        docs = search_related_structures(
                            target_formula=target_formula,
                            api_key=api_key,
                            search_exact=True,
                            search_all_elements=True,
                            search_subsets=True,
                            energy_above_hull=(0, 1.0),
                            theoretical=None,
                            max_results_per_search=30
                        )
                    
                    if not docs:
                        st.warning("No related structures found")
                        metallic_data = []
                    else:
                        metallic_data = get_metallic_nature_data(docs)
                        # Normalize target formula for comparison (element order doesn't matter)
                        target_formula_norm = normalize_formula(target_formula)
                        # Sort by: 1) exact match first, 2) then by space group priority, 3) then by metallic nature
                        def sort_key(x):
                            is_exact_match = normalize_formula(x['formula']) == target_formula_norm
                            exact_priority = 0 if is_exact_match else 1
                            
                            # Space group priority (lower = more metallic/higher priority)
                            space_group_priority = x.get('space_group_priority', 999)
                            
                            # Metallic nature priority
                            if x['is_metallic'] is True:
                                metallic_priority = 0
                                bg_value = 0.0
                            elif x['is_metallic'] is False:
                                metallic_priority = 1
                                bg_value = x['band_gap'] if x['band_gap'] is not None else 999.0
                            elif x['band_gap'] is not None:
                                metallic_priority = 1 if x['band_gap'] > 0 else 0
                                bg_value = x['band_gap'] if x['band_gap'] > 0 else 0.0
                            else:
                                metallic_priority = 2
                                bg_value = 999.0
                            
                            return (exact_priority, space_group_priority, metallic_priority, bg_value)
                        metallic_data = sorted(metallic_data, key=sort_key)
                    
                    if metallic_data:
                        # Check for exact matches (target_formula_norm already defined above)
                        exact_matches = [d for d in metallic_data if normalize_formula(d['formula']) == target_formula_norm]
                        
                        if exact_matches:
                            st.success(f"✓ Found {len(exact_matches)} exact match(es) for {target_formula} (shown first)")
                        else:
                            st.info(f"Found {len(metallic_data)} related structures (no exact match for {target_formula})")
                        
                        # Display results
                        display_data = []
                        for d in metallic_data:
                            is_exact = normalize_formula(d['formula']) == target_formula_norm
                            formula_display = f"⭐ {d['formula']}" if is_exact else d['formula']
                            
                            # Format element types
                            elem_types = d.get('element_types', [])
                            elem_types_str = ", ".join(set(elem_types)) if elem_types else "N/A"
                            
                            display_data.append({
                                "Material ID": d['material_id'],
                                "Formula": formula_display,
                                "Status": d['metallic_status'],
                                "Space Group": d.get('space_group', 'N/A') or 'N/A',
                                "Band Gap (eV)": f"{d['band_gap']:.4f}" if d['band_gap'] is not None else "N/A",
                                "Element Types": elem_types_str,
                                "N (unit cell)": str(d['n_atoms']) if d['n_atoms'] is not None else "N/A",
                                "N (formula)": str(d['n_atoms_per_formula']) if d.get('n_atoms_per_formula') is not None else "N/A"
                            })
                        
                        st.dataframe(display_data, use_container_width=True, hide_index=True)
                        
                        # Statistics
                        metallic_materials = [d for d in metallic_data if d['is_metallic'] is True]
                        non_metallic_materials = [d for d in metallic_data if d['is_metallic'] is False]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Metallic Materials", len(metallic_materials))
                        with col2:
                            st.metric("Non-metallic Materials", len(non_metallic_materials))
                        
                        if non_metallic_materials:
                            band_gaps = [d['band_gap'] for d in non_metallic_materials if d['band_gap'] is not None]
                            if band_gaps:
                                col3, col4, col5 = st.columns(3)
                                with col3:
                                    st.metric("Min Band Gap", f"{min(band_gaps):.4f} eV")
                                with col4:
                                    st.metric("Max Band Gap", f"{max(band_gaps):.4f} eV")
                                with col5:
                                    st.metric("Avg Band Gap", f"{sum(band_gaps)/len(band_gaps):.4f} eV")
                    else:
                        st.warning("No related structures found")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")


def thermodynamic_stability_tab(api_key: str):
    st.markdown('<h2 class="sub-header">Thermodynamic Stability Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Analyze thermodynamic stability</strong> of a compound and related structures.
    <br><br>
    <strong>Key Metrics:</strong>
    <ul>
        <li><strong>Energy Above Hull:</strong> Energy above the convex hull in meV (0 = stable, lower is better)</li>
        <li><strong>Formation Energy:</strong> Energy of formation per atom (eV/atom)</li>
        <li><strong>Stability Status:</strong> Stable (0 meV), Near-stable (&lt;50 meV), Metastable (50-200 meV), or Unstable (&gt;200 meV)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    target_formula = st.text_input(
        "Target Compound Formula",
        value="Fe2O3",
        key="stability_target_formula",
        help="Enter a chemical formula (e.g., Fe2O3, LuMnBRu, GdN)"
    )
    
    show_formula_units = st.checkbox("Show Formula Unit Information", value=True, key="stability_show_formula_units")
    
    if st.button("🔍 Analyze Stability", type="primary", use_container_width=True, key="stability_analyze_button"):
        if not target_formula:
            st.error("Please enter a compound formula")
        else:
            with st.spinner(f"Searching for {target_formula} and analyzing stability..."):
                try:
                    with suppress_stdout():
                        docs = search_related_structures(
                            target_formula=target_formula,
                            api_key=api_key,
                            search_exact=True,
                            search_all_elements=True,
                            search_subsets=True,
                            energy_above_hull=(0, 1.0),
                            theoretical=None,
                            max_results_per_search=30
                        )
                    
                    if not docs:
                        st.warning("No related structures found")
                        stability_data = []
                    else:
                        stability_data = get_thermodynamic_stability_data(docs)
                        # Normalize target formula for comparison (element order doesn't matter)
                        target_formula_norm = normalize_formula(target_formula)
                        # Sort by: 1) exact match first, 2) then by stability
                        def sort_key(x):
                            is_exact_match = normalize_formula(x['formula']) == target_formula_norm
                            exact_priority = 0 if is_exact_match else 1
                            
                            # Stability priority (lower = more stable)
                            if x['is_stable'] is True:
                                stability_priority = 0
                            elif x['energy_above_hull'] is not None:
                                e_hull = x['energy_above_hull']
                                if e_hull == 0:
                                    stability_priority = 0
                                elif e_hull < 0.05:
                                    stability_priority = 1  # Near-stable
                                elif e_hull < 0.2:
                                    stability_priority = 2  # Metastable
                                else:
                                    stability_priority = 3  # Unstable
                            else:
                                stability_priority = 4  # Unknown
                            
                            # Energy above hull value (for tie-breaking)
                            e_hull_value = x['energy_above_hull'] if x['energy_above_hull'] is not None else 999.0
                            
                            return (exact_priority, stability_priority, e_hull_value)
                        stability_data = sorted(stability_data, key=sort_key)
                    
                    if stability_data:
                        # Check for exact matches
                        exact_matches = [d for d in stability_data if normalize_formula(d['formula']) == target_formula_norm]
                        
                        if exact_matches:
                            st.success(f"✓ Found {len(exact_matches)} exact match(es) for {target_formula} (shown first)")
                        else:
                            st.info(f"Found {len(stability_data)} related structures (no exact match for {target_formula})")
                        
                        # Display results
                        display_data = []
                        for d in stability_data:
                            is_exact = normalize_formula(d['formula']) == target_formula_norm
                            formula_display = f"⭐ {d['formula']}" if is_exact else d['formula']
                            
                            # Color-code stability status
                            status = d['stability_status']
                            if "Stable" in status and "on hull" in status:
                                status_display = f"✅ {status}"
                            elif "Near-stable" in status:
                                status_display = f"⚠️ {status}"
                            elif "Metastable" in status:
                                status_display = f"⚡ {status}"
                            elif "Unstable" in status:
                                status_display = f"❌ {status}"
                            else:
                                status_display = status
                            
                            # Convert eV to meV for display
                            e_hull_display = f"{d['energy_above_hull'] * 1000:.1f}" if d['energy_above_hull'] is not None else "N/A"
                            
                            display_data.append({
                                "Material ID": d['material_id'],
                                "Formula": formula_display,
                                "Status": status_display,
                                "E_above_hull (meV)": e_hull_display,
                                "Form E (eV/atom)": f"{d['formation_energy_per_atom']:.4f}" if d['formation_energy_per_atom'] is not None else "N/A",
                                "N (unit cell)": str(d['n_atoms']) if d['n_atoms'] is not None else "N/A",
                                "N (formula)": str(d['n_atoms_per_formula']) if d.get('n_atoms_per_formula') is not None else "N/A"
                            })
                        
                        st.dataframe(display_data, use_container_width=True, hide_index=True)
                        
                        # Statistics
                        stable_materials = [d for d in stability_data if d['is_stable'] is True]
                        near_stable = [d for d in stability_data if d['energy_above_hull'] is not None and 0 < d['energy_above_hull'] < 0.05]
                        metastable = [d for d in stability_data if d['energy_above_hull'] is not None and 0.05 <= d['energy_above_hull'] < 0.2]
                        unstable = [d for d in stability_data if d['energy_above_hull'] is not None and d['energy_above_hull'] >= 0.2]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Stable (on hull)", len(stable_materials))
                        with col2:
                            st.metric("Near-stable (<0.05 eV)", len(near_stable))
                        with col3:
                            st.metric("Metastable (0.05-0.2 eV)", len(metastable))
                        with col4:
                            st.metric("Unstable (>0.2 eV)", len(unstable))
                        
                        if stability_data:
                            e_hull_values = [d['energy_above_hull'] for d in stability_data if d['energy_above_hull'] is not None]
                            if e_hull_values:
                                col5, col6, col7 = st.columns(3)
                        # Convert to meV for display
                        e_hull_mev = [v * 1000 for v in e_hull_values]
                        with col5:
                            st.metric("Min E_above_hull", f"{min(e_hull_mev):.1f} meV")
                        with col6:
                            st.metric("Max E_above_hull", f"{max(e_hull_mev):.1f} meV")
                        with col7:
                            st.metric("Avg E_above_hull", f"{sum(e_hull_mev)/len(e_hull_mev):.1f} meV")
                        
                        # Show stable examples
                        if stable_materials:
                            st.markdown("### ✅ Stable Materials (on convex hull)")
                            stable_display = []
                            for d in stable_materials[:10]:  # Show first 10
                                # Convert eV to meV for display
                                e_hull_display = f"{d['energy_above_hull'] * 1000:.1f}" if d['energy_above_hull'] is not None else "0.0"
                                
                                stable_display.append({
                                    "Material ID": d['material_id'],
                                    "Formula": d['formula'],
                                    "E_above_hull (meV)": e_hull_display,
                                    "Form E (eV/atom)": f"{d['formation_energy_per_atom']:.4f}" if d['formation_energy_per_atom'] is not None else "N/A"
                                })
                            st.dataframe(stable_display, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No stability data found")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")


def compound_analysis_tab(api_key: str):
    st.markdown('<h2 class="sub-header">Comprehensive Compound Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Complete analysis</strong> of a compound including magnetic moments, metallic nature, and thermodynamic stability.
    </div>
    """, unsafe_allow_html=True)
    
    target_formula = st.text_input(
        "Target Compound Formula",
        value="LuMnBRu",
        key="compound_target_formula",
        help="Enter a chemical formula"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analyze_magnetic = st.checkbox("Analyze Magnetic Moments", value=True, key="compound_analyze_magnetic")
    
    with col2:
        analyze_metallic = st.checkbox("Analyze Metallic Nature", value=True, key="compound_analyze_metallic")
    
    with col3:
        analyze_stability = st.checkbox("Analyze Thermodynamic Stability", value=True, key="compound_analyze_stability")
    
    show_formula_units = st.checkbox("Show Formula Unit Information", value=True, key="compound_show_formula_units")
    
    if st.button("🔬 Run Complete Analysis", type="primary", use_container_width=True, key="compound_analyze_button"):
        if not target_formula:
            st.error("Please enter a compound formula")
        elif not (analyze_magnetic or analyze_metallic or analyze_stability):
            st.error("Please select at least one analysis type")
        else:
            # Parse elements for display
            elements = parse_formula_to_elements(target_formula)
            st.info(f"**Target:** {target_formula} | **Elements:** {', '.join(elements)}")
            
            if analyze_magnetic:
                st.markdown("### 🧲 Magnetic Moment Analysis")
                with st.spinner("Analyzing magnetic moments..."):
                    try:
                        with suppress_stdout():
                            docs = search_related_structures(
                                target_formula=target_formula,
                                api_key=api_key,
                                search_exact=True,
                                search_all_elements=True,
                                search_subsets=True,
                                energy_above_hull=(0, 1.0),
                                theoretical=None,
                                max_results_per_search=30
                            )
                        
                        if docs:
                            mag_data = get_magnetization_data(docs)
                            # Normalize target formula for comparison (element order doesn't matter)
                            target_formula_norm = normalize_formula(target_formula)
                            # Sort by: 1) exact match first, 2) then by magnetization per atom (descending)
                            mag_data = sorted(
                                mag_data,
                                key=lambda x: (
                                    0 if normalize_formula(x['formula']) == target_formula_norm else 1,  # Exact matches first
                                    -(x['magnetization_per_atom'] if x['magnetization_per_atom'] is not None else -999)  # Then by magnetization (descending)
                                )
                            )
                        else:
                            mag_data = []
                        
                        if mag_data:
                            # Check for exact matches
                            exact_matches = [d for d in mag_data if normalize_formula(d['formula']) == target_formula_norm]
                            
                            if exact_matches:
                                st.success(f"✓ Found {len(exact_matches)} exact match(es) for {target_formula} (shown first)")
                            
                            display_data = []
                            for d in mag_data:
                                is_exact = normalize_formula(d['formula']) == target_formula_norm
                                formula_display = f"⭐ {d['formula']}" if is_exact else d['formula']
                                display_data.append({
                                    "Material ID": d['material_id'],
                                    "Formula": formula_display,
                                    "μ/atom (μB)": f"{d['magnetization_per_atom']:.4f}" if d['magnetization_per_atom'] is not None else "N/A",
                                    "Total μ (μB)": f"{d['total_magnetization']:.4f}" if d['total_magnetization'] is not None else "N/A"
                                })
                            
                            st.dataframe(display_data, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No magnetic data found")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if analyze_metallic:
                st.markdown("### ⚡ Metallic Nature Analysis")
                with st.spinner("Analyzing metallic nature..."):
                    try:
                        with suppress_stdout():
                            docs = search_related_structures(
                                target_formula=target_formula,
                                api_key=api_key,
                                search_exact=True,
                                search_all_elements=True,
                                search_subsets=True,
                                energy_above_hull=(0, 1.0),
                                theoretical=None,
                                max_results_per_search=30
                            )
                        
                        if docs:
                            metallic_data = get_metallic_nature_data(docs)
                            # Normalize target formula for comparison (element order doesn't matter)
                            target_formula_norm = normalize_formula(target_formula)
                            # Sort by: 1) exact match first, 2) then by metallic nature
                            def sort_key(x):
                                is_exact_match = normalize_formula(x['formula']) == target_formula_norm
                                exact_priority = 0 if is_exact_match else 1
                                
                                if x['is_metallic'] is True:
                                    return (exact_priority, 0, 0.0)  # Exact matches first, then metallic
                                elif x['is_metallic'] is False:
                                    bg = x['band_gap'] if x['band_gap'] is not None else 999.0
                                    return (exact_priority, 1, bg)  # Exact matches first, then non-metallic by band gap
                                elif x['band_gap'] is not None:
                                    return (exact_priority, 1 if x['band_gap'] > 0 else 0, x['band_gap'] if x['band_gap'] > 0 else 0.0)
                                else:
                                    return (exact_priority, 2, 999.0)  # Exact matches first, then unknown last
                            metallic_data = sorted(metallic_data, key=sort_key)
                        else:
                            metallic_data = []
                        
                        if metallic_data:
                            # Check for exact matches (target_formula_norm already defined above if docs exist)
                            target_formula_norm = normalize_formula(target_formula)
                            exact_matches = [d for d in metallic_data if normalize_formula(d['formula']) == target_formula_norm]
                            
                            if exact_matches:
                                st.success(f"✓ Found {len(exact_matches)} exact match(es) for {target_formula} (shown first)")
                            
                            # Sort by space group priority
                            def sort_key(x):
                                is_exact_match = normalize_formula(x['formula']) == target_formula_norm
                                exact_priority = 0 if is_exact_match else 1
                                space_group_priority = x.get('space_group_priority', 999)
                                
                                if x['is_metallic'] is True:
                                    metallic_priority = 0
                                    bg_value = 0.0
                                elif x['is_metallic'] is False:
                                    metallic_priority = 1
                                    bg_value = x['band_gap'] if x['band_gap'] is not None else 999.0
                                elif x['band_gap'] is not None:
                                    metallic_priority = 1 if x['band_gap'] > 0 else 0
                                    bg_value = x['band_gap'] if x['band_gap'] > 0 else 0.0
                                else:
                                    metallic_priority = 2
                                    bg_value = 999.0
                                
                                return (exact_priority, space_group_priority, metallic_priority, bg_value)
                            metallic_data = sorted(metallic_data, key=sort_key)
                            
                            display_data = []
                            for d in metallic_data:
                                is_exact = normalize_formula(d['formula']) == target_formula_norm
                                formula_display = f"⭐ {d['formula']}" if is_exact else d['formula']
                                display_data.append({
                                    "Material ID": d['material_id'],
                                    "Formula": formula_display,
                                    "Status": d['metallic_status'],
                                    "Space Group": d.get('space_group', 'N/A') or 'N/A',
                                    "Band Gap (eV)": f"{d['band_gap']:.4f}" if d['band_gap'] is not None else "N/A"
                                })
                            
                            st.dataframe(display_data, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No metallic nature data found")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if analyze_stability:
                st.markdown("### 🌡️ Thermodynamic Stability Analysis")
                with st.spinner("Analyzing thermodynamic stability..."):
                    try:
                        with suppress_stdout():
                            docs = search_related_structures(
                                target_formula=target_formula,
                                api_key=api_key,
                                search_exact=True,
                                search_all_elements=True,
                                search_subsets=True,
                                energy_above_hull=(0, 1.0),
                                theoretical=None,
                                max_results_per_search=30
                            )
                        
                        if docs:
                            stability_data = get_thermodynamic_stability_data(docs)
                            # Normalize target formula for comparison (element order doesn't matter)
                            target_formula_norm = normalize_formula(target_formula)
                            # Sort by: 1) exact match first, 2) then by stability
                            def sort_key(x):
                                is_exact_match = normalize_formula(x['formula']) == target_formula_norm
                                exact_priority = 0 if is_exact_match else 1
                                
                                # Stability priority (lower = more stable)
                                if x['is_stable'] is True:
                                    stability_priority = 0
                                elif x['energy_above_hull'] is not None:
                                    e_hull = x['energy_above_hull']
                                    if e_hull == 0:
                                        stability_priority = 0
                                    elif e_hull < 0.05:
                                        stability_priority = 1  # Near-stable
                                    elif e_hull < 0.2:
                                        stability_priority = 2  # Metastable
                                    else:
                                        stability_priority = 3  # Unstable
                                else:
                                    stability_priority = 4  # Unknown
                                
                                # Energy above hull value (for tie-breaking)
                                e_hull_value = x['energy_above_hull'] if x['energy_above_hull'] is not None else 999.0
                                
                                return (exact_priority, stability_priority, e_hull_value)
                            stability_data = sorted(stability_data, key=sort_key)
                        else:
                            stability_data = []
                        
                        if stability_data:
                            # Check for exact matches
                            exact_matches = [d for d in stability_data if normalize_formula(d['formula']) == target_formula_norm]
                            
                            if exact_matches:
                                st.success(f"✓ Found {len(exact_matches)} exact match(es) for {target_formula} (shown first)")
                            
                            display_data = []
                            for d in stability_data:
                                is_exact = normalize_formula(d['formula']) == target_formula_norm
                                formula_display = f"⭐ {d['formula']}" if is_exact else d['formula']
                                
                                # Color-code stability status
                                status = d['stability_status']
                                if "Stable" in status and "on hull" in status:
                                    status_display = f"✅ {status}"
                                elif "Near-stable" in status:
                                    status_display = f"⚠️ {status}"
                                elif "Metastable" in status:
                                    status_display = f"⚡ {status}"
                                elif "Unstable" in status:
                                    status_display = f"❌ {status}"
                                else:
                                    status_display = status
                                
                                # Convert eV to meV for display
                                e_hull_display = f"{d['energy_above_hull'] * 1000:.1f}" if d['energy_above_hull'] is not None else "N/A"
                                
                                display_data.append({
                                    "Material ID": d['material_id'],
                                    "Formula": formula_display,
                                    "Status": status_display,
                                    "E_above_hull (meV)": e_hull_display,
                                    "Form E (eV/atom)": f"{d['formation_energy_per_atom']:.4f}" if d['formation_energy_per_atom'] is not None else "N/A"
                                })
                            
                            st.dataframe(display_data, use_container_width=True, hide_index=True)
                            
                            # Statistics
                            stable_materials = [d for d in stability_data if d['is_stable'] is True]
                            near_stable = [d for d in stability_data if d['energy_above_hull'] is not None and 0 < d['energy_above_hull'] < 0.05]
                            metastable = [d for d in stability_data if d['energy_above_hull'] is not None and 0.05 <= d['energy_above_hull'] < 0.2]
                            unstable = [d for d in stability_data if d['energy_above_hull'] is not None and d['energy_above_hull'] >= 0.2]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Stable (on hull)", len(stable_materials))
                            with col2:
                                st.metric("Near-stable (<50 meV)", len(near_stable))
                            with col3:
                                st.metric("Metastable (50-200 meV)", len(metastable))
                            with col4:
                                st.metric("Unstable (>200 meV)", len(unstable))
                        else:
                            st.warning("No stability data found")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

