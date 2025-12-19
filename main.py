#%%
from typing import List, Optional, Set
from mp_api.client import MPRester
import os
import re
try:
    from pymatgen.core import Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


def list_compounds_by_elements(
    elements: List[str],
    api_key: str,
    num_elements: Optional[int] = None,
    energy_above_hull: tuple = (0, 0.05),
    theoretical: bool = False,
    metallic_only: Optional[bool] = None,
    max_results: Optional[int] = None
) -> List:
    """
    List all compounds containing certain elements, optionally filtered by metallic properties.
    
    Args:
        elements: List of element symbols (e.g., ["Au", "Ni"])
        api_key: Materials Project API key
        num_elements: Exact number of elements in the compound (None for any)
        energy_above_hull: Tuple (min, max) for energy above hull filter
        theoretical: If False, only include materials with experimental matches
        metallic_only: If True, only return metallic materials (band_gap=0).
                      If False, only return non-metallic materials (band_gap>0).
                      If None, return all materials.
        max_results: Maximum number of results to return (None for all)
    
    Returns:
        List of material documents
    """
    with MPRester(api_key) as mpr:
        # Base search parameters
        search_params = {
            "elements": elements,
            "energy_above_hull": energy_above_hull,
            "theoretical": theoretical,
            "fields": [
                "material_id",
                "formula_pretty",
                "structure",
                "symmetry",
                "band_gap",
                "is_metal",
                "total_magnetization"
            ]
        }
        
        # Add num_elements if specified
        if num_elements is not None:
            search_params["num_elements"] = num_elements
        
        # Filter by metallic properties
        if metallic_only is True:
            # Metallic materials have band_gap = 0
            search_params["band_gap"] = (0, 0)
        elif metallic_only is False:
            # Non-metallic materials have band_gap > 0
            search_params["band_gap"] = (0.01, None)  # Small threshold to exclude exactly 0
        
        # Perform search
        docs = mpr.materials.summary.search(**search_params)
        
        # Convert to list and apply max_results limit
        docs_list = list(docs)
        if max_results is not None:
            docs_list = docs_list[:max_results]
        
        return docs_list


def get_magnetization_per_atom(doc) -> Optional[float]:
    """
    Calculate magnetization per atom from a material document.
    
    Args:
        doc: Material document from MP API
    
    Returns:
        Magnetization per atom in μB, or None if not available
    """
    # Try to get total_magnetization field first
    if hasattr(doc, 'total_magnetization') and doc.total_magnetization is not None:
        total_mag = abs(doc.total_magnetization)
        if hasattr(doc, 'structure') and doc.structure is not None:
            n_atoms = len(doc.structure)
            if n_atoms > 0:
                return total_mag / n_atoms
    
    # Fall back to calculating from site magmoms
    if hasattr(doc, 'structure') and doc.structure is not None:
        structure = doc.structure
        if "magmom" in structure.site_properties:
            magmoms = structure.site_properties["magmom"]
            if magmoms:
                total_mag = sum(abs(m) for m in magmoms)
                n_atoms = len(magmoms)
                if n_atoms > 0:
                    return total_mag / n_atoms
    
    return None


def get_total_magnetization(doc) -> Optional[float]:
    """
    Get total magnetization from a material document.
    
    Args:
        doc: Material document from MP API
    
    Returns:
        Total magnetization in μB, or None if not available
    """
    # Try total_magnetization field first
    if hasattr(doc, 'total_magnetization') and doc.total_magnetization is not None:
        return abs(doc.total_magnetization)
    
    # Fall back to summing site magmoms
    if hasattr(doc, 'structure') and doc.structure is not None:
        structure = doc.structure
        if "magmom" in structure.site_properties:
            magmoms = structure.site_properties["magmom"]
            if magmoms:
                return sum(abs(m) for m in magmoms)
    
    return None


def print_materials_info(docs: List, show_metallic_info: bool = True, show_magnetization: bool = False):
    """
    Print information about materials.
    
    Args:
        docs: List of material documents
        show_metallic_info: If True, show metallic/non-metallic classification
        show_magnetization: If True, show magnetization per atom
    """
    print(f"\nFound {len(docs)} materials:\n")
    print("-" * 100)
    
    for doc in docs:
        metallic_status = "Unknown"
        if hasattr(doc, 'is_metal') and doc.is_metal is not None:
            metallic_status = "Metallic" if doc.is_metal else "Non-metallic"
        elif hasattr(doc, 'band_gap') and doc.band_gap is not None:
            metallic_status = "Metallic" if doc.band_gap == 0 else f"Non-metallic (band_gap={doc.band_gap:.3f} eV)"
        
        info = f"{doc.material_id:15s} {doc.formula_pretty:20s}"
        if show_metallic_info:
            info += f" {metallic_status:20s}"
        
        if show_magnetization:
            mag_per_atom = get_magnetization_per_atom(doc)
            if mag_per_atom is not None:
                total_mag = get_total_magnetization(doc)
                n_atoms = len(doc.structure) if hasattr(doc, 'structure') and doc.structure else "?"
                info += f" μ/atom={mag_per_atom:6.3f} μB (total={total_mag:.3f} μB, {n_atoms} atoms)"
            else:
                info += f" μ/atom=N/A"
        
        print(info)
    
    print("-" * 100)


def filter_metallic(docs: List) -> tuple[List, List]:
    """
    Separate materials into metallic and non-metallic lists.
    
    Args:
        docs: List of material documents
    
    Returns:
        Tuple of (metallic_docs, non_metallic_docs)
    """
    metallic = []
    non_metallic = []
    
    for doc in docs:
        is_metallic = False
        
        # Check is_metal field first (if available)
        if hasattr(doc, 'is_metal') and doc.is_metal is not None:
            is_metallic = doc.is_metal
        # Otherwise check band_gap
        elif hasattr(doc, 'band_gap') and doc.band_gap is not None:
            is_metallic = (doc.band_gap == 0)
        
        if is_metallic:
            metallic.append(doc)
        else:
            non_metallic.append(doc)
    
    return metallic, non_metallic


def parse_formula_to_elements(formula: str) -> List[str]:
    """
    Parse a chemical formula to extract element symbols.
    
    Handles formulas like "LuMnBRu", "Fe2O3", "GdH6C4NO9", etc.
    
    Args:
        formula: Chemical formula string
    
    Returns:
        List of unique element symbols as strings
    """
    if PYMATGEN_AVAILABLE:
        try:
            comp = Composition(formula)
            # Convert Element objects to their symbol strings
            elements = [elem.symbol for elem in comp.elements]
            return sorted(elements)
        except:
            pass
    
    # Fallback: regex-based parsing
    # Match element symbols (1-2 capital letters followed by optional lowercase)
    pattern = r'([A-Z][a-z]?)(?:\d*\.?\d*)?'
    matches = re.findall(pattern, formula)
    # Remove duplicates and sort
    elements = sorted(list(set(matches)))
    return elements


def normalize_formula(formula: str) -> str:
    """
    Normalize a chemical formula by sorting elements alphabetically.
    
    This makes "Er3Yb" and "YbEr3" equivalent for comparison purposes.
    Element order does not matter - only the composition counts.
    
    Args:
        formula: Chemical formula string (e.g., "Er3Yb", "YbEr3", "Fe2O3")
    
    Returns:
        Normalized formula string with elements sorted alphabetically
    """
    if PYMATGEN_AVAILABLE:
        try:
            comp = Composition(formula)
            # Get elements and their counts, sorted by element symbol
            elements_sorted = sorted(comp.elements, key=lambda x: x.symbol)
            # Reconstruct formula with sorted elements
            formula_parts = []
            for elem in elements_sorted:
                count = comp[elem]
                if count == 1:
                    formula_parts.append(elem.symbol)
                else:
                    # Format count as integer if whole number, otherwise as float
                    if count == int(count):
                        formula_parts.append(f"{elem.symbol}{int(count)}")
                    else:
                        formula_parts.append(f"{elem.symbol}{count}")
            return "".join(formula_parts)
        except:
            pass
    
    # Fallback: simple regex-based normalization
    # This is less accurate but works for simple cases
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    if matches:
        # Sort by element symbol
        matches_sorted = sorted(matches, key=lambda x: x[0])
        formula_parts = []
        for elem, count_str in matches_sorted:
            if count_str:
                formula_parts.append(f"{elem}{count_str}")
            else:
                formula_parts.append(elem)
        return "".join(formula_parts)
    
    # If parsing fails, return uppercase, space-removed version
    return formula.replace(" ", "").upper()


def search_related_structures(
    target_formula: str,
    api_key: str,
    search_exact: bool = True,
    search_all_elements: bool = True,
    search_subsets: bool = False,
    energy_above_hull: tuple = (0, 1.0),
    theoretical: Optional[bool] = None,
    max_results_per_search: Optional[int] = 50
) -> List:
    """
    Search for structures related to a target compound formula.
    
    Searches in order:
    1. Exact formula match (if search_exact=True)
    2. All elements present (if search_all_elements=True)
    3. Subsets of elements (if search_subsets=True)
    
    Args:
        target_formula: Target compound formula (e.g., "LuMnBRu")
        api_key: Materials Project API key
        search_exact: Search for exact formula match
        search_all_elements: Search for structures containing all elements
        search_subsets: Search for structures with subsets of elements
        energy_above_hull: Energy above hull filter
        theoretical: Filter by theoretical/experimental (None = both)
        max_results_per_search: Max results per search type
    
    Returns:
        List of material documents, sorted by relevance
    """
    elements = parse_formula_to_elements(target_formula)
    print(f"Target formula: {target_formula}")
    print(f"Elements found: {elements}")
    print(f"\nSearching Materials Project database...")
    
    all_results = []
    seen_ids = set()
    
    with MPRester(api_key) as mpr:
        # 1. Try exact formula match (search by elements first, then filter)
        if search_exact:
            try:
                # Search for structures with all elements, then filter for exact formula match
                exact_search_params = {
                    "elements": elements,
                    "num_elements": len(elements),
                    "energy_above_hull": energy_above_hull,
                    "fields": ["material_id", "formula_pretty", "structure", "total_magnetization", "band_gap", "is_metal", "symmetry", "energy_above_hull", "formation_energy_per_atom", "is_stable"],
                    "num_chunks": 1
                }
                # Only add theoretical if it's not None
                if theoretical is not None:
                    exact_search_params["theoretical"] = theoretical
                docs = mpr.materials.summary.search(**exact_search_params)
                # Filter for exact formula match (normalize both formulas for comparison, ignoring element order)
                target_formula_norm = normalize_formula(target_formula)
                exact_matches = [
                    doc for doc in docs 
                    if normalize_formula(doc.formula_pretty) == target_formula_norm
                ]
                if exact_matches:
                    print(f"✓ Found {len(exact_matches)} exact formula match(es)")
                    for doc in exact_matches:
                        if doc.material_id not in seen_ids:
                            all_results.append(doc)
                            seen_ids.add(doc.material_id)
                else:
                    print("✗ No exact formula match found")
            except Exception as e:
                print(f"✗ Exact search failed: {e}")
        
        # 2. Search for all elements present
        if search_all_elements:
            try:
                search_params = {
                    "elements": elements,
                    "num_elements": len(elements),  # Exact number of elements
                    "energy_above_hull": energy_above_hull,
                    "fields": ["material_id", "formula_pretty", "structure", "total_magnetization", "band_gap", "is_metal", "symmetry", "energy_above_hull", "formation_energy_per_atom", "is_stable"],
                    "num_chunks": 1
                }
                # Only add theoretical if it's not None
                if theoretical is not None:
                    search_params["theoretical"] = theoretical
                docs = mpr.materials.summary.search(**search_params)
                all_elements_results = list(docs)
                if max_results_per_search:
                    all_elements_results = all_elements_results[:max_results_per_search]
                
                new_results = [d for d in all_elements_results if d.material_id not in seen_ids]
                if new_results:
                    print(f"✓ Found {len(new_results)} structure(s) with all {len(elements)} elements")
                    all_results.extend(new_results)
                    seen_ids.update(d.material_id for d in new_results)
                else:
                    print(f"✗ No structures found with all {len(elements)} elements")
            except Exception as e:
                print(f"✗ All-elements search failed: {e}")
        
        # 3. Search for subsets (if requested)
        if search_subsets and len(elements) > 2:
            print(f"\nSearching for structures with element subsets...")
            for subset_size in range(len(elements) - 1, 1, -1):  # From n-1 down to 2
                from itertools import combinations
                for subset in combinations(elements, subset_size):
                    try:
                        subset_search_params = {
                            "elements": list(subset),
                            "num_elements": subset_size,
                            "energy_above_hull": energy_above_hull,
                            "fields": ["material_id", "formula_pretty", "structure", "total_magnetization", "band_gap", "is_metal", "symmetry", "energy_above_hull", "formation_energy_per_atom", "is_stable"],
                            "num_chunks": 1
                        }
                        # Only add theoretical if it's not None
                        if theoretical is not None:
                            subset_search_params["theoretical"] = theoretical
                        docs = mpr.materials.summary.search(**subset_search_params)
                        subset_results = list(docs)
                        if max_results_per_search:
                            subset_results = subset_results[:max_results_per_search]
                        
                        new_results = [d for d in subset_results if d.material_id not in seen_ids]
                        if new_results:
                            print(f"  ✓ Found {len(new_results)} structure(s) with {subset}")
                            all_results.extend(new_results)
                            seen_ids.update(d.material_id for d in new_results)
                    except Exception as e:
                        pass  # Skip failed subset searches
    
    print(f"\nTotal unique structures found: {len(all_results)}")
    return all_results


def compare_magnetic_moments_for_compound(
    target_formula: str,
    api_key: str,
    show_formula_units: bool = True
) -> List[dict]:
    """
    Find and compare magnetic moments for structures related to a target compound.
    
    Args:
        target_formula: Target compound formula (e.g., "LuMnBRu")
        api_key: Materials Project API key
        show_formula_units: Show formula unit information in output
    
    Returns:
        List of magnetization data dictionaries
    """
    # Search for related structures
    docs = search_related_structures(
        target_formula=target_formula,
        api_key=api_key,
        search_exact=True,
        search_all_elements=True,
        search_subsets=True,  # Also search subsets for comparison
        energy_above_hull=(0, 1.0),
        theoretical=None,
        max_results_per_search=30
    )
    
    if not docs:
        print(f"\nNo related structures found for {target_formula}")
        return []
    
    # Get magnetization data
    mag_data = get_magnetization_data(docs)
    
    # Normalize target formula for comparison (element order doesn't matter)
    target_formula_norm = normalize_formula(target_formula)
    
    # Sort by: 1) exact match first, 2) then by magnetization per atom (descending)
    mag_data_sorted = sorted(
        mag_data,
        key=lambda x: (
            0 if normalize_formula(x['formula']) == target_formula_norm else 1,  # Exact matches first
            -(x['magnetization_per_atom'] if x['magnetization_per_atom'] is not None else -999)  # Then by magnetization (descending)
        )
    )
    
    # Check for exact matches
    exact_matches = [d for d in mag_data_sorted if normalize_formula(d['formula']) == target_formula_norm]
    
    # Print summary
    print(f"\n{'='*110}")
    print(f"MAGNETIC MOMENT COMPARISON FOR: {target_formula}")
    if exact_matches:
        print(f"✓ Found {len(exact_matches)} exact match(es) - shown first in results")
    print(f"{'='*110}")
    print_magnetization_summary(mag_data_sorted, show_formula_units=show_formula_units)
    
    # Statistics
    magnetic_materials = [d for d in mag_data_sorted if d['magnetization_per_atom'] is not None and d['magnetization_per_atom'] > 0.01]
    if magnetic_materials:
        moments = [d['magnetization_per_atom'] for d in magnetic_materials]
        print(f"\nStatistics for materials with μ/atom > 0.01 μB:")
        print(f"  Count: {len(magnetic_materials)}")
        print(f"  Average μ/atom: {sum(moments)/len(moments):.4f} μB")
        print(f"  Maximum μ/atom: {max(moments):.4f} μB")
        print(f"  Minimum μ/atom: {min(moments):.4f} μB")
    
    return mag_data_sorted


# Element classification
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"
}

ALKALI_ALKALINE_EARTH = {
    "Li", "Na", "K", "Rb", "Cs", "Fr",  # Alkali metals
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra"   # Alkaline earth metals
}

LANTHANIDES = {
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", 
    "Er", "Tm", "Yb", "Lu"
}

# Space group priority (lower number = higher priority/more metallic)
SPACE_GROUP_PRIORITY = {
    "Fm-3m": 1,      # Most metallic (cubic)
    "Im-3m": 2,      # Second
    "P6_3/mmc": 3,   # Third
}


def classify_element_type(element: str) -> str:
    """
    Classify an element as transition metal, alkali/alkaline earth, lanthanide, or other.
    
    Args:
        element: Element symbol (e.g., "Fe", "Na", "Gd")
    
    Returns:
        Classification string: "transition_metal", "alkali_alkaline", "lanthanide", or "other"
    """
    elem_upper = element.capitalize()
    
    if elem_upper in TRANSITION_METALS:
        return "transition_metal"
    elif elem_upper in ALKALI_ALKALINE_EARTH:
        return "alkali_alkaline"
    elif elem_upper in LANTHANIDES:
        return "lanthanide"
    else:
        return "other"


def get_space_group_priority(space_group: Optional[str]) -> int:
    """
    Get priority for space group (lower = more metallic/higher priority).
    
    Args:
        space_group: Space group symbol (e.g., "Fm-3m")
    
    Returns:
        Priority integer (1-3 for known groups, 999 for others)
    """
    if space_group is None:
        return 999
    
    # Normalize space group (handle variations)
    sg_normalized = space_group.strip()
    
    if sg_normalized in SPACE_GROUP_PRIORITY:
        return SPACE_GROUP_PRIORITY[sg_normalized]
    else:
        return 999  # Unknown space groups at the end


def is_metallic_by_elements(formula: str) -> Optional[bool]:
    """
    Determine if a compound is likely metallic based on majority element types.
    
    If majority of elements are transition metals, alkali/alkaline earth, or lanthanides,
    classify as metallic.
    
    Args:
        formula: Chemical formula (e.g., "Fe2O3", "LuMnBRu")
    
    Returns:
        True if metallic by element composition, False if not, None if cannot determine
    """
    elements = parse_formula_to_elements(formula)
    
    if not elements:
        return None
    
    # Count element types
    metallic_types = {"transition_metal": 0, "alkali_alkaline": 0, "lanthanide": 0}
    other_count = 0
    
    for elem in elements:
        elem_type = classify_element_type(elem)
        if elem_type in metallic_types:
            metallic_types[elem_type] += 1
        else:
            other_count += 1
    
    total_metallic = sum(metallic_types.values())
    total_elements = len(elements)
    
    # If majority are metallic-type elements, classify as metallic
    if total_metallic > total_elements / 2:
        return True
    elif other_count > total_elements / 2:
        return False
    else:
        # Equal or unclear - could go either way
        return None


def get_metallic_nature_data(docs: List) -> List[dict]:
    """
    Extract metallic nature data for a list of materials.
    
    Includes element classification and space group information.
    
    Args:
        docs: List of material documents
    
    Returns:
        List of dictionaries with metallic nature information
    """
    results = []
    for doc in docs:
        is_metallic = None
        band_gap = None
        
        # Get band_gap if available
        if hasattr(doc, 'band_gap') and doc.band_gap is not None:
            band_gap = doc.band_gap
        
        # Get space group
        space_group = None
        if hasattr(doc, 'symmetry') and doc.symmetry is not None:
            if hasattr(doc.symmetry, 'symbol'):
                space_group = doc.symmetry.symbol
            elif hasattr(doc.symmetry, 'space_group_symbol'):
                space_group = doc.symmetry.space_group_symbol
            elif isinstance(doc.symmetry, str):
                space_group = doc.symmetry
            elif hasattr(doc.symmetry, '__dict__'):
                # Try to find symbol in dict
                sym_dict = doc.symmetry.__dict__ if hasattr(doc.symmetry, '__dict__') else {}
                space_group = sym_dict.get('symbol') or sym_dict.get('space_group_symbol')
        
        # Fallback: try to get from structure
        if space_group is None and hasattr(doc, 'structure') and doc.structure is not None:
            try:
                if hasattr(doc.structure, 'get_space_group_info'):
                    sg_info = doc.structure.get_space_group_info()
                    if sg_info and len(sg_info) > 0:
                        space_group = sg_info[0]
            except:
                pass
        
        # Classify elements and determine metallic nature by element composition
        formula = doc.formula_pretty
        elements = parse_formula_to_elements(formula)
        element_types = [classify_element_type(elem) for elem in elements]
        is_metallic_by_elem = is_metallic_by_elements(formula)
        
        # Check is_metal field first
        if hasattr(doc, 'is_metal') and doc.is_metal is not None:
            is_metallic = doc.is_metal
        # Otherwise determine from band_gap
        elif band_gap is not None:
            is_metallic = (band_gap == 0)
        # Fall back to element-based classification
        elif is_metallic_by_elem is not None:
            is_metallic = is_metallic_by_elem
        
        metallic_status = "Unknown"
        if is_metallic is True:
            metallic_status = "Metallic"
        elif is_metallic is False:
            metallic_status = "Non-metallic"
        elif band_gap is not None:
            metallic_status = "Metallic" if band_gap == 0 else f"Non-metallic"
        elif is_metallic_by_elem is not None:
            metallic_status = "Metallic (by elements)" if is_metallic_by_elem else "Non-metallic (by elements)"
        
        # Get space group priority
        space_group_priority = get_space_group_priority(space_group)
        
        results.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "is_metallic": is_metallic,
            "band_gap": band_gap,
            "metallic_status": metallic_status,
            "space_group": space_group,
            "space_group_priority": space_group_priority,
            "element_types": element_types,
            "is_metallic_by_elements": is_metallic_by_elem,
            "n_atoms": get_n_atoms_in_unit_cell(doc),
            "n_atoms_per_formula": get_n_atoms_per_formula_unit(doc),
            "doc": doc  # Keep reference to original doc
        })
    
    return results


def print_metallic_nature_summary(metallic_data: List[dict], show_formula_units: bool = False):
    """
    Print a summary of metallic nature data.
    
    Args:
        metallic_data: List of dictionaries from get_metallic_nature_data()
        show_formula_units: Show formula unit information in output
    """
    if show_formula_units:
        print(f"\n{'Material ID':<15} {'Formula':<20} {'Status':<20} {'Space Group':<12} {'Band Gap (eV)':<15} {'N (unit cell)':<15} {'N (formula)':<12}")
        print("-" * 130)
        
        for data in metallic_data:
            bg_str = f"{data['band_gap']:.4f}" if data['band_gap'] is not None else "N/A"
            n_atoms_str = str(data['n_atoms']) if data['n_atoms'] is not None else "N/A"
            n_formula_str = str(data['n_atoms_per_formula']) if data.get('n_atoms_per_formula') is not None else "N/A"
            space_group_str = data.get('space_group', 'N/A') or 'N/A'
     
            print(f"{data['material_id']:<15} {data['formula']:<20} {data['metallic_status']:<20} {space_group_str:<12} {bg_str:<15} {n_atoms_str:<15} {n_formula_str:<12}")
    else:
        print(f"\n{'Material ID':<15} {'Formula':<20} {'Status':<20} {'Space Group':<12} {'Band Gap (eV)':<15}")
        print("-" * 120)
        
        for data in metallic_data:
            bg_str = f"{data['band_gap']:.4f}" if data['band_gap'] is not None else "N/A"
            space_group_str = data.get('space_group', 'N/A') or 'N/A'
     
            print(f"{data['material_id']:<15} {data['formula']:<20} {data['metallic_status']:<20} {space_group_str:<12} {bg_str:<15}")
    
    print("-" * (130 if show_formula_units else 120))


def compare_metallic_nature_for_compound(
    target_formula: str,
    api_key: str,
    show_formula_units: bool = True
) -> List[dict]:
    """
    Find and compare metallic nature for structures related to a target compound.
    
    Classifies elements as transition metals, alkali/alkaline earth, or lanthanides.
    If majority of elements are one of these types, labels as metallic.
    
    Ranks materials by:
    1. Exact formula matches first
    2. Space group priority (Fm-3m > Im-3m > P6_3/mmc > others)
    3. Metallic nature (metallic > non-metallic by band gap)
    
    Args:
        target_formula: Target compound formula (e.g., "LuMnBRu")
        api_key: Materials Project API key
        show_formula_units: Show formula unit information in output
    
    Returns:
        List of metallic nature data dictionaries, sorted by priority
    """
    # Search for related structures
    docs = search_related_structures(
        target_formula=target_formula,
        api_key=api_key,
        search_exact=True,
        search_all_elements=True,
        search_subsets=True,  # Also search subsets for comparison
        energy_above_hull=(0, 1.0),
        theoretical=None,
        max_results_per_search=30
    )
    
    if not docs:
        print(f"\nNo related structures found for {target_formula}")
        return []
    
    # Get metallic nature data
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
    
    metallic_data_sorted = sorted(metallic_data, key=sort_key)
    
    # Check for exact matches
    exact_matches = [d for d in metallic_data_sorted if normalize_formula(d['formula']) == target_formula_norm]
    
    # Print summary
    print(f"\n{'='*130}")
    print(f"METALLIC NATURE COMPARISON FOR: {target_formula}")
    if exact_matches:
        print(f"✓ Found {len(exact_matches)} exact match(es) - shown first in results")
    print(f"Sorting: Exact matches > Space group (Fm-3m > Im-3m > P6_3/mmc > others) > Metallic nature")
    print(f"{'='*130}")
    print_metallic_nature_summary(metallic_data_sorted, show_formula_units=show_formula_units)
    
    # Statistics
    metallic_materials = [d for d in metallic_data_sorted if d['is_metallic'] is True]
    non_metallic_materials = [d for d in metallic_data_sorted if d['is_metallic'] is False]
    unknown_materials = [d for d in metallic_data_sorted if d['is_metallic'] is None and d['band_gap'] is None]
    
    print(f"\nStatistics:")
    print(f"  Metallic materials: {len(metallic_materials)}")
    print(f"  Non-metallic materials: {len(non_metallic_materials)}")
    if unknown_materials:
        print(f"  Unknown: {len(unknown_materials)}")
    
    if metallic_materials:
        print(f"\n  Metallic examples:")
        for d in metallic_materials[:5]:  # Show first 5
            print(f"    {d['formula']:<20} {d['material_id']}")
    
    if non_metallic_materials:
        band_gaps = [d['band_gap'] for d in non_metallic_materials if d['band_gap'] is not None]
        if band_gaps:
            print(f"\n  Non-metallic band gap range:")
            print(f"    Min: {min(band_gaps):.4f} eV")
            print(f"    Max: {max(band_gaps):.4f} eV")
            print(f"    Average: {sum(band_gaps)/len(band_gaps):.4f} eV")
    
    return metallic_data_sorted


def get_thermodynamic_stability_data(docs: List) -> List[dict]:
    """
    Extract thermodynamic stability data for a list of materials.
    
    Args:
        docs: List of material documents
    
    Returns:
        List of dictionaries with stability information
    """
    results = []
    for doc in docs:
        # Get stability metrics
        energy_above_hull = None
        if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None:
            energy_above_hull = doc.energy_above_hull
        
        formation_energy_per_atom = None
        if hasattr(doc, 'formation_energy_per_atom') and doc.formation_energy_per_atom is not None:
            formation_energy_per_atom = doc.formation_energy_per_atom
        
        is_stable = None
        if hasattr(doc, 'is_stable') and doc.is_stable is not None:
            is_stable = doc.is_stable
        # Determine from energy_above_hull if available
        elif energy_above_hull is not None:
            is_stable = (energy_above_hull == 0)
        
        # Get decomposition information if available
        decomposition = None
        if hasattr(doc, 'decomposition') and doc.decomposition is not None:
            decomposition = doc.decomposition
        
        # Stability classification (convert eV to meV for display)
        stability_status = "Unknown"
        if is_stable is True:
            stability_status = "Stable (on hull)"
        elif is_stable is False:
            stability_status = "Unstable (above hull)"
        elif energy_above_hull is not None:
            energy_mev = energy_above_hull * 1000  # Convert eV to meV
            if energy_above_hull == 0:
                stability_status = "Stable (on hull)"
            elif energy_above_hull < 0.05:
                stability_status = f"Near-stable ({energy_mev:.1f} meV)"
            elif energy_above_hull < 0.2:
                stability_status = f"Metastable ({energy_mev:.1f} meV)"
            else:
                stability_status = f"Unstable ({energy_mev:.1f} meV)"
        
        results.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "energy_above_hull": energy_above_hull,
            "formation_energy_per_atom": formation_energy_per_atom,
            "is_stable": is_stable,
            "stability_status": stability_status,
            "decomposition": decomposition,
            "n_atoms": get_n_atoms_in_unit_cell(doc),
            "n_atoms_per_formula": get_n_atoms_per_formula_unit(doc),
            "doc": doc  # Keep reference to original doc
        })
    
    return results


def print_stability_summary(stability_data: List[dict], show_formula_units: bool = False):
    """
    Print a summary of thermodynamic stability data.
    
    Args:
        stability_data: List of dictionaries from get_thermodynamic_stability_data()
        show_formula_units: Show formula unit information in output
    """
    if show_formula_units:
        print(f"\n{'Material ID':<15} {'Formula':<20} {'Status':<25} {'E_above_hull (meV)':<18} {'Form E (eV/atom)':<18} {'N (unit cell)':<15} {'N (formula)':<12}")
        print("-" * 140)
        
        for data in stability_data:
            # Convert eV to meV for display
            if data['energy_above_hull'] is not None:
                e_hull_str = f"{data['energy_above_hull'] * 1000:.1f}"
            else:
                e_hull_str = "N/A"
            form_e_str = f"{data['formation_energy_per_atom']:.4f}" if data['formation_energy_per_atom'] is not None else "N/A"
            n_atoms_str = str(data['n_atoms']) if data['n_atoms'] is not None else "N/A"
            n_formula_str = str(data['n_atoms_per_formula']) if data.get('n_atoms_per_formula') is not None else "N/A"
     
            print(f"{data['material_id']:<15} {data['formula']:<20} {data['stability_status']:<25} {e_hull_str:<18} {form_e_str:<18} {n_atoms_str:<15} {n_formula_str:<12}")
    else:
        print(f"\n{'Material ID':<15} {'Formula':<20} {'Status':<25} {'E_above_hull (meV)':<18} {'Form E (eV/atom)':<18}")
        print("-" * 120)
        
        for data in stability_data:
            # Convert eV to meV for display
            if data['energy_above_hull'] is not None:
                e_hull_str = f"{data['energy_above_hull'] * 1000:.1f}"
            else:
                e_hull_str = "N/A"
            form_e_str = f"{data['formation_energy_per_atom']:.4f}" if data['formation_energy_per_atom'] is not None else "N/A"
     
            print(f"{data['material_id']:<15} {data['formula']:<20} {data['stability_status']:<25} {e_hull_str:<18} {form_e_str:<18}")
    
    print("-" * (140 if show_formula_units else 120))


def compare_thermodynamic_stability_for_compound(
    target_formula: str,
    api_key: str,
    show_formula_units: bool = True
) -> List[dict]:
    """
    Find and compare thermodynamic stability for structures related to a target compound.
    
    Ranks materials by:
    1. Exact formula matches first
    2. Stability (stable > near-stable > metastable > unstable)
    3. Energy above hull (lower is better)
    
    Args:
        target_formula: Target compound formula (e.g., "LuMnBRu")
        api_key: Materials Project API key
        show_formula_units: Show formula unit information in output
    
    Returns:
        List of stability data dictionaries, sorted by stability
    """
    # Search for related structures
    docs = search_related_structures(
        target_formula=target_formula,
        api_key=api_key,
        search_exact=True,
        search_all_elements=True,
        search_subsets=True,
        energy_above_hull=(0, 1.0),  # Allow wider range to see stability spectrum
        theoretical=None,
        max_results_per_search=30
    )
    
    if not docs:
        print(f"\nNo related structures found for {target_formula}")
        return []
    
    # Get stability data
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
    
    stability_data_sorted = sorted(stability_data, key=sort_key)
    
    # Check for exact matches
    exact_matches = [d for d in stability_data_sorted if normalize_formula(d['formula']) == target_formula_norm]
    
    # Print summary
    print(f"\n{'='*140}")
    print(f"THERMODYNAMIC STABILITY COMPARISON FOR: {target_formula}")
    if exact_matches:
        print(f"✓ Found {len(exact_matches)} exact match(es) - shown first in results")
    print(f"Sorting: Exact matches > Stability (stable > near-stable > metastable > unstable) > Energy above hull (meV)")
    print(f"{'='*140}")
    print_stability_summary(stability_data_sorted, show_formula_units=show_formula_units)
    
    # Statistics
    stable_materials = [d for d in stability_data_sorted if d['is_stable'] is True]
    near_stable = [d for d in stability_data_sorted if d['energy_above_hull'] is not None and 0 < d['energy_above_hull'] < 0.05]
    metastable = [d for d in stability_data_sorted if d['energy_above_hull'] is not None and 0.05 <= d['energy_above_hull'] < 0.2]
    unstable = [d for d in stability_data_sorted if d['energy_above_hull'] is not None and d['energy_above_hull'] >= 0.2]
    
    print(f"\nStatistics:")
    print(f"  Stable (on hull): {len(stable_materials)}")
    print(f"  Near-stable (<0.05 eV): {len(near_stable)}")
    print(f"  Metastable (0.05-0.2 eV): {len(metastable)}")
    print(f"  Unstable (>0.2 eV): {len(unstable)}")
    
    if stable_materials:
        print(f"\n  Stable examples:")
        for d in stable_materials[:5]:  # Show first 5
            print(f"    {d['formula']:<20} {d['material_id']}")
    
    if stability_data_sorted:
        e_hull_values = [d['energy_above_hull'] for d in stability_data_sorted if d['energy_above_hull'] is not None]
        if e_hull_values:
            e_hull_mev = [v * 1000 for v in e_hull_values]  # Convert to meV
            print(f"\n  Energy above hull range:")
            print(f"    Min: {min(e_hull_mev):.1f} meV")
            print(f"    Max: {max(e_hull_mev):.1f} meV")
            print(f"    Average: {sum(e_hull_mev)/len(e_hull_mev):.1f} meV")
    
    return stability_data_sorted


def get_n_atoms_in_unit_cell(doc) -> Optional[int]:
    """
    Get the number of atoms in the unit cell (total sites in structure).
    
    This is len(structure), which gives the total number of atomic sites
    in the unit cell. This may be larger than the formula unit if the
    structure contains multiple formula units.
    
    Args:
        doc: Material document from MP API
    
    Returns:
        Number of atoms in unit cell, or None if not available
    """
    if hasattr(doc, 'structure') and doc.structure is not None:
        return len(doc.structure)
    return None


def get_n_atoms_per_formula_unit(doc) -> Optional[int]:
    """
    Get the number of atoms per formula unit.
    
    Calculates from the structure's composition, which gives the
    reduced formula unit count.
    
    Args:
        doc: Material document from MP API
    
    Returns:
        Number of atoms per formula unit, or None if not available
    """
    if hasattr(doc, 'structure') and doc.structure is not None:
        structure = doc.structure
        # Get composition and sum all element counts
        composition = structure.composition
        return int(sum(composition.values()))
    return None


def get_magnetization_data(docs: List) -> List[dict]:
    """
    Extract magnetization data for a list of materials.
    
    Args:
        docs: List of material documents
    
    Returns:
        List of dictionaries with magnetization information
    """
    results = []
    for doc in docs:
        mag_per_atom = get_magnetization_per_atom(doc)
        total_mag = get_total_magnetization(doc)
        n_atoms_unit_cell = get_n_atoms_in_unit_cell(doc)
        n_atoms_formula = get_n_atoms_per_formula_unit(doc)
        
        results.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "magnetization_per_atom": mag_per_atom,
            "total_magnetization": total_mag,
            "n_atoms": n_atoms_unit_cell,  # Total atoms in unit cell
            "n_atoms_per_formula": n_atoms_formula,  # Atoms per formula unit
            "doc": doc  # Keep reference to original doc
        })
    
    return results


def print_magnetization_summary(magnetization_data: List[dict], show_formula_units: bool = False):
    """
    Print a summary of magnetization data.
    
    Args:
        magnetization_data: List of dictionaries from get_magnetization_data()
        show_formula_units: If True, show atoms per formula unit in addition to unit cell atoms
    """
    if show_formula_units:
        print(f"\n{'Material ID':<15} {'Formula':<20} {'μ/atom (μB)':<15} {'Total μ (μB)':<15} {'N (unit cell)':<15} {'N (formula)':<12}")
        print("-" * 110)
        
        for data in magnetization_data:
            mag_str = f"{data['magnetization_per_atom']:.4f}" if data['magnetization_per_atom'] is not None else "N/A"
            total_str = f"{data['total_magnetization']:.4f}" if data['total_magnetization'] is not None else "N/A"
            n_atoms_str = str(data['n_atoms']) if data['n_atoms'] is not None else "N/A"
            n_formula_str = str(data['n_atoms_per_formula']) if data.get('n_atoms_per_formula') is not None else "N/A"
     
            print(f"{data['material_id']:<15} {data['formula']:<20} {mag_str:<15} {total_str:<15} {n_atoms_str:<15} {n_formula_str:<12}")
    else:
        print(f"\n{'Material ID':<15} {'Formula':<20} {'μ/atom (μB)':<15} {'Total μ (μB)':<15} {'N atoms':<10}")
        print("-" * 100)
        
        for data in magnetization_data:
            mag_str = f"{data['magnetization_per_atom']:.4f}" if data['magnetization_per_atom'] is not None else "N/A"
            total_str = f"{data['total_magnetization']:.4f}" if data['total_magnetization'] is not None else "N/A"
            n_atoms_str = str(data['n_atoms']) if data['n_atoms'] is not None else "N/A"
     
            print(f"{data['material_id']:<15} {data['formula']:<20} {mag_str:<15} {total_str:<15} {n_atoms_str:<10}")
    
    print("-" * (110 if show_formula_units else 100))


def main():
    # Get API key from environment or set directly
    API_KEY = os.environ.get("MP_API_KEY", "BzFaWHoNjsE7d4DfqxDw5BBcfFCFihYE")
    
    # Example 1: List all compounds containing Au and Ni (any number of elements)
    print("=" * 80)
    print("Example 1: All compounds containing Au and Ni")
    print("=" * 80)
    docs_all = list_compounds_by_elements(
        elements=["Au", "Ni"],
        api_key=API_KEY,
        energy_above_hull=(0, 0.05),
        theoretical=False,
        metallic_only=None,  # Get all materials
        max_results=20
    )
    print_materials_info(docs_all)
    
    # Example 2: Only metallic compounds
    print("\n" + "=" * 80)
    print("Example 2: Metallic compounds only (band_gap = 0)")
    print("=" * 80)
    docs_metallic = list_compounds_by_elements(
        elements=["Au", "Ni"],
        api_key=API_KEY,
        num_elements=3,  # Exactly 3 elements
        energy_above_hull=(0, 0.05),
        theoretical=False,
        metallic_only=True,
        max_results=20
    )
    print_materials_info(docs_metallic)
    
    # Example 3: Only non-metallic compounds
    print("\n" + "=" * 80)
    print("Example 3: Non-metallic compounds only (band_gap > 0)")
    print("=" * 80)
    docs_non_metallic = list_compounds_by_elements(
        elements=["Au", "Ni"],
        api_key=API_KEY,
        num_elements=3,
        energy_above_hull=(0, 0.05),
        theoretical=False,
        metallic_only=False,
        max_results=20
    )
    print_materials_info(docs_non_metallic)
    
    # Example 4: Filter existing results
    print("\n" + "=" * 80)
    print("Example 4: Filtering existing results into metallic/non-metallic")
    print("=" * 80)
    metallic, non_metallic = filter_metallic(docs_all)
    print(f"\nMetallic materials: {len(metallic)}")
    print(f"Non-metallic materials: {len(non_metallic)}")
    
    if metallic:
        print("\nMetallic materials:")
        print_materials_info(metallic, show_metallic_info=False)
    
    if non_metallic:
        print("\nNon-metallic materials:")
        print_materials_info(non_metallic, show_metallic_info=False)
    
    # Example 5: Get magnetization per atom
    print("\n" + "=" * 80)
    print("Example 5: Magnetization per atom for metallic compounds")
    print("=" * 80)
    docs_with_mag = list_compounds_by_elements(
        elements=["Fe", "Ni"],  # Using Fe and Ni as they're more likely to be magnetic
        api_key=API_KEY,
        energy_above_hull=(0, 0.05),
        theoretical=False,
        metallic_only=True,
        max_results=10
    )
    
    # Get magnetization data
    mag_data = get_magnetization_data(docs_with_mag)
    print_magnetization_summary(mag_data)
    
    # Filter to only show materials with non-zero magnetization
    magnetic_materials = [d for d in mag_data if d['magnetization_per_atom'] is not None and d['magnetization_per_atom'] > 0.01]
    if magnetic_materials:
        print(f"\nFound {len(magnetic_materials)} materials with non-zero magnetization:")
        print_magnetization_summary(magnetic_materials)


if __name__ == "__main__":
    main()

# %%
# Example: Get magnetization per atom for specific elements
# 
# Note on n_atoms calculation:
# - "N (unit cell)": Total number of atomic sites in the structure's unit cell (len(structure))
#   This is what's used to calculate μ/atom = total_magnetization / n_atoms
# - "N (formula)": Number of atoms per formula unit (from composition)
#   For example, GdH6C4NO9 has 21 atoms per formula unit, but if the unit cell
#   contains 2 formula units, then N (unit cell) = 42
API_KEY = os.environ.get("MP_API_KEY", "BzFaWHoNjsE7d4DfqxDw5BBcfFCFihYE")

# Broadened search: include all compounds containing Mn and Lu, with minimal filtering
docs = list_compounds_by_elements(
    elements=["Ag","Al"],
    api_key=API_KEY,
    num_elements=4,            # Any number of elements permitted
    metallic_only=None,           # Include both metals and non-metals
    energy_above_hull=(0, 1.0),   # Allow higher energy above hull (up to 1 eV) for broader results
    theoretical=None,             # Allow both theoretical and experimental
    max_results=None              # No limit on number of results
)

# Get magnetization data
mag_data = get_magnetization_data(docs)

# Print summary (showing both unit cell and formula unit atom counts)
print_magnetization_summary(mag_data, show_formula_units=True)

# Filter for materials with significant magnetization
magnetic = [d for d in mag_data if d['magnetization_per_atom'] is not None and d['magnetization_per_atom'] > 0.1]
print(f"\nMaterials with μ/atom > 0.1 μB:")
print_magnetization_summary(magnetic, show_formula_units=True)



# %%
# Example: Find and compare magnetic moments for a specific compound
# 
# This function searches for:
# 1. Exact formula match (if available in MP database)
# 2. Structures with all the same elements
# 3. Structures with subsets of elements (for comparison)
#
# Useful when the exact compound might not be in the database but you want
# to estimate its magnetic moment from related structures.

API_KEY = os.environ.get("MP_API_KEY", "BzFaWHoNjsE7d4DfqxDw5BBcfFCFihYE")

# Example: Search for LuMnBRu and related structures
target_compound = "NpReSn"
mag_data = compare_magnetic_moments_for_compound(
    target_formula=target_compound,
    api_key=API_KEY,
    show_formula_units=True
)

# You can also use it for other compounds:
#mag_data = compare_magnetic_moments_for_compound("Fe2O3", API_KEY)
# mag_data = compare_magnetic_moments_for_compound("GdN", API_KEY)

# %%
# Example: Find and compare metallic nature for a specific compound
# 
# This function searches for related structures and ranks them by metallic nature:
# 1. Metallic materials (band_gap = 0) are shown first
# 2. Non-metallic materials are sorted by band gap (smallest first)
#
# Useful when you want to estimate whether a compound is metallic or non-metallic
# based on related structures in the database.

API_KEY = os.environ.get("MP_API_KEY", "BzFaWHoNjsE7d4DfqxDw5BBcfFCFihYE")

# Example: Search for MnTe and related structures, ranked by metallic nature
target_compound = "MnT"
metallic_data = compare_metallic_nature_for_compound(
    target_formula=target_compound,
    api_key=API_KEY,
    show_formula_units=True
)

# You can also use it for other compounds:
# metallic_data = compare_metallic_nature_for_compound("Fe2O3", API_KEY)
# metallic_data = compare_metallic_nature_for_compound("CuAg", API_KEY)

# %%
