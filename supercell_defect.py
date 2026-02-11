import numpy as np
import random
import os
import re
import argparse
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.geometry import find_mic
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE
from ase.build import make_supercell

# =========================================================
# 1. Potential for Resolving Atomic Overlaps
# =========================================================
class SoftRepulsivePotential(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, cutoff=1.2, repulsion_strength=20.0, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.strength = repulsion_strength

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.positions
        cell = atoms.cell
        forces = np.zeros_like(positions)
        energy = 0.0

        nl = NeighborList([self.cutoff] * len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)

        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                vec = positions[j] + np.dot(offset, cell) - positions[i]
                dist = np.linalg.norm(vec)
                if 0.01 < dist < self.cutoff:
                    delta = self.cutoff - dist
                    force_mag = self.strength * (delta ** 2) / dist
                    forces[i] -= force_mag * vec
                    energy += (self.strength / 3.0) * (delta ** 3)
        
        self.results['energy'] = energy
        self.results['forces'] = forces

def run_pre_optimization(atoms, fmax=0.1, steps=100):
    """Executes a short relaxation to push overlapping atoms apart."""
    old_calc = atoms.calc
    atoms.calc = SoftRepulsivePotential()
    dyn = FIRE(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    
    max_f = np.sqrt((atoms.get_forces()**2).sum(axis=1).max())
    atoms.calc = old_calc
    print(f"    ⚙️ [Pre-Opt] Done. (Max Force: {max_f:.4f} eV/A)")
    return atoms

# =========================================================
# 2. Geometric Utilities
# =========================================================
def reconstruct_molecule_pbc(atoms, start_idx):
    """Reconstructs a molecule split across periodic boundaries."""
    cutoffs = [c * 1.25 for c in natural_cutoffs(atoms)]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    visited = {start_idx}; queue = [start_idx]; cluster_indices = [start_idx]
    work_atoms = atoms.copy()
    while queue:
        current_idx = queue.pop(0)
        curr_pos = work_atoms.positions[current_idx]
        for n_idx in nl.get_neighbors(current_idx)[0]:
            if n_idx not in visited:
                visited.add(n_idx); queue.append(n_idx); cluster_indices.append(n_idx)
                vec, _ = find_mic(work_atoms.positions[n_idx] - curr_pos, work_atoms.cell, work_atoms.pbc)
                work_atoms.positions[n_idx] = curr_pos + vec
    return work_atoms[cluster_indices], cluster_indices

def get_principal_axes(mol):
    """Returns principal axes sorted by moments of inertia."""
    vals, vecs = mol.get_moments_of_inertia(vectors=True)
    return vecs[np.argsort(vals)]

def get_180_rotation_matrix(axis):
    """Generates a 180-degree rotation matrix around a specific axis."""
    u = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(u, u) - np.eye(3)

def is_redundant_flip(mol_orig, pos_flip, tol=0.5):
    """Checks symmetry using distance-based matching (cdist) to handle noise."""
    from scipy.spatial.distance import cdist
    numbers = mol_orig.get_atomic_numbers()
    unique_nums = np.unique(numbers)
    
    for num in unique_nums:
        mask = (numbers == num)
        orig_p, flip_p = mol_orig.positions[mask], pos_flip[mask]
        min_dists = np.min(cdist(flip_p, orig_p), axis=1)
        if np.any(min_dists > tol):
            return False
    return True

def find_all_molecules(atoms):
    """Identifies and groups all unique molecules in the structure."""
    all_indices = set(range(len(atoms))); visited_global = set(); molecules_info = [] 
    for idx in all_indices:
        if idx not in visited_global:
            _, indices = reconstruct_molecule_pbc(atoms, idx)
            visited_global.update(indices)
            molecules_info.append({
                'indices': indices, 
                'symbols': atoms[indices].get_chemical_formula(), 
                'center_atom_idx': indices[0]
            })
    return molecules_info

# =========================================================
# 3. Supercell Generation Logic
# =========================================================
def generate_supercell(atoms, target_size=30.0):
    """
    Generates a supercell based on a target size (in Angstroms).
    """
    cell_lengths = atoms.cell.lengths()
    
    # Calculate multipliers (at least 1)
    nx = max(1, int(round(target_size / cell_lengths[0])))
    ny = max(1, int(round(target_size / cell_lengths[1])))
    nz = max(1, int(round(target_size / cell_lengths[2])))
    
    print(f"  🏗️ Extending Cell: {nx}x{ny}x{nz} (Target: ~{target_size} Å)")
    
    # Create supercell matrix
    P = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]
    supercell = make_supercell(atoms, P)
    
    return supercell, (nx, ny, nz)

# =========================================================
# 4. Main Workflow
# =========================================================
def run_pipeline(target_file, source_file, concentrations, target_size=30.0, target_example_idx=0, wrap_coords=True, seed=None):
    """
    concentrations: List of percentages (e.g., [10, 20, 50])
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    
    # File Naming Logic
    t_prefix = os.path.basename(target_file).split('_')[0]
    s_prefix = re.sub(r'\d+$', '', os.path.splitext(os.path.basename(source_file))[0])
    
    print(f"--- [Pipeline] Step 1: Supercell Generation ({t_prefix}) ---")
    unit_cell = read(target_file)
    supercell, (nx, ny, nz) = generate_supercell(unit_cell, target_size)
    
    # Prefix for output files
    supercell_prefix = f"{t_prefix}_{nx}x{ny}x{nz}_{s_prefix}"
    
    print(f"--- [Pipeline] Step 2: Substitution ({s_prefix}) ---")
    source_atoms = read(source_file)
    
    # Process source molecule
    source_mol, _ = reconstruct_molecule_pbc(source_atoms, 0)
    source_mol.positions -= source_mol.get_center_of_mass()
    
    # Rotation and Symmetry logic
    s_axes = get_principal_axes(source_mol)
    rot_mat = get_180_rotation_matrix(s_axes[0])
    is_sym = is_redundant_flip(source_mol, np.dot(source_mol.positions, rot_mat))
    
    modes = ['orig'] if is_sym else ['orig', 'flip']
    if is_sym: print(f"  ✨ Symmetry Detected: '{s_prefix}' is symmetric. Skipping 'flip' mode.")

    # Find molecules in the SUPERCELL
    print("  🔍 Identifying molecules in the supercell...")
    all_mols = find_all_molecules(supercell)
    
    # Identify target formula from unit cell
    unit_mol, _ = reconstruct_molecule_pbc(unit_cell, target_example_idx)
    target_formula = unit_mol.get_chemical_formula()
    
    candidates = [m for m in all_mols if m['symbols'] == target_formula]
    total_sites = len(candidates)
    print(f"  📊 Found {total_sites} candidate sites for substitution.")

    # Loop over specified concentrations
    for pct in concentrations:
        # Calculate number of molecules to substitute
        count = int(round((pct / 100.0) * total_sites))
        
        # Ensure at least 1 molecule is substituted if pct > 0
        if count == 0 and pct > 0:
            print(f"  ⚠️ Warning: {pct}% of {total_sites} is < 1. Forcing 1 substitution.")
            count = 1
        
        if count > total_sites:
            count = total_sites
            
        print(f"\n  🎯 Processing Concentration: {pct}% ({count}/{total_sites} molecules)")
        
        selected = random.sample(candidates, count)
        remove_idx = [i for m in selected for i in m['indices']]
        
        # Create base structure by removing selected molecules
        base_struct = supercell[[i for i in range(len(supercell)) if i not in remove_idx]]

        for mode in modes:
            current = base_struct.copy()
            for m in selected:
                old_mol, _ = reconstruct_molecule_pbc(supercell, m['center_atom_idx'])
                t_axes = get_principal_axes(old_mol)
                align_rot = np.linalg.solve(s_axes, t_axes)
                
                new_mol = source_mol.copy()
                if mode == 'flip': new_mol.positions = np.dot(new_mol.positions, rot_mat)
                new_mol.positions = np.dot(new_mol.positions, align_rot) + old_mol.get_center_of_mass()
                current.extend(new_mol)
            
            # Stabilization
            current = run_pre_optimization(current)
            if wrap_coords: current.wrap(pbc=current.pbc)
            
            # File naming: Host_Size_Guest_Concentration_Mode.xyz
            out_name = f"{supercell_prefix}_{pct}pct_{mode}.xyz"
            write(out_name, current)
            print(f"  └─ Saved: {out_name}")

if __name__ == "__main__":
    # CONFIGURATION
    run_pipeline(
        target_file='ACRDIN04.xyz',                  # Host Unit Cell
        source_file='NICOAM01.cif',                  # Guest Molecule
        concentrations=[10, 25, 50],                 # List of concentrations (%)
        target_size=30.0,                            # Target Supercell Size in Angstroms
        target_example_idx=0,                        # Atom index to identify host molecule
        seed=42                                      # Random seed
    )
