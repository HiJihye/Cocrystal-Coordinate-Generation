import numpy as np
import os
import re
import argparse
import itertools
from scipy.spatial.distance import cdist
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.geometry import find_mic
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE

# =========================================================
# 1. Soft Repulsive Potential (For resolving steric clashes)
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
    """Executes a short relaxation to resolve atomic overlaps."""
    old_calc = atoms.calc
    atoms.calc = SoftRepulsivePotential()
    dyn = FIRE(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    atoms.calc = old_calc
    return atoms

# =========================================================
# 2. Geometric Utilities
# =========================================================
def reconstruct_molecule_pbc(atoms, start_idx):
    """Reconstructs molecules split across Periodic Boundary Conditions (PBC)."""
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
    u = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(u, u) - np.eye(3)

def check_structure_match(mol_ref, pos_target, tol=0.5):
    """
    Checks if pos_target is structurally identical to mol_ref.
    ** Ignores Hydrogen atoms (Atomic Number 1) during comparison. **
    """
    numbers = mol_ref.get_atomic_numbers()
    non_h_mask = (numbers != 1)
    
    # If the molecule is pure Hydrogen (e.g., H2), use all atoms
    if np.sum(non_h_mask) == 0: non_h_mask = (numbers == 1)
    
    ref_nums_filtered = numbers[non_h_mask]
    ref_pos_filtered = mol_ref.positions[non_h_mask]
    target_pos_filtered = pos_target[non_h_mask]
    
    unique_nums = np.unique(ref_nums_filtered)
    for num in unique_nums:
        mask_sub = (ref_nums_filtered == num)
        ref_p = ref_pos_filtered[mask_sub]
        target_p = target_pos_filtered[mask_sub]
        
        # Use cdist for order-independent distance check
        min_dists = np.min(cdist(target_p, ref_p), axis=1)
        if np.any(min_dists > tol): return False 
    return True

def find_all_molecules(atoms):
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
# 3. Main Pipeline with Mixed Mode Generation
# =========================================================
def run_rigorous_mixed_mode_pipeline(target_file, source_file, concentrations, target_example_idx=0, wrap_coords=True, max_configs=2000):
    
    # 1. File Setup
    t_prefix = os.path.basename(target_file).split('.')[0]
    s_prefix = re.sub(r'\d+$', '', os.path.splitext(os.path.basename(source_file))[0])
    
    print(f"--- [Rigorous Mixed-Mode] Target: {t_prefix} + Source: {s_prefix} ---")
    
    unit_cell = read(target_file)
    
    # 2. Source Preparation & Unique Orientation Discovery
    source_atoms = read(source_file)
    source_mol, _ = reconstruct_molecule_pbc(source_atoms, 0)
    source_mol.positions -= source_mol.get_center_of_mass()
    s_axes = get_principal_axes(source_mol)
    
    # Identify unique modes (ignoring Hydrogens)
    accepted_modes = {'orig': source_mol.positions.copy()}
    
    # Check all 3 principal axes: Primary(0), Secondary(1), Normal(2)
    axes_to_check = [('flip_ax0', s_axes[0]), ('flip_ax1', s_axes[1]), ('flip_ax2', s_axes[2])]
    
    print("  🔍 Identifying Unique Orientation Modes:")
    for mode_name, axis in axes_to_check:
        rot_mat = get_180_rotation_matrix(axis)
        pos_rotated = np.dot(source_mol.positions, rot_mat)
        
        is_unique = True
        for existing_name, existing_pos in accepted_modes.items():
            temp_mol = source_mol.copy(); temp_mol.positions = existing_pos
            if check_structure_match(temp_mol, pos_rotated, tol=0.5):
                print(f"     '{mode_name}' is redundant with '{existing_name}' -> Skipped.")
                is_unique = False; break
        
        if is_unique:
            print(f"     '{mode_name}' -> Accepted.")
            accepted_modes[mode_name] = pos_rotated

    mode_names = list(accepted_modes.keys())
    print(f"  ✅ Active Modes: {mode_names}")

    # 3. Identify Candidate Sites in Host
    all_mols = find_all_molecules(unit_cell)
    target_mol_sample, _ = reconstruct_molecule_pbc(unit_cell, target_example_idx)
    target_formula = target_mol_sample.get_chemical_formula()
    candidates = [m for m in all_mols if m['symbols'] == target_formula]
    total_sites = len(candidates)
    print(f"  📊 Available Sites: {total_sites}")

    # 4. Rigorous Generation Loop
    total_generated = 0
    
    for pct in concentrations:
        k = int(round((pct / 100.0) * total_sites))
        if k == 0 and pct > 0: k = 1
        if k > total_sites: k = total_sites
        
        print(f"\n  🎯 Concentration: {pct}% ({k} molecules)")
        
        # Step A: Select Positions (itertools.combinations)
        site_combinations = itertools.combinations(candidates, k)
        
        for config_idx, selected_sites in enumerate(site_combinations):
            selected_sites = list(selected_sites)
            
            # Step B: Select Orientation Modes for these positions (itertools.product)
            # This accounts for independent orientations for each substituted site.
            mode_combinations = itertools.product(mode_names, repeat=k)
            
            for perm_idx, selected_modes in enumerate(mode_combinations):
                if total_generated >= max_configs:
                    print(f"     ⚠️ Global limit ({max_configs}) reached. Stopping.")
                    return

                # Build the new structure
                remove_idx = [i for m in selected_sites for i in m['indices']]
                current = unit_cell[[i for i in range(len(unit_cell)) if i not in remove_idx]].copy()
                
                # Apply specific mode to specific site
                for m, mode_name in zip(selected_sites, selected_modes):
                    old_mol, _ = reconstruct_molecule_pbc(unit_cell, m['center_atom_idx'])
                    t_axes = get_principal_axes(old_mol)
                    align_rot = np.linalg.solve(s_axes, t_axes)
                    
                    new_mol = source_mol.copy()
                    new_mol.positions = accepted_modes[mode_name].copy() # Apply orientation
                    new_mol.positions = np.dot(new_mol.positions, align_rot) + old_mol.get_center_of_mass()
                    current.extend(new_mol)
                
                # Stabilization (Pre-optimization)
                current = run_pre_optimization(current)
                if wrap_coords: current.wrap(pbc=current.pbc)
                
                # Filename: Indicates Config ID (position) and Permutation ID (orientation)
                out_name = f"{t_prefix}_{s_prefix}_{pct}pct_conf{config_idx+1:03d}_perm{perm_idx+1:03d}.xyz"
                write(out_name, current)
                
                total_generated += 1
                if total_generated % 50 == 0:
                    print(f"     ... Generated {total_generated} structures.")

    print(f"\n  🎉 Done. Total structures generated: {total_generated}")

if __name__ == "__main__":
    run_rigorous_mixed_mode_pipeline(
        target_file='ACRDIN.xyz',
        source_file='27DH.xyz',
        concentrations=[25, 50],
        target_example_idx=0,
        max_configs=2000 # Safety limit
    )
