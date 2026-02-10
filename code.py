import numpy as np
import random
import os
import re
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.geometry import find_mic
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE

# =========================================================
# 1. Custom Calculator for Overlap Resolution
# =========================================================
class SoftRepulsivePotential(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, cutoff=1.2, repulsion_strength=20.0, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.strength = repulsion_strength

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        forces = np.zeros_like(positions)
        energy = 0.0

        nl = NeighborList([self.cutoff] * len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)

        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                vec = positions[j] + np.dot(offset, cell) - positions[i]
                dist = np.linalg.norm(vec)
                if dist < self.cutoff and dist > 0.01:
                    delta = self.cutoff - dist
                    force_mag = self.strength * (delta ** 2) / dist
                    forces[i] -= force_mag * vec
                    energy += (self.strength / 3.0) * (delta ** 3)
        
        self.results['energy'] = energy
        self.results['forces'] = forces

def run_pre_optimization(atoms, fmax=0.1, steps=100):
    """ Executes relaxation and prints the actual final Max Force. """
    old_calc = atoms.calc
    atoms.calc = SoftRepulsivePotential()
    dyn = FIRE(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    
    # Calculate actual final max force for user verification
    final_forces = atoms.get_forces()
    max_f = np.sqrt((final_forces**2).sum(axis=1).max())
    
    atoms.calc = old_calc
    print(f"  ⚙️ [Pre-Opt] Done. (Actual Max Force: {max_f:.4f} eV/A)")
    return atoms

# =========================================================
# 2. Geometric Utilities
# =========================================================
def reconstruct_molecule_pbc(atoms, start_idx):
    cutoffs = [c * 1.25 for c in natural_cutoffs(atoms)]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    visited = {start_idx}; queue = [start_idx]; cluster_indices = [start_idx]
    work_atoms = atoms.copy()
    while queue:
        current_idx = queue.pop(0)
        current_pos = work_atoms.positions[current_idx]
        for n_idx in nl.get_neighbors(current_idx)[0]:
            if n_idx not in visited:
                visited.add(n_idx); queue.append(n_idx); cluster_indices.append(n_idx)
                vec = work_atoms.positions[n_idx] - current_pos
                vec, _ = find_mic(vec, work_atoms.cell, work_atoms.pbc)
                work_atoms.positions[n_idx] = current_pos + vec
    return work_atoms[cluster_indices], cluster_indices

def get_principal_axes(mol):
    vals, vecs = mol.get_moments_of_inertia(vectors=True)
    return vecs[np.argsort(vals)]

def get_180_rotation_matrix(axis):
    u = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(u, u) - np.eye(3)

def is_redundant_flip(pos_orig, pos_flip, tol=1e-2):
    """ 
    Checks if the flipped orientation is identical to the original one
    by comparing sorted coordinate sets.
    """
    s_orig = pos_orig[np.lexsort(pos_orig.T)]
    s_flip = pos_flip[np.lexsort(pos_flip.T)]
    return np.allclose(s_orig, s_flip, atol=tol)

def find_all_molecules(atoms):
    cutoffs = [c * 1.25 for c in natural_cutoffs(atoms)]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
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
# 3. Main Workflow
# =========================================================
def run_series_substitution_optimized(target_file, source_file, target_example_idx=0, wrap_coords=True, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    
    # Strict Prefix Naming
    target_prefix = os.path.basename(target_file).split('_')[0]
    source_base = os.path.splitext(os.path.basename(source_file))[0]
    source_prefix = re.sub(r'\d+$', '', source_base)
    final_prefix = f"{target_prefix}_{source_prefix}"
    
    print(f"--- [Pipeline] Prefix: {final_prefix} ---")

    target_atoms_origin = read(target_file)
    source_atoms = read(source_file)
    
    # Prepare Source Molecule
    source_mol, _ = reconstruct_molecule_pbc(source_atoms, 0)
    source_mol_centered = source_mol.copy()
    source_mol_centered.positions -= source_mol_centered.get_center_of_mass()
    
    # Flip Logic based on User's Ip[0] requirement
    source_axes = get_principal_axes(source_mol_centered)
    rotation_axis = source_axes[0] 
    flip_matrix = get_180_rotation_matrix(rotation_axis)
    
    # Symmetry Check: Is flipping redundant (e.g., Anthracene)?
    flipped_pos = np.dot(source_mol_centered.positions, flip_matrix)
    is_symmetric = is_redundant_flip(source_mol_centered.positions, flipped_pos)
    
    modes = ['orig'] if is_symmetric else ['orig', 'flip']
    if is_symmetric:
        print(f"✨ Symmetry Detected: Source molecule is symmetric about Ip[0]. Skipping 'flip' mode.")

    all_mols = find_all_molecules(target_atoms_origin)
    target_example_mol, _ = reconstruct_molecule_pbc(target_atoms_origin, target_example_idx)
    target_formula = target_example_mol.get_chemical_formula()
    
    candidates = [m for m in all_mols if m['symbols'] == target_formula]
    max_limit = len(candidates) // 2
    
    for count in range(1, max_limit + 1):
        selected_mols = random.sample(candidates, count)
        atoms_to_remove = set()
        for m in selected_mols: atoms_to_remove.update(m['indices'])
        
        remaining_indices = [i for i in range(len(target_atoms_origin)) if i not in atoms_to_remove]
        base_structure = target_atoms_origin[remaining_indices]

        for mode in modes:
            current_structure = base_structure.copy()
            for mol_info in selected_mols:
                old_mol, _ = reconstruct_molecule_pbc(target_atoms_origin, mol_info['center_atom_idx'])
                lattice_axes = get_principal_axes(old_mol)
                align_rot = np.linalg.solve(source_axes, lattice_axes)
                
                new_mol = source_mol_centered.copy()
                if mode == 'flip':
                    new_mol.positions = np.dot(new_mol.positions, flip_matrix)
                
                new_mol.positions = np.dot(new_mol.positions, align_rot) + old_mol.get_center_of_mass()
                current_structure.extend(new_mol)
            
            # Stabilization & Printing actual Force
            current_structure = run_pre_optimization(current_structure)
            if wrap_coords: current_structure.wrap(pbc=current_structure.pbc)
            
            out_name = f"{final_prefix}_{count:03d}_{mode}.xyz"
            write(out_name, current_structure)
            print(f"  └─ Saved: {out_name}")

    print(f"\n🚀 Pipeline completed successfully!")

if __name__ == "__main__":
    run_series_substitution_optimized(
        target_file='ACRDIN12_vasp_output_0030.xyz', 
        source_file='27DHN01.cif' # Or 'ANTCEN01.cif' for symmetry check
    )
