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
    old_calc = atoms.calc
    atoms.calc = SoftRepulsivePotential()
    dyn = FIRE(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    
    max_f = np.sqrt((atoms.get_forces()**2).sum(axis=1).max())
    atoms.calc = old_calc
    print(f"  ⚙️ [Pre-Opt] Done. (Max Force: {max_f:.4f} eV/A)")
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
        curr_pos = work_atoms.positions[current_idx]
        for n_idx in nl.get_neighbors(current_idx)[0]:
            if n_idx not in visited:
                visited.add(n_idx); queue.append(n_idx); cluster_indices.append(n_idx)
                vec, _ = find_mic(work_atoms.positions[n_idx] - curr_pos, work_atoms.cell, work_atoms.pbc)
                work_atoms.positions[n_idx] = curr_pos + vec
    return work_atoms[cluster_indices], cluster_indices

def get_principal_axes(mol):
    vals, vecs = mol.get_moments_of_inertia(vectors=True)
    return vecs[np.argsort(vals)]

def get_180_rotation_matrix(axis):
    u = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(u, u) - np.eye(3)

def is_redundant_flip(mol_orig, pos_flip, tol=0.5):
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
def run_series_substitution(target_file, source_file, target_example_idx=0, wrap_coords=True, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    
    t_prefix = os.path.basename(target_file).split('_')[0]
    s_prefix = re.sub(r'\d+$', '', os.path.splitext(os.path.basename(source_file))[0])
    final_prefix = f"{t_prefix}_{s_prefix}"
    
    print(f"--- [Pipeline] Processing: {t_prefix} + {s_prefix} ---")

    target_atoms = read(target_file)
    source_atoms = read(source_file)
    
    source_mol, _ = reconstruct_molecule_pbc(source_atoms, 0)
    source_mol.positions -= source_mol.get_center_of_mass()
    
    s_axes = get_principal_axes(source_mol)
    rot_mat = get_180_rotation_matrix(s_axes[0])
    is_sym = is_redundant_flip(source_mol, np.dot(source_mol.positions, rot_mat))
    
    modes = ['orig'] if is_sym else ['orig', 'flip']
    if is_sym: print(f"✨ Symmetry Detected: '{s_prefix}' is symmetric. Skipping 'flip' mode.")

    all_mols = find_all_molecules(target_atoms)
    target_formula = reconstruct_molecule_pbc(target_atoms, target_example_idx)[0].get_chemical_formula()
    candidates = [m for m in all_mols if m['symbols'] == target_formula]
    total_sites = len(candidates)
    
    for count in range(1, total_sites // 2 + 1):
        concentration = (count / total_sites) * 100
        
        selected = random.sample(candidates, count)
        remove_idx = [i for m in selected for i in m['indices']]
        base_struct = target_atoms[[i for i in range(len(target_atoms)) if i not in remove_idx]]

        for mode in modes:
            current = base_struct.copy()
            for m in selected:
                old_mol, _ = reconstruct_molecule_pbc(target_atoms, m['center_atom_idx'])
                t_axes = get_principal_axes(old_mol)
                align_rot = np.linalg.solve(s_axes, t_axes)
                
                new_mol = source_mol.copy()
                if mode == 'flip': new_mol.positions = np.dot(new_mol.positions, rot_mat)
                new_mol.positions = np.dot(new_mol.positions, align_rot) + old_mol.get_center_of_mass()
                current.extend(new_mol)
            
            current = run_pre_optimization(current)
            if wrap_coords: current.wrap(pbc=current.pbc)
            
            out_name = f"{final_prefix}_{concentration:.0f}%_{mode}.xyz"
            write(out_name, current)
            print(f"  └─ Saved: {out_name} ({count}/{total_sites} molecules)")

if __name__ == "__main__":
    run_series_substitution(
        target_file='ACRDIN04_vasp_output_0030.xyz', 
        source_file='NICOAM01.cif',
        target_example_idx=0,
        seed=42
    )
