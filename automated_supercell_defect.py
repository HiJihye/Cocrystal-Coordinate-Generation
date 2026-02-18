import numpy as np
import random
import os
import re
import argparse
from scipy.spatial.distance import cdist
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.geometry import find_mic
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE
from ase.build import make_supercell

# =========================================================
# 1. Soft Repulsive Potential (Steric Clash Resolver)
# =========================================================
class SoftRepulsivePotential(Calculator):
    """
    Custom calculator to resolve atomic overlaps without explosion.
    Uses a soft cubic potential: E ~ (rc - r)^3
    """
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
    """Run FIRE optimizer with Soft Potential to relax structure."""
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
    """Reconstruct molecules broken by Periodic Boundary Conditions."""
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
    """Calculate principal axes of inertia."""
    vals, vecs = mol.get_moments_of_inertia(vectors=True)
    return vecs[np.argsort(vals)]

def get_180_rotation_matrix(axis):
    """Get rotation matrix for 180-degree rotation around an axis."""
    u = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(u, u) - np.eye(3)

def find_all_molecules(atoms):
    """Identify all unique molecules in the system."""
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
# 3. Smart Strategy & Alignment Logic
# =========================================================
def decide_strategy_automatically(mol, tolerance=1.0):
    """
    Analyze symmetry to select strategy.
    """
    mol_pos = mol.positions
    mol_pos_centered = mol_pos - mol.get_center_of_mass()
    s_axes = get_principal_axes(mol)
    
    test_axes = [s_axes[0], s_axes[1]]
    
    for axis in test_axes:
        rot = get_180_rotation_matrix(axis)
        pos_rotated = np.dot(mol_pos_centered, rot)
        
        dists = cdist(pos_rotated, mol_pos_centered)
        mismatch_score = np.sum(np.min(dists, axis=1))
        avg_error = mismatch_score / len(mol)
        
        if avg_error < tolerance:
            print(f"  🧠 AI Analysis: Molecule is SYMMETRIC (Avg Error: {avg_error:.2f} A).")
            return 'random_flip'
            
    print(f"  🧠 AI Analysis: Molecule is ASYMMETRIC (Avg Error > {tolerance} A).")
    return 'best_fit'

def get_best_shape_match_rotation(source_mol, target_mol):
    """Finds the single best rotation to match target shape."""
    s_axes = get_principal_axes(source_mol)
    t_axes = get_principal_axes(target_mol)
    t_com = target_mol.get_center_of_mass()
    t_pos_centered = target_mol.positions - t_com
    
    flips = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    best_rot, min_score = None, float('inf')
    
    for f in flips:
        t_axes_flipped = t_axes * np.array(f)[:, np.newaxis]
        rot_candidate = np.linalg.solve(s_axes, t_axes_flipped)
        pos_rotated = np.dot(source_mol.positions, rot_candidate)
        dists = cdist(pos_rotated, t_pos_centered)
        score = np.sum(np.min(dists, axis=1))
        
        if score < min_score:
            min_score = score
            best_rot = rot_candidate
    return best_rot

# =========================================================
# [CORE] Maximin Distance Selector
# =========================================================
def select_farthest_sites(candidates, count, supercell):
    """
    Selects sites using Maximin algorithm to maximize spacing.
    Also returns the minimum distance in the final selection.
    """
    if count >= len(candidates):
        return candidates, 0.0
    
    # 1. Start with a random site
    first_choice_idx = random.randint(0, len(candidates) - 1)
    selected_indices = [first_choice_idx]
    
    # 2. Iteratively add the site farthest from existing set
    while len(selected_indices) < count:
        max_min_dist = -1.0
        best_candidate_idx = -1
        
        # Current centers (atoms indices)
        selected_atom_indices = [candidates[i]['center_atom_idx'] for i in selected_indices]
        
        for i in range(len(candidates)):
            if i in selected_indices: continue
                
            curr_atom_idx = candidates[i]['center_atom_idx']
            
            # Distance to ALL selected sites (with PBC)
            dists = supercell.get_distances(curr_atom_idx, selected_atom_indices, mic=True, vector=False)
            
            # The closest neighbor to THIS candidate
            min_dist_to_existing = np.min(dists)
            
            # We want the candidate whose closest neighbor is farthest away
            if min_dist_to_existing > max_min_dist:
                max_min_dist = min_dist_to_existing
                best_candidate_idx = i
        
        selected_indices.append(best_candidate_idx)
    
    # Calculate final minimum distance among selected
    final_atom_indices = [candidates[i]['center_atom_idx'] for i in selected_indices]
    final_dists = []
    for i in range(len(final_atom_indices)):
        for j in range(i + 1, len(final_atom_indices)):
            d = supercell.get_distance(final_atom_indices[i], final_atom_indices[j], mic=True)
            final_dists.append(d)
    
    min_separation = min(final_dists) if final_dists else 0.0
    
    return [candidates[i] for i in selected_indices], min_separation

# =========================================================
# 4. Supercell Generation Logic
# =========================================================
def generate_supercell(atoms, target_size=30.0):
    """Create a supercell approximating the target size (Angstroms)."""
    cell_lengths = atoms.cell.lengths()
    nx = max(1, int(round(target_size / cell_lengths[0])))
    ny = max(1, int(round(target_size / cell_lengths[1])))
    nz = max(1, int(round(target_size / cell_lengths[2])))
    print(f"  🏗️ Extending Cell: {nx}x{ny}x{nz} (Target: ~{target_size} A)")
    P = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]
    supercell = make_supercell(atoms, P)
    return supercell, (nx, ny, nz)

# =========================================================
# 5. Main Workflow
# =========================================================
def run_supercell_pipeline(target_file, source_file, concentrations, target_size=30.0, target_example_idx=0, wrap_coords=True, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    
    t_prefix = os.path.basename(target_file).split('.')[0]
    s_prefix = re.sub(r'\d+$', '', os.path.splitext(os.path.basename(source_file))[0])
    
    print(f"--- [Supercell Universal + MaxDist] {t_prefix} + {s_prefix} ---")

    # 1. Prepare Source Molecule
    source_atoms = read(source_file)
    source_mol, _ = reconstruct_molecule_pbc(source_atoms, 0)
    source_mol.positions -= source_mol.get_center_of_mass()
    s_axes = get_principal_axes(source_mol)
    
    # 2. Auto-Detect Strategy
    strategy = decide_strategy_automatically(source_mol)
    print(f"  👉 Strategy: {strategy.upper()}")

    # 3. Generate Supercell
    unit_cell = read(target_file)
    supercell, (nx, ny, nz) = generate_supercell(unit_cell, target_size)
    supercell_prefix = f"{t_prefix}_{nx}x{ny}x{nz}_{s_prefix}"

    # 4. Identify Substitution Sites
    print("  🔍 Identifying molecules in the supercell...")
    all_mols = find_all_molecules(supercell)
    
    unit_mol, _ = reconstruct_molecule_pbc(unit_cell, target_example_idx)
    target_formula = unit_mol.get_chemical_formula()
    
    candidates = [m for m in all_mols if m['symbols'] == target_formula]
    total_sites = len(candidates)
    print(f"  📊 Found {total_sites} candidate sites.")

    # 5. Substitution Loop
    for pct in concentrations:
        count = int(round((pct / 100.0) * total_sites))
        if count == 0 and pct > 0: count = 1
        if count > total_sites: count = total_sites
            
        print(f"\n  🎯 Concentration: {pct}% ({count}/{total_sites} molecules)")
        
        # [MAXIMIN] Select Optimal Sites
        selected, min_dist = select_farthest_sites(candidates, count, supercell)
        print(f"     ✅ Optimized Spacing: Min Distance = {min_dist:.2f} A")
        
        remove_idx = [i for m in selected for i in m['indices']]
        
        # Base structure
        base_struct = supercell[[i for i in range(len(supercell)) if i not in remove_idx]]

        modes_to_run = ['best_fit'] if strategy == 'best_fit' else ['orig', 'flip']
        
        for mode_name in modes_to_run:
            current = base_struct.copy()
            
            for m in selected:
                old_mol, _ = reconstruct_molecule_pbc(supercell, m['center_atom_idx'])
                new_mol = source_mol.copy()
                
                if strategy == 'best_fit':
                    best_rot = get_best_shape_match_rotation(source_mol, old_mol)
                    new_mol.positions = np.dot(new_mol.positions, best_rot)
                else: 
                    t_axes = get_principal_axes(old_mol)
                    align_rot = np.linalg.solve(s_axes, t_axes)
                    if mode_name == 'flip':
                        rot_mat = get_180_rotation_matrix(s_axes[0])
                        new_mol.positions = np.dot(new_mol.positions, rot_mat)
                    new_mol.positions = np.dot(new_mol.positions, align_rot)

                new_mol.positions += old_mol.get_center_of_mass()
                current.extend(new_mol)
            
            # Stabilization
            current = run_pre_optimization(current)
            if wrap_coords: current.wrap(pbc=current.pbc)
            
            # Save File
            out_name = f"{supercell_prefix}_{pct}pct_{mode_name}.xyz"
            write(out_name, current)
            print(f"  └─ Saved: {out_name}")

if __name__ == "__main__":
    run_supercell_pipeline(
        target_file='ACRDIN04.xyz',          
        source_file='27DH.xyz',                
        concentrations=[1, 5, 10],         
        target_size=30.0,                    
        target_example_idx=0,                
        seed=42                              
    )