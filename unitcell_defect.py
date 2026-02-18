import numpy as np
import random
import os
import re
import itertools
from scipy.spatial.distance import cdist
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.geometry import find_mic
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE

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

def check_structure_match(mol_ref, pos_target, tol=0.5):
    """Check if two structures are geometrically identical."""
    numbers = mol_ref.get_atomic_numbers()
    unique_nums = np.unique(numbers)
    for num in unique_nums:
        mask = (numbers == num)
        ref_p = mol_ref.positions[mask]
        target_p = pos_target[mask]
        min_dists = np.min(cdist(target_p, ref_p), axis=1)
        if np.any(min_dists > tol): return False 
    return True

# =========================================================
# 3. AI Auto-Decision Logic
# =========================================================
def decide_strategy_automatically(mol, tolerance=1.0):
    """
    Analyze molecular symmetry to select the best substitution strategy.
    
    Logic:
    1. Rotate molecule 180 degrees around principal axes.
    2. Check similarity to original shape (RMSD-like score).
    3. Low error -> Symmetric -> Use 'exhaust' (explore all combos).
    4. High error -> Asymmetric -> Use 'best_fit' (single best match).
    """
    mol_pos = mol.positions
    mol_pos_centered = mol_pos - mol.get_center_of_mass()
    s_axes = get_principal_axes(mol)
    
    # Check rotation around Primary(0) and Secondary(1) axes
    test_axes = [s_axes[0], s_axes[1]]
    
    for i, axis in enumerate(test_axes):
        rot = get_180_rotation_matrix(axis)
        pos_rotated = np.dot(mol_pos_centered, rot)
        
        # Calculate shape mismatch score
        dists = cdist(pos_rotated, mol_pos_centered)
        mismatch_score = np.sum(np.min(dists, axis=1))
        avg_error = mismatch_score / len(mol)
        
        if avg_error < tolerance:
            print(f"  🧠 AI Analysis: Molecule is SYMMETRIC (Avg Error: {avg_error:.2f} A).")
            return 'exhaust'
            
    print(f"  🧠 AI Analysis: Molecule is ASYMMETRIC (Avg Error > {tolerance} A).")
    return 'best_fit'

# =========================================================
# 4. Substitution Strategies
# =========================================================

# Strategy A: Best Fit (For Complex/Asymmetric Molecules)
def get_best_shape_match_rotation(source_mol, target_mol):
    s_axes = get_principal_axes(source_mol)
    t_axes = get_principal_axes(target_mol)
    t_com = target_mol.get_center_of_mass()
    t_pos_centered = target_mol.positions - t_com
    
    # Test 4 flip combinations to solve sign ambiguity
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
    return best_rot, min_score

# Strategy B: Exhaustive Modes (For Symmetric/Quasi-Symmetric Molecules)
def get_unique_modes(source_mol):
    s_axes = get_principal_axes(source_mol)
    axes_to_check = [('orig', None), 
                     ('flip_ax0', s_axes[0]), 
                     ('flip_ax1', s_axes[1]), 
                     ('flip_ax2', s_axes[2])]
    accepted_modes = {}
    
    for name, axis in axes_to_check:
        if axis is None:
            pos = source_mol.positions.copy()
        else:
            rot = get_180_rotation_matrix(axis)
            pos = np.dot(source_mol.positions, rot)
            
        # Check for redundancy
        is_unique = True
        temp_mol = source_mol.copy()
        for exist_name, exist_pos in accepted_modes.items():
            temp_mol.positions = exist_pos
            if check_structure_match(temp_mol, pos, tol=0.5):
                is_unique = False
                break
        
        if is_unique:
            accepted_modes[name] = pos
    return accepted_modes

# =========================================================
# 5. Main Pipeline
# =========================================================
def run_ultimate_pipeline(target_file, source_file, concentrations, strategy='auto', target_example_idx=0, wrap_coords=True, max_configs=2000):
    
    t_prefix = os.path.basename(target_file).split('.')[0]
    s_prefix = re.sub(r'\d+$', '', os.path.splitext(os.path.basename(source_file))[0])
    print(f"--- [Ultimate Pipeline] {t_prefix} + {s_prefix} ---")

    unit_cell = read(target_file)
    source_atoms = read(source_file)
    source_mol, _ = reconstruct_molecule_pbc(source_atoms, 0)
    source_mol.positions -= source_mol.get_center_of_mass()
    s_axes = get_principal_axes(source_mol)

    # 1. Identify Target Sites
    all_mols = find_all_molecules(unit_cell)
    target_sample, _ = reconstruct_molecule_pbc(unit_cell, target_example_idx)
    target_formula = target_sample.get_chemical_formula()
    candidates = [m for m in all_mols if m['symbols'] == target_formula]
    total_sites = len(candidates)
    print(f"  📊 Sites: {total_sites}")

    # 2. Auto-Detect Strategy
    if strategy == 'auto':
        print(f"  🤖 AI Auto-Detecting Strategy for '{s_prefix}'...")
        detected = decide_strategy_automatically(source_mol)
        print(f"     👉 Decision: {detected.upper()}")
        strategy = detected
    else:
        print(f"  👉 Strategy: {strategy.upper()} (User Forced)")

    # 3. Mode Preparation
    unique_modes_data = {}
    mode_names = []
    
    if strategy == 'exhaust':
        unique_modes_data = get_unique_modes(source_mol)
        mode_names = list(unique_modes_data.keys())
        print(f"  🔍 Found {len(mode_names)} Unique Modes: {mode_names}")
    else:
        print(f"  🔍 Best Fit Mode: Dynamically calculated per site.")

    # 4. Generation Loop
    total_generated = 0
    for pct in concentrations:
        k = int(round((pct / 100.0) * total_sites))
        if k == 0 and pct > 0: k = 1
        
        print(f"\n  🎯 Concentration: {pct}% ({k} molecules)")
        
        # Position Combinations
        site_combos = itertools.combinations(candidates, k)
        
        for config_idx, selected_sites in enumerate(site_combos):
            if total_generated >= max_configs: break
            selected_sites = list(selected_sites)
            
            # --- BRANCH: EXHAUSTIVE STRATEGY ---
            if strategy == 'exhaust':
                # Generate all permutations of orientations for selected sites
                mode_perms = itertools.product(mode_names, repeat=k)
                for perm_idx, selected_modes in enumerate(mode_perms):
                    if total_generated >= max_configs: break
                    
                    remove_idx = [i for m in selected_sites for i in m['indices']]
                    current = unit_cell[[i for i in range(len(unit_cell)) if i not in remove_idx]].copy()
                    
                    for m, mode_name in zip(selected_sites, selected_modes):
                        old_mol, _ = reconstruct_molecule_pbc(unit_cell, m['center_atom_idx'])
                        t_axes = get_principal_axes(old_mol)
                        align_rot = np.linalg.solve(s_axes, t_axes)
                        
                        new_mol = source_mol.copy()
                        new_mol.positions = unique_modes_data[mode_name].copy()
                        new_mol.positions = np.dot(new_mol.positions, align_rot) + old_mol.get_center_of_mass()
                        current.extend(new_mol)
                    
                    current = run_pre_optimization(current)
                    if wrap_coords: current.wrap(pbc=current.pbc)
                    out_name = f"{t_prefix}_{s_prefix}_{pct}pct_conf{config_idx+1:03d}_perm{perm_idx+1:03d}.xyz"
                    write(out_name, current)
                    total_generated += 1

            # --- BRANCH: BEST FIT STRATEGY ---
            elif strategy == 'best_fit':
                # Use only the single best orientation for each site
                remove_idx = [i for m in selected_sites for i in m['indices']]
                current = unit_cell[[i for i in range(len(unit_cell)) if i not in remove_idx]].copy()
                
                scores = []
                for m in selected_sites:
                    old_mol, _ = reconstruct_molecule_pbc(unit_cell, m['center_atom_idx'])
                    best_rot, score = get_best_shape_match_rotation(source_mol, old_mol)
                    scores.append(score)
                    
                    new_mol = source_mol.copy()
                    new_mol.positions = np.dot(new_mol.positions, best_rot) + old_mol.get_center_of_mass()
                    current.extend(new_mol)
                
                current = run_pre_optimization(current)
                if wrap_coords: current.wrap(pbc=current.pbc)
                out_name = f"{t_prefix}_{s_prefix}_{pct}pct_conf{config_idx+1:03d}_BestFit.xyz"
                write(out_name, current)
                total_generated += 1
                if total_generated % 10 == 0: print(f"    ..Gen {total_generated}, Score: {np.mean(scores):.2f}")

    print(f"\n  🎉 Total Generated: {total_generated}")

if __name__ == "__main__":
    run_ultimate_pipeline(
        target_file='DM1.xyz',    
        source_file='27DH.xyz',    
        concentrations=[25], 
        strategy='auto',  # Set to 'auto', 'exhaust', or 'best_fit'
        target_example_idx=0,          
        max_configs=2000 
    )
