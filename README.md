# Automated Molecular Substitution Toolkit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ASE](https://img.shields.io/badge/Library-ASE-green)

A high-performance Python toolkit for generating doped crystal structures by substituting host molecules with guest molecules. It automates the entire pipeline: from symmetry analysis to steric clash resolution.

This repository provides two specialized tools depending on your simulation needs:

1. automated_unitcell_defect.py: A Combinatorial Generator for exhaustive search in unit cells.
2. automated_supercell_defect.py: A Statistical Generator for realistic distribution in large supercells.
---

## 🚀 Key Features
### 1. AI-Driven Strategy Auto-Detection: The toolkit automatically analyzes the geometric symmetry of the guest molecule to determine the best substitution strategy

* Symmetric Molecules (e.g., Anthracene):

  - Detected as EXHAUST (Unit Cell) or RANDOM_FLIP (Supercell).
  - Explores orientation flips (e.g., 0° vs 180°) since they might be energetically distinct in the lattice.
  - Uses a distance-based check (`cdist`) to identify symmetric molecules. It automatically skips redundant **"flip"** orientations if the rotated structure is geometrically identical to the original (within 0.5 Å) except for hydrogen atoms.

* Asymmetric Molecules (e.g., Cypermethrin):

  - Detected as BEST_FIT.
  - Performs a rigorous 4-way shape matching scan to find the single best alignment, preventing unphysical "head-to-tail" flipping errors.

### 2. Physics-Aware Steric Clash Resolution: Doping often creates atomic overlaps. We solve this using a custom physics engine

* Soft Repulsive Potential: A custom ASE calculator ($E \propto (r_c - r)^3$) that gently pushes overlapping atoms apart without energy explosion (unlike Lennard-Jones).
* FIRE Optimizer: Rapidly relaxes the geometry to a physically reasonable state (Force < 0.1 eV/Å), ensuring valid inputs for DFT calculations.

### 3. Maximin Distance Sampling (Supercell Only): For large-scale doping, random placement is inefficient.

* The Maximin Algorithm iteratively selects substitution sites that maximize the distance to the nearest existing dopant.
* It accounts for Periodic Boundary Conditions (PBC) to ensure a truly uniform spatial distribution.

---

## 🛠 Detailed Workflow

### 1. Molecular Reconstruction & Identification
The script identifies all unique molecules in the `target_file`. It uses a Breadth-First Search (BFS) combined with MIC to reconstruct molecules that wrap around unit cell boundaries, ensuring accurate geometric centers are calculated.

### 2. Orientation & Principal Axes
To ensure the guest molecule fits the host site:
* **Center of Mass (COM)** is set to $(0,0,0)$.
* **Principal Axes** are calculated via the inertia tensor eigen-decomposition.
* A **Rotation Matrix** is applied to map the guest's axes onto the host's axes.

### 3. Symmetry Check (The "Flip" Logic)
To account for molecular dipoles (e.g., **Nicotinamide**), the script generates two orientations:
1.  **`orig`**: Direct alignment.
2.  **`flip`**: $180^\circ$ rotation around the primary principal axis.

**Intelligence:** The script compares the `orig` and `flip` structures using `scipy.spatial.distance.cdist`.
* If they match (tolerance < 0.5 Å), the molecule is flagged as **Symmetric**, and the `flip` mode is skipped.
* If they differ (e.g., asymmetric molecules), both files are saved.

### 4. Overlap Resolution
Substituted molecules often clash with neighbors. The script applies a cubic soft potential:
$$V(r) = \frac{k}{3}(d_{cut} - r)^3 \quad \text{for } r < d_{cut}$$
The system is relaxed until the maximum force is below **0.1 eV/Å**.

---

## 💻 Script Comparison & Usage

### 1. Rigorous Combinatorial Tool (automated_unitcell_defect.py)
* Use this when: You need to study every possible configuration (permutations of sites and orientations) in a small system. Ideal for finding the thermodynamic ground state of a defect.
* Logic: $\binom{N}{k}$ Site Combinations $\times$ Orientation Permutations.
* Output: Generates many files (e.g., ...conf001_perm001.xyz, ...conf001_perm002.xyz).

```bash
# Open automated_unitcell_defect.py and edit the bottom block:
if __name__ == "__main__":
    run_ultimate_pipeline(
        target_file='host_unitcell.xyz',
        source_file='guest_molecule.xyz',
        concentrations=[25, 50],       # Target concentrations (%)
        strategy='auto',               # 'auto' (Recommended), 'exhaust', or 'best_fit'
        max_configs=2000               # Limit total generated structures
    )
```

### Realistic Supercell Tool (automated_supercell_defect.py)

* Use this when: You need to prepare large-scale models for MD or DFT with a specific doping concentration. It focuses on creating a spatially uniform distribution.
* Logic: Unit Cell Expansion $\rightarrow$ Maximin Sampling $\rightarrow$ Best Fit Alignment.
* Output: Generates minimized files with optimal spacing (e.g., ...3x3x3_25pct_best_fit.xyz).

```bash
# Open automated_supercell_defect.py and edit the bottom block:
if __name__ == "__main__":
    run_supercell_pipeline(
        target_file='host_unitcell.xyz',
        source_file='guest_molecule.xyz',
        concentrations=[10, 25],       # Target concentrations (%)
        target_size=30.0,              # Target Supercell Size in Angstroms (e.g., 30Å)
        target_example_idx=0,          # Atom index to identify the host molecule
        seed=42                        # For reproducible random sampling
    )
```

---
## 🧩 Algorithm Details
### The "Best Fit" Alignment
* For complex asymmetric molecules, simple inertial alignment is risky due to sign ambiguity ($v$ vs $-v$).
1. Simulates 4 possible principal axis flips: [1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1].
2. Calculates the Chamfer Distance between the rotated guest and the host site.
3. Selects the rotation matrix that minimizes shape error.
### Soft Repulsive Potential
* To resolve overlaps:
  
    $$V(r) = \frac{k}{3}(d_{cut} - r)^3 \quad \text{for } r < d_{cut}$$

  This potential is purely repulsive and bounded, making it numerically stable for high-overlap initial states.

---

## 📦 Requirements

* Python 3.8+
* ASE (Atomic Simulation Environment)
* NumPy
* SciPy

```bash
pip install ase numpy scipy
```
