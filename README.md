# Molecular Substitution & Steric Resolution Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ASE](https://img.shields.io/badge/Library-ASE-green)

A high-performance Python pipeline for substituting molecules within crystal lattices. It automates the generation of doped structures with varying concentrations while handling periodic boundary conditions, molecular orientation, and steric clash resolution.

---

## 🚀 Key Features

* **PBC-Aware Reconstruction**: Automatically reconstructs molecules split across periodic boundary conditions using the **Minimum Image Convention (MIC)**.
* **Inertial Frame Alignment**: Aligns the guest molecule's principal axes ($I_p$) to the host molecule's orientation to preserve the local lattice structure.
* **Robust Symmetry Detection**: Uses a distance-based check (`cdist`) to identify symmetric molecules. It automatically skips redundant **"flip"** orientations if the rotated structure is geometrically identical to the original (within 0.5 Å).
* **Steric Resolution (Pre-Opt)**: Resolves atomic overlaps using a **Soft Repulsive Potential** and **FIRE** optimizer, preventing "exploding" atoms in subsequent DFT calculations.
* **Concentration-Based Naming**: Files are automatically named based on the substitution percentage (e.g., `25%`, `50%`).

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

## 📦 Requirements

* Python 3.8+
* ASE (Atomic Simulation Environment)
* NumPy
* SciPy

```bash
pip install ase numpy scipy
```

## 💻. Usage

Modify the `if __name__ == "__main__":` block to set your input files. The script will automatically calculate the maximum possible substitution (up to 50% of candidates) and generate the series.

| Parameter | Description |
| :--- | :--- |
| `target_file` | Host crystal structure. |
| `source_file` | Guest molecule to insert. |
