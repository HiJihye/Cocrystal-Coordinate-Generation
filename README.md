# Molecular Substitution & Overlap Resolution Pipeline

This Python-based pipeline automates the process of substituting molecules within a crystal lattice while maintaining structural integrity. It handles periodic boundary conditions (PBC), molecular orientation via principal axis alignment, and resolves steric clashes using a custom soft-repulsive potential.

---

## 1. Features

* **PBC-Aware Reconstruction**: Automatically detects and repairs molecules "broken" across unit cell boundaries using the Minimum Image Convention (MIC).
* **Principal Axis Alignment**: Maps the source molecule onto the target site by aligning their respective inertial frames.
* **Symmetry-Aware Flipping**: 
    * Generates both `orig` and `flip` (180° rotation) orientations.
    * Includes a **Symmetry Check** to skip redundant `flip` structures if the molecule is centrosymmetric or symmetric along its primary axis.
* **Soft-Repulsive Pre-Optimization**: Implements a custom ASE `Calculator` to "push" overlapping atoms apart before saving, ensuring the structures are ready for DFT or higher-level force-field calculations.

---

## 2. Requirements

This pipeline requires Python 3.x and the following libraries:
* **ASE** (Atomic Simulation Environment)
* **NumPy**

```bash
pip install ase numpy
```

---

## 3. Workflow Details

The substitution process follows a four-stage geometric and physical protocol:

### A. PBC-Aware Molecular Reconstruction
In periodic systems, molecules are often "broken" across unit cell boundaries. 
* The script utilizes the **Minimum Image Convention (MIC)** to determine the shortest distance between atoms across periodic boundaries.
* It performs a Breadth-First Search (BFS) using a `NeighborList` to group all atoms belonging to a single molecule, shifting their positions into a continuous cluster.

### B. Inertial Frame & Orientation Alignment
To ensure the new guest molecule (source) is oriented exactly like the host molecule (target), the script aligns their **Principal Axes of Inertia**:
1.  **Inertia Tensor Calculation**: Computes the tensor $I_{ij} = \sum m_a (\delta_{ij} r_a^2 - r_{ai} r_{aj})$.
2.  **Diagonalization**: Extracts the eigenvectors (principal axes) $v_1, v_2, v_3$.
3.  **Rotation Matrix**: A transformation matrix is derived to map the source's coordinate system onto the target's coordinate system.

### C. Symmetry-Based Flip Logic
For non-centrosymmetric molecules, orientation matters.
* The script generates an alternative orientation by rotating the molecule **180°** around its primary principal axis ($I_p[0]$).
* **Redundancy Check**: If the flipped coordinates match the original coordinates (within a $10^{-2}$ tolerance), the molecule is flagged as symmetric, and the redundant "flip" mode is skipped to save computational resources.

### D. Steric Resolution via Soft-Repulsive Potential
Newly inserted molecules may overlap with the existing lattice. We resolve this using a custom **SoftRepulsivePotential**:
* **Potential Form**: 
  $$V(r) = \frac{k}{3}(d_{cut} - r)^3 \quad \text{for } r < d_{cut}$$
* **Optimization**: The **FIRE (Fast Inertial Relaxation Engine)** algorithm is used to minimize this "overlap energy," gently pushing atoms apart until the maximum force is below $0.1$ eV/Å.

---

## 4. Usage

### Configuration
The pipeline is executed through the `run_series_substitution_optimized` function. You only need to modify the arguments in the `__main__` block.

```python
run_series_substitution_optimized(
    target_file='host_structure.xyz',   # The crystal lattice file (e.g., .xyz, .vasp)
    source_file='guest_molecule.cif',  # The molecule to be inserted (.cif, .xyz)
)
