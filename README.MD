# MetaSolverS4

**MetaSolverS4** is a small, easy-to-run driver around the **S4** RCWA solver that uses a flexible YAML config and MPI to sweep wavelengths, incidence angles, and geometry variants. It computes **transmission** and **reflection** intensity and phase per diffraction order for s and p polarizations.

## Features

- YAML config (`s4_config.yaml`) with **value** / **choices** / **linspace** / **logspace** specs.
- Sweeps wavelength, angles, period *a*, layer thicknesses, and shape parameters.
- Geometry assembled from rectangles and ellipses; domain = superstrate / single metasurface layer / substrate.
- Material dispersion from the refractiveindex.info database (via the `refractiveindex` Python package).
- MPI parallelism (via `mpi4py`) that splits work by wavelength; geometry variants run sequentially.

## How it works (MPI)

S4’s Python bindings aren’t MPI-aware, so the code parallelizes the outer simulation loops:
- Rank 0 prints status and writes results.
- All ranks build the same geometry variant, then each rank processes a round-robin subset of wavelengths.
- Results are reduced (Allreduce) and saved as <filename>_<variant>.npz.

## Config

All simulation inputs live in `s4_config.yaml`. Key parts:
- lattice.a and each layer thickness accept value/choices/lin/log.
- sweep.wl, sweep.theta, sweep.phi control spectral/angle sweeps.
- materials: map material keys (e.g., Air, Meta, Sub) to refractiveindex.info datasets (or constants like source: n=1.0).
- shapes: list of rectangles/ellipses with center, size, rotation; fields support sweeps and simple ties.
- S4 parameters and options: to control simulation properties

## Usage
1) Edit s4_config.yaml to your liking.
2) Run with the bash script (16 processes)

    NP=16 ./run.sh 
    
3) Outputs: one NPZ per geometry variant
<filename>_<vidx>.npz containing:
- variant — Python dict of the exact geometry used (unwrap with .item()).
- T, R — float64 arrays, shape (2, num_orders, N_theta, N_wl): per-order intensity (fraction of incident power). First axis: 0 = s, 1 = p.
- T_phase, R_phase — float64 arrays (radians), same shape as above: per-order phase in [-π, π].
- wl — wavelength grid (µm), shape (N_wl,).
- theta — incidence polar angles (deg), shape (N_theta,).
- phi — azimuth angles (deg).
- orders — list of (m, n) diffraction orders (same ordering as S4’s GetBasisSet()).
- materials — dict with material specs used to build ε(λ).

## Requirements (tested pins)

- Python 3.11
- NumPy 1.26 (pinned; S4 wheels from phoeby expect this)
- S4 (clone from https://github.com/phoebe-p/S4)
- mpi4py
- refractiveindex (Python package by toftul) for refractiveindex.info parsing
- scikit-image 0.25.2 for quick geometry raster previews

If you install it in the same env, do not upgrade NumPy
- pip install --no-deps scikit-image lazy_loader

## Acknowledgements

- S4 (Stanford Stratified Structure Solver) by Victor Liu & Shanhui Fan.
Binary builds used here come from the phoebe-p channel.
- refractiveindex.info database curated by Mikhail Polyanskiy (and contributors).
- refractiveindex Python package by toftul for convenient access to the database.