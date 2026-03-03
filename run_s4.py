import numpy as np
import yaml
from refractiveindex import RefractiveIndexMaterial
from mpi4py import MPI

from s4_interface import build_S_from_variant, make_eps_fns
from config_utils import enumerate_variants, _expand_spec, from_vector

def iter_variants(cfg, *, vectors=None):
    """
    Yield a stream of geometry variants to simulate.

    Output:
        Yields tuples (vidx, variant)
            vidx    : integer variant index (used for distribution + filenames)
            variant : concrete geometry dict consumable by build_S_from_variant()

    Behavior:
      - If `vectors` is provided: geometries come ONLY from vectors.
      - Otherwise: geometries come from YAML sweeps via enumerate_variants(cfg).

    Notes:
      - `max_variants` is a safety cap from cfg to stop endless/huge sweeps.
    """
    max_variants = cfg.get("max_variants", None)

    if vectors is not None:
        # Loop over user-provided vector-defined metasurfaces.
        # Each vector is converted into a concrete `variant` dict.
        for vidx, vec in enumerate(vectors):
            yield vidx, from_vector(
                cfg, vec,
                default_layer="Pillar",
                meta_key="Meta",
                background_key="Air",
                thickness_target_layer="Pillar",
            )
            if max_variants is not None and (vidx + 1) >= max_variants:
                return
    else:
         # Loop over YAML-defined sweeps (period, thickness, shapes, etc.)
        for vidx, variant in enumerate(enumerate_variants(cfg, max_variants=max_variants)):
            yield vidx, variant


def run_s4(cfg, *, timer: bool = False, vectors=None, group_size: int | None = None):
    """
    Run S4 simulations in parallel with MPI using 2-level decomposition:

    1) "Variant groups" (coarse parallelism):
       - MPI ranks are split into n_groups groups.
       - Each group is responsible for a subset of variants:
           group_id handles variants where (vidx % n_groups) == group_id

    2) "Wavelength split within a group" (fine parallelism):
       - Inside each group, ranks split the wavelength indices.
       - Each rank computes only its wl slice for every (theta, pol) point.
       - Results are combined inside the group with Allreduce.

    This design keeps:
      - multiple variants running concurrently (across groups)
      - each variant also parallelized over wavelength (within group)
    """

    # ---------------- MPI world info ----------------
    comm = MPI.COMM_WORLD          # communicator containing all ranks
    rank = comm.Get_rank()         # this process's rank id
    size = comm.Get_size()         # total number of ranks
    
    if rank == 0:
        print(f"Starting run_s4 with MPI size={MPI.COMM_WORLD.Get_size()}", flush=True)

    # ---------------- Parse / precompute config ----------------
    eps_fns = make_eps_fns(cfg["materials"])
    # eps_fns: dict material_key -> function(lam)->epsilon
    
    num_basis = cfg["num_basis"]   # number of Fourier harmonics used by S4
    num_orders = cfg["num_orders"] # how many diffraction orders are stored

    # Expand sweep specs from YAML into explicit lists.
    wl    = _expand_spec(cfg["sweep"]["wl"])      # list of wavelengths
    theta = _expand_spec(cfg["sweep"]["theta"])   # list of incidence polar angles
    phi   = _expand_spec(cfg["sweep"]["phi"])     # list of incidence azimuthal angles
    n_wl = len(wl)

    # ---------------- Decide grouping layout ----------------
    # If vectors are provided, total number of variants is known.
    if vectors is not None:
        n_variants_total = min(len(vectors), cfg.get("max_variants", len(vectors)))
    else:
        n_variants_total = None  # unknown without counting enumerate_variants() first

    # Choose group_size (ranks per group).
    if group_size is None:
        if n_variants_total is not None and n_variants_total > 0:
            # Aim for up to one group per variant (but limited by MPI size),
            # then allocate remaining ranks to wavelength splitting.
            n_groups = min(n_variants_total, size)
            group_size = max(1, min(n_wl, size // n_groups))
        else:
            # If variants count is unknown, prioritize wavelength splitting.
            group_size = min(n_wl, size)

    # Clamp to valid range
    group_size = int(max(1, min(group_size, size)))

    # Number of groups in the world communicator
    n_groups = (size + group_size - 1) // group_size

    # Group identity of this rank
    group_id = rank // group_size
    rank_in_group = rank % group_size

    # Create a sub-communicator per group (collectives happen here)
    gcomm = comm.Split(color=group_id, key=rank_in_group)
    grank = gcomm.Get_rank()
    gsize = gcomm.Get_size()

    # Wavelength indices assigned to THIS rank inside its group:
    # rank 0 gets 0, gsize, 2*gsize, ...
    # rank 1 gets 1, gsize+1, ...
    my_widx = np.arange(grank, n_wl, gsize, dtype=int)

    # ---------------- Optional timing synchronization ----------------
    # Barrier ensures all ranks start timing at the same moment.
    if timer:
        comm.Barrier()
        t0 = MPI.Wtime()

    # ================= Main simulation loop over variants =================
    for vidx, variant in iter_variants(cfg, vectors=vectors):
        
        # -------- Variant distribution across groups --------
        # Only the group whose id matches (vidx % n_groups) will process this variant.
        # All other groups skip.
        if (vidx % n_groups) != group_id:
            continue

        # -------- Allocate output arrays (full wavelength axis) --------
        # Each rank fills only its assigned wavelengths; later Allreduce sums them together.
        # Array dims:
        #   pol: 2 (s/p)
        #   order: num_orders
        #   theta index
        #   wavelength index
        T = np.zeros((2, num_orders, len(theta), n_wl), dtype=np.float64)
        R = np.zeros((2, num_orders, len(theta), n_wl), dtype=np.float64)
        T_phase = np.zeros((2, num_orders, len(theta), n_wl), dtype=np.float64)
        R_phase = np.zeros((2, num_orders, len(theta), n_wl), dtype=np.float64)

        # -------- Build S4 model for this geometry variant --------
        # Creates S4 simulation object with:
        #   - lattice period
        #   - layer stack + thicknesses
        #   - regions/shapes in layers
        #   - basis size (num_basis)
        S = build_S_from_variant(cfg, variant, num_basis=num_basis)

        # -------- Apply S4 numerical options from config --------
        S.SetOptions(
            Verbosity=cfg["S4_options"]["Verbosity"],
            LatticeTruncation=cfg["S4_options"]["LatticeTruncation"],
            DiscretizedEpsilon=cfg["S4_options"]["DiscretizedEpsilon"],
            DiscretizationResolution=cfg["S4_options"]["DiscretizationResolution"],
            PolarizationDecomposition=cfg["S4_options"]["PolarizationDecomposition"],
            PolarizationBasis=cfg["S4_options"]["PolarizationBasis"],
            LanczosSmoothing=cfg["S4_options"]["LanczosSmoothing"],
            SubpixelSmoothing=cfg["S4_options"]["SubpixelSmoothing"],
            ConserveMemory=cfg["S4_options"]["ConserveMemory"],
        )

        # ================= Nested sweeps (theta × polarization × wavelength) =================
        for a_idx, alpha in enumerate(theta):
            for pidx in range(2):
                # ---- Set plane-wave excitation for this angle/polarization ----
                # pidx=0: s-polarized only
                # pidx=1: p-polarized only
                S.SetExcitationPlanewave(
                    IncidenceAngles=(alpha, phi[0]),
                    sAmplitude=(1 - pidx) + 0j,
                    pAmplitude=pidx + 0j,
                    Order=0,
                )
                
                # ---- Wavelength sweep split across ranks in the group ----
                for w_idx in my_widx:
                    lam = wl[w_idx]
                    
                    # Set frequency (S4 uses frequency; you use 1/λ in your units)
                    S.SetFrequency(1.0 / lam)

                    # Update dispersive materials: for each material key, set epsilon(λ)
                    for key, fn in eps_fns.items():
                        S.SetMaterial(Name=key, Epsilon=complex(fn(lam)))

                    # Normalize by incident forward power in Top layer
                    f_top, _ = S.GetPowerFlux(Layer="Top")
                    P_inc = f_top.real

                    # Power per diffraction order in Top and Bottom
                    P_top = S.GetPowerFluxByOrder(Layer="Top")
                    P_bot = S.GetPowerFluxByOrder(Layer="Bottom")
                    
                    # Complex amplitudes (for phases) at Top(back) and Bottom(forward)
                    _, back_top_amp = S.GetAmplitudes("Top")
                    fwd_bot_amp, _  = S.GetAmplitudes("Bottom")

                    # ---- Extract per diffraction order (j = 0..num_orders-1) ----
                    # Assuming arrays are ordered consistently with S.GetBasisSet().
                    for j in range(num_orders):
                        # Intensities:
                        _, back_top = P_top[j]
                        forw_bot, _ = P_bot[j]

                        # Reflection is Top/backward; sign convention -> (-back_top).real
                        R[pidx, j, a_idx, w_idx] = (-back_top).real / P_inc
                        T[pidx, j, a_idx, w_idx] = ( forw_bot).real / P_inc

                        # Phases:
                        # base index assumes 2 channels per order (Jones basis): [s, p]
                        base = 2 * j
                        chan = pidx
                        r_amp = back_top_amp[base + chan]
                        t_amp = fwd_bot_amp[base + chan]

                        R_phase[pidx, j, a_idx, w_idx] = np.angle(r_amp)
                        T_phase[pidx, j, a_idx, w_idx] = np.angle(t_amp)

        # ================= Combine wavelength slices within the group =================
        # Each rank filled only its my_widx columns; others are zeros.
        # Allreduce SUM merges them into full arrays on all ranks in the group.
        T_g = np.zeros_like(T)
        R_g = np.zeros_like(R)
        Tph_g = np.zeros_like(T_phase)
        Rph_g = np.zeros_like(R_phase)

        gcomm.Allreduce(T, T_g, op=MPI.SUM)
        gcomm.Allreduce(R, R_g, op=MPI.SUM)
        gcomm.Allreduce(T_phase, Tph_g, op=MPI.SUM)
        gcomm.Allreduce(R_phase, Rph_g, op=MPI.SUM)

        # Diffraction order (m,n) labels corresponding to S4 internal basis ordering
        orders = S.GetBasisSet()[:num_orders]

        # ================= Save results (one writer per group) =================
        # Only the group leader writes to disk to avoid file collisions.
        if grank == 0:
            outpath = f"{cfg['filename']}_{vidx}.npz"
            print(f"[group {group_id} writer rank {rank}] Saving {outpath}", flush=True)
            np.savez(
                outpath,
                variant=variant,
                T=T_g, R=R_g,
                T_phase=Tph_g, R_phase=Rph_g,
                wl=wl, theta=theta, phi=phi,
                orders=orders,
                materials=cfg["materials"],
                vidx=vidx,
                group_id=group_id,
                group_size=gsize,
            )

    # ---------------- Timing end (optional) ----------------
    if timer:
        comm.Barrier()
        elapsed = MPI.Wtime() - t0
        wall = comm.reduce(elapsed, op=MPI.MAX, root=0)
        if rank == 0:
            print(f"Simulation time (wall, max over {size} ranks): {wall:.3f} s", flush=True)
                        
if __name__ == '__main__':
    cfg = yaml.safe_load(open("s4_config.yaml")) 
    shape1 = ['rectangle', 0, 0, 0.2, 0.3]
    shape2 = ['ellipse', 0, 0, 0.1, 0.15]
    shape3 = ['ring', shape1, shape2]
    
    vector1 = [shape1, 0.5, 0.2] # shapes..., period, thickness
    vector2 = [shape2, 0.4, 0.2]
    vector3 = [shape3, 0.5, 0.2]
    vector4 = [shape1, shape2, 0.6, 0.2]
    
    vectors = [vector1, vector2, vector3, vector4]
    
    run_s4(cfg, timer=True, vectors=vectors)         
                