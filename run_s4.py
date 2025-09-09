import numpy as np
import time
import yaml
import S4
from refractiveindex import RefractiveIndexMaterial
from mpi4py import MPI

from s4_interface import build_S_from_variant, make_eps_fns
from config_utils import enumerate_variants, _expand_spec

def run_s4(cfg, *, timer: bool = False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("The process has been started with", size, "MPI ranks.")
        
    #Read config
    eps_fns = make_eps_fns(cfg['materials']) # create epsilon functions
    num_basis = cfg['num_basis']
    num_orders = cfg['num_orders']
    max_variants = cfg['max_variants']
    
    # sweep wavelengths/angles from cfg['sweep'] at runtime
    wl = _expand_spec(cfg['sweep']['wl'])
    theta = _expand_spec(cfg['sweep']['theta'])
    phi = _expand_spec(cfg['sweep']['phi'])
    
    # Distribute wavelength indices round-robin for MPI
    my_idx = np.arange(rank, len(wl), size, dtype=int)
    
    if timer:
        comm.Barrier()
        t0 = MPI.Wtime() # Start counting time
    
    for vidx, variant in enumerate(enumerate_variants(cfg, max_variants=max_variants)): # Iterates over all geometry variants
        # Init transmission and reflection
        T = np.zeros((2, num_orders, len(theta), len(wl)), dtype=np.float64) #first dim is polarization
        R = np.zeros((2, num_orders, len(theta), len(wl)), dtype=np.float64)
        
        T_phase = np.zeros((2, num_orders, len(theta), len(wl)), dtype=np.float64)
        R_phase = np.zeros((2, num_orders, len(theta), len(wl)), dtype=np.float64)
        
        # Build the model
        S = build_S_from_variant(cfg, variant, num_basis=num_basis)
        
        #Set options
        S.SetOptions( # these are the defaults
                Verbosity = cfg['S4_options']['Verbosity'],
                LatticeTruncation = cfg['S4_options']['LatticeTruncation'],
                DiscretizedEpsilon = cfg['S4_options']['DiscretizedEpsilon'],
                DiscretizationResolution = cfg['S4_options']['DiscretizationResolution'],
                PolarizationDecomposition = cfg['S4_options']['PolarizationDecomposition'],
                PolarizationBasis = cfg['S4_options']['PolarizationBasis'],
                LanczosSmoothing = cfg['S4_options']['LanczosSmoothing'],
                SubpixelSmoothing = cfg['S4_options']['SubpixelSmoothing'],
                ConserveMemory = cfg['S4_options']['ConserveMemory']
        )
        
        # Angle and wavelength sweep
        for a_idx, alpha in enumerate(theta): # Iterates over angles
            for pidx in range(2): # Iterates over polarizations
                S.SetExcitationPlanewave(IncidenceAngles=(alpha, phi[0]), # phi is a single value
                                        sAmplitude=(1 - pidx)+0j, 
                                        pAmplitude=pidx+0j, 
                                        Order=0) 
                
                for idx in my_idx: # Iterates over wavelengths with MPI
                    lam = wl[idx]
                    S.SetFrequency(1.0/lam)
                    
                    for key, fn in eps_fns.items():
                        S.SetMaterial(Name=key, Epsilon=complex(fn(lam)))
                        
                    # Normalize by incident forward power in Top
                    f_top, _ = S.GetPowerFlux(Layer="Top")
                    P_inc = f_top.real
                    
                    # Power by Fourier order for top/bottom layers: sequence matches Glist
                    P_top = S.GetPowerFluxByOrder(Layer="Top")      # [(forw, back) for each order]
                    P_bot = S.GetPowerFluxByOrder(Layer="Bottom")
                    
                    # --- Complex amplitudes for phase ---
                    _, back_top_amp = S.GetAmplitudes("Top")     # reflection lives in 'back' at Top
                    fwd_bot_amp, _ = S.GetAmplitudes("Bottom")  # transmission lives in 'fwd' at Bottom
                    
                    # Extract each requested (m,n)
                    for j in range(num_orders):
                        
                        # Intensity
                        _, back_top = P_top[j]
                        forw_bot, _ = P_bot[j]
                        # Reflection lives in Top/backward (negative z). Transmission in Bottom/forward.
                        R[pidx, j, a_idx, idx] = (-back_top).real / P_inc
                        T[pidx, j, a_idx, idx] = ( forw_bot).real / P_inc
                        
                        # Phase
                        # reflection (Top/back), transmission (Bottom/forward)
                        base = 2*j          # two pol channels per order
                        chan = pidx                  # 0 = s, 1 = p (because of Jones basis)

                        r_amp = back_top_amp[base + chan]   # reflection: Top/back
                        t_amp = fwd_bot_amp[base + chan]    # transmission: Bottom/forward

                        R_phase[pidx, j, a_idx, idx] = np.angle(r_amp)  # [-pi, pi]
                        T_phase[pidx, j, a_idx, idx] = np.angle(t_amp)
                        
        T_global = np.empty_like(T)
        R_global = np.empty_like(R)
        T_phase_global = np.zeros_like(T_phase)
        R_phase_global = np.zeros_like(R_phase)
        
        comm.Allreduce(T, T_global, op=MPI.SUM)
        comm.Allreduce(R, R_global, op=MPI.SUM)
        comm.Allreduce(T_phase, T_phase_global, op=MPI.SUM)
        comm.Allreduce(R_phase, R_phase_global, op=MPI.SUM)
        
        #Get diffraction orders
        Glist = S.GetBasisSet()  # list of (m,n) integer pairs in the same order as power arrays
        orders = Glist[:num_orders]

        # ---- save from a single writer
        if rank == 0:
            np.savez(f"{cfg['filename']}_{vidx}.npz", 
                     variant=variant, 
                     T=T_global, 
                     R=R_global,
                     T_phase = T_phase_global,
                     R_phase = R_phase_global,
                     wl=wl,
                     theta=theta,
                     phi=phi,
                     orders = orders,
                     materials = cfg['materials'])
                    
    if timer:
        comm.Barrier()
        elapsed = MPI.Wtime() - t0
        wall = comm.reduce(elapsed, op=MPI.MAX, root=0)
        if rank == 0:
            print(f"Simulation time (wall, max over {size} ranks): {wall:.3f} s")
                        
if __name__ == '__main__':
    cfg = yaml.safe_load(open("s4_config.yaml")) 
    run_s4(cfg, timer=True)         
                