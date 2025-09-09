import numpy as np
import S4
from refractiveindex import RefractiveIndexMaterial

from skimage.draw import polygon, ellipse

# ---------- shapes -> S4 ------------------------------------------------------
def add_shapes_to_S4(S, design_instances):
    for s in design_instances:
        layer = s["layer"]
        mat   = s["material_key"]
        (cx, cy) = s["center"]
        ang = float(s.get("angle_deg", 0.0))
        t = s["type"].lower()
        if t == "rectangle":
            hx, hy = s["params"]["halfwidths"]
            S.SetRegionRectangle(Layer=layer, Material=mat,
                                 Center=(cx, cy), Angle=ang, Halfwidths=(hx, hy))
        elif t == "ellipse":
            rx, ry = s["params"]["radii"]
            S.SetRegionEllipse(Layer=layer, Material=mat,
                               Center=(cx, cy), Angle=ang, Halfwidths=(rx, ry))
        else:
            raise ValueError(f"Unknown shape type: {s['type']}")
        
# ---------- refractiveindex.info -> epsilon(w) --------------------------------
def make_eps_fns(materials):
    """Return dict key -> eps(lam_um) for each material_key in your config."""

    fns = {}
    for key, m in materials.items():
        src = str(m.get("source", "")).strip()
        if src.startswith("n="):                      # constant index
            n0 = float(src.split("=", 1)[1])
            fns[key] = lambda lam, n0=n0: (n0 + 0j)**2
        else:                                         # from refractiveindex.info
            # Dataset from the DB
            mat_ri = RefractiveIndexMaterial(shelf=m.get("shelf", ""), 
                                             book=m.get("name", ""), 
                                             page=m.get("source", ""))

            def eps_of_w(w_um, _mat=mat_ri):
                lam = float(w_um)*1000
                return _mat.get_epsilon(lam)
            
            fns[key] = eps_of_w
    return fns

def build_S_from_variant(cfg, variant, num_basis=169):
    a = variant["a"]  # use the swept period
    S = S4.New(Lattice=((a, 0.0), (0.0, a)), NumBasis=num_basis)

    # Register materials once (we’ll update epsilon(λ) later)
    for key in cfg["materials"].keys():
        S.SetMaterial(Name=key, Epsilon=1.0)

    # Add layers with swept thickness (fallback to cfg if not swept)
    t_by_name = variant["thickness"]
    for L in cfg["layers"]:
        name = L["name"]
        t = float(t_by_name.get(name, L["thickness"] if isinstance(L["thickness"], (int,float)) else 0.0))
        S.AddLayer(Name=name, Thickness=t, Material=L["material_key"])

    # Add concrete shapes for this variant
    for s in variant["shapes"]:
        layer = s["layer"]; mat = s["material_key"]
        (cx, cy) = s["center"]; ang = s.get("angle_deg", 0.0)
        if s["type"] == "rectangle":
            hx, hy = s["params"]["halfwidths"]
            S.SetRegionRectangle(Layer=layer, Material=mat, Center=(cx, cy), Angle=ang, Halfwidths=(hx, hy))
        else:
            rx, ry = s["params"]["radii"]
            S.SetRegionEllipse(Layer=layer, Material=mat, Center=(cx, cy), Angle=ang, Halfwidths=(rx, ry))
    return S

def plot_metasurface(variant, 
                     grid_resolution: int = 128, 
                     material_values=None, 
                     show: bool = False):
    """
    Rasterize shapes in one geometry variant onto a grid.
    - variant['a'] is the period (µm), shapes use center/halfwidths/radii in µm, angle in deg.
    - material_values maps material_key -> pixel value (e.g., {'Meta':1.0,'Air':0.0}).
    Returns: 2D numpy array of shape (grid_resolution, grid_resolution)
    """
    shapes = variant['shapes']
    period = float(variant['a'])
    scale = period / grid_resolution  # µm per pixel

    if material_values is None:
        material_values = {'Meta': 1.0, 'Air': 0.0, 'Sub': 0.0}

    img = np.zeros((grid_resolution, grid_resolution), dtype=float)

    for idx, shape in enumerate(shapes):
        stype = shape['type'].lower()
        cx = float(shape['center'][0]) + period/2.0  # shift to [0, period]
        cy = float(shape['center'][1]) + period/2.0
        ang = np.deg2rad(float(shape.get('angle_deg', 0.0)))

        if stype == 'rectangle':
            hx, hy = map(float, shape['params']['halfwidths'])
            # rectangle corners around (0,0)
            corners = np.array([[-hx, -hy],
                                [ hx, -hy],
                                [ hx,  hy],
                                [-hx,  hy]])
            # rotate then translate
            R = np.array([[np.cos(ang), -np.sin(ang)],
                          [np.sin(ang),  np.cos(ang)]])
            verts = corners @ R.T + np.array([cx, cy])
            # convert to pixel coords: (row, col) = (y/scale, x/scale)
            rr, cc = polygon(verts[:,1] / scale, verts[:,0] / scale, shape=img.shape)

        elif stype == 'ellipse':
            rx, ry = map(float, shape['params']['radii'])   # radii in x,y (µm)
            r0 = cy / scale
            c0 = cx / scale
            rr, cc = ellipse(r0, c0, ry/scale, rx/scale, shape=img.shape, rotation=ang)

        else:
            raise ValueError(f"Unsupported shape type: {shape['type']}")

        val = material_values.get(shape.get('material_key', 'Meta'), 1.0)
        img[rr.astype(int), cc.astype(int)] = val  # last shape wins

    if show:
        import matplotlib.pyplot as plt
        plt.imshow(img, origin='lower',
                   extent=[-period/2, period/2, -period/2, period/2])
        plt.xlabel('x (µm)'); plt.ylabel('y (µm)'); plt.title('Metasurface')
        plt.show()

    return img
            
            
            
    
    
