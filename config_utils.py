# design_expander.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from itertools import product
import numpy as np

# ---------- generic spec → list of values ------------------------------------
def _expand_spec(spec: Any) -> List[float]:
    if isinstance(spec, (int, float)):      # bare number
        return [float(spec)]
    if not isinstance(spec, dict):          # unknown → treat as single value
        return [float(spec)]
    if "value" in spec:
        return [float(spec["value"])]
    if "lin" in spec:                        # [start, stop, num]
        lo, hi, num = spec["lin"]
        return list(np.linspace(float(lo), float(hi), int(num)))
    if "choices" in spec:
        return [float(x) for x in spec["choices"]]
    if "log" in spec:                        # optional: decades
        lo, hi, num = spec["log"]
        return list(np.logspace(float(lo), float(hi), int(num)))
    raise ValueError(f"Unsupported sweep spec: {spec!r}")

# ---------- shapes ------------------------------------------------------------
def _expand_shape(name: str, shp: Dict[str, Any]) -> List[Dict[str, Any]]:
    stype = shp["type"].lower()
    layer = shp.get("layer", "Pillar")
    mkey  = shp.get("material_key", "Meta")

    cx = _expand_spec(shp["center"]["x"])
    cy = _expand_spec(shp["center"]["y"])
    ang = _expand_spec(shp.get("rotation_deg", {"value": 0}))

    if stype == "rectangle":
        hx_list = _expand_spec(shp["halfwidths"]["x"])
        hy_spec = shp["halfwidths"]["y"]
        tie_y   = isinstance(hy_spec, dict) and hy_spec.get("tie") == "x"
        
        if tie_y:
            # elementwise tie: hy = hx for each hx value
            combos = (
                (x0, y0, a0, hx_val, hx_val)
                for x0 in cx
                for y0 in cy
                for a0 in ang
                for hx_val in hx_list
            )
        else:
            hy_list = _expand_spec(hy_spec)
            combos = product(cx, cy, ang, hx_list, hy_list)
        key = "halfwidths"
        
    elif stype == "ellipse":
        rx = _expand_spec(shp["radii"]["x"])
        ry_spec = shp["radii"]["y"]
        ry = rx if isinstance(ry_spec, dict) and ry_spec.get("tie") == "x" else _expand_spec(ry_spec)
        combos = product(cx, cy, ang, rx, ry)
        key = "radii"
    else:
        raise ValueError(f"Unsupported shape type: {stype}")

    out = []
    for (x0, y0, a0, p1, p2) in combos:
        out.append({
            "name": name,
            "type": stype,
            "layer": layer,
            "material_key": mkey,
            "center": (float(x0), float(y0)),
            "angle_deg": float(a0),
            "params": {key: (float(p1), float(p2))}
        })
    return out

# ---------- enumerate full design variants -----------------------------------
def enumerate_variants(cfg: Dict[str, Any], max_variants: int | None = None):
    """Yield dicts with concrete numbers for: a, layer thicknesses, and shapes."""
    # Period sweep
    a_candidates = _expand_spec(cfg["lattice"]["a"])

    # Layer thickness sweeps (keep order of layers in cfg)
    layer_defs = cfg["layers"]
    layer_candidates: List[List[Tuple[str, float]]] = []
    for L in layer_defs:
        name = L["name"]
        th = L["thickness"]
        # Allow 0 or numeric → treat as single value
        cand = _expand_spec(th) if isinstance(th, (int, float, dict)) else [float(th)]
        layer_candidates.append([(name, float(v)) for v in cand])

    # Shape sweeps
    per_shape_lists: List[List[Dict[str, Any]]] = []
    for sname, shp in (cfg.get("shapes") or {}).items():
        per_shape_lists.append(_expand_shape(sname, shp))
    if not per_shape_lists:
        per_shape_lists = [[ ]]  # one empty combination if no shapes

    # Cartesian product
    count = 0
    for a_val in a_candidates:
        for layer_combo in product(*layer_candidates):
            thickness_map = {name: val for (name, val) in layer_combo}
            for shape_combo in product(*per_shape_lists):
                variant = {
                    "a": float(a_val),
                    "thickness": thickness_map,       # dict: layer_name -> thickness
                    "shapes": list(shape_combo),      # list of concrete shape dicts
                }
                yield variant
                count += 1
                if max_variants is not None and count >= max_variants:
                    return
                
# Utils for variants from vector representation
def _decode_primitive_shape(
        s: List,
        *,
        name: str,
        layer: str,
        material_key: str,
) -> Dict[str, Any]:
    """
    Decode ['rectangle'|'ellipse', xc, yc, w, h] into the concrete shape dict.
    xc, yc - center in nm; w, h - width/height in nm.
    """
    if not isinstance(s, list) or len(s) < 1:
        raise ValueError(f"Bad shape vector: {s!r}")

    stype = str(s[0]).lower()
    if stype not in ("rectangle", "ellipse"):
        raise ValueError(f"Unsupported primitive shape type: {stype!r} in {s!r}")

    if len(s) != 5:
        raise ValueError(f"Expected [type, xc, yc, w, h], got: {s!r}")

    _, xc, yc, w, h = s
    p1, p2 = 0.5*w, 0.5*h  

    if stype == "rectangle":
        key = "halfwidths"
    else:
        key = "radii"

    return {
        "name": name,
        "type": stype,
        "layer": layer,
        "material_key": material_key,
        "center": (float(xc), float(yc)),
        "angle_deg": 0.0,  
        "params": {key: (float(p1), float(p2))}
    }
        
def _decode_shape_recursive(
        s: List,
        *,
        name_prefix: str,
        layer: str,
        meta_key: str,
        background_key: str,
        _counter: List[int],
) -> List[Dict[str, Any]]:
    """
    Returns a list of concrete shapes.
    - primitive -> [shape]
    - ring -> [outer_meta_shape, inner_background_shape]
    """
    if not isinstance(s, list) or len(s) < 1:
        raise ValueError(f"Bad shape vector: {s!r}")

    stype = str(s[0]).lower()

    if stype in ("rectangle", "ellipse"):
        _counter[0] += 1
        nm = f"{name_prefix}{_counter[0]}"
        return [_decode_primitive_shape(s, name=nm, layer=layer, material_key=meta_key)]

    if stype == "ring":
        if len(s) != 3:
            raise ValueError(f"Ring must be ['ring', outer_shape, inner_shape], got: {s!r}")
        outer, inner = s[1], s[2]

        # decode outer as Meta
        _counter[0] += 1
        outer_name = f"{name_prefix}{_counter[0]}_outer"
        outer_shape = _decode_primitive_shape(outer, name=outer_name, layer=layer, material_key=meta_key)

        # decode inner as background cutout
        _counter[0] += 1
        inner_name = f"{name_prefix}{_counter[0]}_inner_cutout"
        inner_shape = _decode_primitive_shape(inner, name=inner_name, layer=layer, material_key=background_key)

        # minimal sanity checks 
        oc = outer_shape["center"]; ic = inner_shape["center"]
        if (abs(oc[0] - ic[0]) > 1e-9) or (abs(oc[1] - ic[1]) > 1e-9):
            raise ValueError(f"Ring outer/inner centers differ: outer={oc}, inner={ic}")

        # ensure inner is smaller than outer in both axes
        okey = "halfwidths" if outer_shape["type"] == "rectangle" else "radii"
        ikey = "halfwidths" if inner_shape["type"] == "rectangle" else "radii"
        ow, oh = outer_shape["params"][okey]
        iw, ih = inner_shape["params"][ikey]
        if not (iw < ow and ih < oh):
            raise ValueError(f"Ring inner must be smaller than outer. outer={(ow,oh)}, inner={(iw,ih)}")

        # ordering: outer first, then inner cutout
        return [outer_shape, inner_shape]

    raise ValueError(f"Unsupported shape type: {stype!r} in {s!r}")

def _base_thickness_map(cfg: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for L in cfg["layers"]:
        th = L["thickness"]
        if isinstance(th, dict) and "value" in th:
            out[L["name"]] = float(th["value"])
        else:
            out[L["name"]] = float(th)
    return out

def from_vector(
    cfg: Dict[str, Any],
    vec: List[Any],
    *,
    default_layer: str = "Pillar",
    meta_key: str = "Meta",
    background_key: str = "Air",
    thickness_target_layer: str = "Pillar",
) -> Dict[str, Any]:
    """
    vec format: [shape1, shape2, ..., shapeN, period, thickness]
      - shape: ['rectangle'| 'ellipse', xc, yc, w, h]
      - ring:  ['ring', shape_outer, shape_inner]
    Produces: variant dict with concrete numeric shapes.
    """
    if not isinstance(vec, list) or len(vec) < 3:
        raise ValueError("vec must be [shapes..., period, thickness]")

    period = float(vec[-2])
    thickness_val = float(vec[-1])
    shape_vecs = vec[:-2]

    # thickness map from cfg, override one layer thickness by "thickness"
    thickness_map = _base_thickness_map(cfg)
    thickness_map[thickness_target_layer] = thickness_val

    # decode shapes
    shapes_out: List[Dict[str, Any]] = []
    counter = [0]
    for s in shape_vecs:
        decoded = _decode_shape_recursive(
            s,
            name_prefix="vshape_",
            layer=default_layer,
            meta_key=meta_key,
            background_key=background_key,
            _counter=counter,
        )
        shapes_out.extend(decoded)

    return {
        "a": period,
        "thickness": thickness_map,
        "shapes": shapes_out,
    }
    
def enumerate_variants_from_vectors(cfg, vectors, *, max_variants=None, **from_vector_kwargs):
    """
    Yield concrete S4 variants from a list of vector-defined metasurfaces.
    vectors: iterable of vec = [shape1, ..., shapeN, period, thickness]
    """
    count = 0
    for vec in vectors:
        yield from_vector(cfg, vec, **from_vector_kwargs)
        count += 1
        if max_variants is not None and count >= max_variants:
            return