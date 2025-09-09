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
        hx = _expand_spec(shp["halfwidths"]["x"])
        # support tie for y
        hy_spec = shp["halfwidths"]["y"]
        hy = cx if isinstance(hy_spec, dict) and hy_spec.get("tie") == "x" else _expand_spec(hy_spec)
        combos = product(cx, cy, ang, hx, hy)
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