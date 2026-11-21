"""How much of the mesh's link traffic is global exchange, from outputs/cells.json.

If global traffic is phased (exchange, then compute) and links go quiet when idle, the comm-band leak
swings by about this fraction once per superstep. If it runs flat out for the whole superstep, as
cells.json books it (global_bits_per_superstep = bisection x superstep), the leak is steady."""
import json, numpy as np
from pathlib import Path
d = json.load(open(Path(__file__).parents[2] / "outputs" / "cells.json"))
o, w, f = d["fiducial"]["optimum"], d["workloads"], d["fiducial"]
R = 220 * 1.495978707e11
n_nodes = f["n_cells_optimum"] * o["nodes_per_cell"]
capacity = n_nodes * o["z"] * o["b_link"] / 2                       # bit-hops per second, all links
stretch = d["lattices"]["fcc"]["stretch_mean"]
hops_global = stretch * (np.pi * R / 2) / o["d"]                     # mean great-circle path, uniform pairs
global_load = w["cloud"]["bisection_bits_s"] * hops_global
local_load = f["beta"] * w["cloud"]["bitops_s"] * 2                  # beta bits per op, ~2 hops
print(f"link capacity            {capacity:.2e} bit-hops/s")
print(f"global exchange at full bisection: {global_load:.2e}  ({global_load/capacity:.2f} of capacity)")
print(f"cell-level (beta) traffic:         {local_load:.2e}  ({local_load/capacity:.2f} of capacity)")
print(f"max superstep swing if exchange is phased: {global_load/(global_load+local_load):.2f} of the traffic")
