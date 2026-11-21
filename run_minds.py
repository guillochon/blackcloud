"""Mind-hosting figures for the discussion: reads outputs/cells.json, writes outputs/minds.json.
Run after run_cells.py."""
import json
from pathlib import Path
from blackcloud.params import CloudParams, k_B, LN2
from blackcloud import minds

OUT = Path(__file__).parent / 'outputs'
p = CloudParams()
res = minds.hosting(json.load(open(OUT / 'cells.json')), k_B * p.T_out * LN2)
json.dump(res, open(OUT / 'minds.json', 'w'), indent=2)
for k, v in res.items():
    print(f'{k:42s} {v:.3g}')
