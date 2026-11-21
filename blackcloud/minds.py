"""Minds as a workload: what it costs a Black Cloud to run a human-brain-scale emulation.

Inputs are measured brain quantities plus two stated assumptions:
  * N_NEURON  86.1e9 neurons in the adult human brain (Azevedo et al. 2009)
  * N_SYNAPSE 1.64e14 synapses in the neocortex (Tang et al. 2001)
  * ACTIVE_FRACTION_MAX 0.01: energy limits fewer than 1% of cortical neurons to be strongly
    active at once (Lennie 2003)
  * ACTIVE_RATE_HZ 50 (assumption): the rate of a strongly active neuron.  Together with the
    1% bound this caps the mean rate at 0.5 Hz; we take 0.1 Hz as the low end.
  * BITOPS_PER_EVENT: one multiply-add per synaptic event, 3e4 bit operations (range 1e4-1e5),
    the same circuit estimate as the bits-per-operation ratio (paper section 3.3).
  * T_STEP_S 1e-3 (assumption): the shortest timescale an emulation must resolve.
"""
import numpy as np

N_NEURON = 86.1e9
N_SYNAPSE = 1.64e14
ACTIVE_FRACTION_MAX = 0.01
ACTIVE_RATE_HZ = 50.0
RATE_LOW_HZ = 0.1
BITOPS_PER_EVENT = (1e4, 3e4, 1e5)
T_STEP_S = 1e-3
BRAIN_POWER_W = 20.0          # glucose supply of the brain (Levy & Calvert 2020)
SPEECH_BITS_S = 39.0          # information rate of human speech (Coupe et al. 2019)


def mind_cost(rate_hz, bitops_per_event):
    """Bit operations per second to emulate one brain at synapse level, in real time."""
    return N_SYNAPSE * rate_hz * bitops_per_event


def spike_traffic(rate_hz):
    """Bits per second needed to ship every spike as an address event (log2 N bits each)."""
    return N_NEURON * rate_hz * np.log2(N_NEURON)


def hosting(cells_json, kTln2):
    """Mind-hosting figures for the fiducial design, from outputs/cells.json."""
    w = cells_json['workloads']
    node, cell, cloud = w['node'], w['cell'], w['cloud']
    rate_hi = ACTIVE_FRACTION_MAX * ACTIVE_RATE_HZ
    lo, mid, hi = BITOPS_PER_EVENT
    C = mind_cost(rate_hi, mid)
    C_range = (mind_cost(RATE_LOW_HZ, lo), mind_cost(rate_hi, hi))
    traffic = spike_traffic(rate_hi)
    per_node = node['bitops_s'] / C
    s_compute_node = per_node
    s_latency = T_STEP_S / node['neighbor_latency_s']
    yr = 3.15576e7
    return dict(
        mean_rate_max_hz=rate_hi,
        synaptic_events_s=N_SYNAPSE * rate_hi,
        mind_bitops_s=C, mind_bitops_s_low=C_range[0], mind_bitops_s_high=C_range[1],
        mind_power_W=C * kTln2, biological_over_cloud_power=BRAIN_POWER_W / (C * kTln2),
        spike_traffic_bits_s=traffic, mind_bits_per_op=traffic / C,
        minds_per_node=per_node, minds_per_node_low=node['bitops_s'] / C_range[1],
        minds_per_node_high=node['bitops_s'] / C_range[0],
        spike_traffic_fraction_of_node_links=per_node * traffic / node['link_bw_bits_s'],
        link_bits_s_per_mind=node['link_bw_bits_s'] / per_node,
        link_over_speech=node['link_bw_bits_s'] / per_node / SPEECH_BITS_S,
        minds_per_cell=cell['bitops_s'] / C,
        speedup_one_node=s_compute_node, speedup_latency_limit=s_latency,
        nodes_for_max_speedup=s_latency / s_compute_node,
        minds_total=cloud['bitops_s'] / C,
        minds_total_low=cloud['bitops_s'] / C_range[1], minds_total_high=cloud['bitops_s'] / C_range[0],
        mind_years_lifetime=cloud['ops_lifetime'] / (C * yr),
        subjective_superstep_yr_at_max_speedup=s_latency * cloud['superstep_backbone_d'] / 365.25,
        subjective_superstep_d_one_node=s_compute_node * cloud['superstep_backbone_d'],
        subjective_antipode_yr_at_max_speedup=s_latency * cloud['superstep_backbone_d'] / 2 / 365.25,
    )
