# Black Cloud megacomputer: a toy model

This is a toy model of the Laughlin/Adams "Black Cloud": a computer built in the wind of an AGB star, about 220 AU across, near 175 K, holding about 1 M⊕ of material, with nodes that talk over millimeter-wave (3 mm) links. It asks two questions: how do signals propagate through the structure, and what leaks out where a distant observer can see it?

```bash
python run_model.py
```

Every knob is in `blackcloud/params.py`. Figures and `results.json` go to `outputs/`. A full run takes about 15 s.

| module | contents |
|---|---|
| `thermo.py` | Landauer and entropy bit budgets, SEDs (engineered shell vs. natural dust envelope), and how each IR band responds to the Mira pulsation |
| `network.py` | Link budget (Friis plus Shannon at a noise floor of T_sys ≈ T_cloud), multi-hop economics, and Dijkstra propagation on a coarse-grained shell (mesh vs. mesh + backbone) |
| `observe.py` | Unresolved light curves with light-travel delays, the shell's transfer function, and the millimeter comm-band spectrum with outflow Doppler smearing |

## Fiducial numbers

L* = 2000 L☉ and R = 220 AU, which gives T_out = 177 K. The comm band is 285–315 GHz. Node tiles are 10 m across, backbone hubs 10 km, the per-hop latency is 1 ms, and 1% of the power goes to communications.

| quantity | value |
|---|---|
| Irreversible bit rate, L/(kT ln2) | 4.5 × 10⁵⁰ bit/s (entropy bound 5.6 × 10⁵⁰) |
| Areal mass of 1 M⊕ spread over the shell | 0.44 g/m² |
| Thermal time constant of that layer | ≤ 0.36 s (Dulong–Petit bound), so IR tracks dissipation instantly |
| Light-crossing time, 2R/c | 2.54 d |

## Results

### 1. Signals propagate slowly, and most of the computation has to stay local
- The cheapest mesh hop is the edge of the near field, d₀ = A/λ ≈ 130 km. Shorter hops waste relay overhead. Longer hops pay (d/d₀)² in path loss. This sets the mesh signal speed to v = d₀/(d₀/c + τ_proc) ≈ 0.31c. A broadcast takes about 13 days to reach the antipode (fig 3, fig 4).
- Over the mesh, one bit costs about 2 × 10⁹ kT ln2 to reach the antipode. Large hubs that beam straight across the cavity cut this to about 10⁷ kT ln2 and about 2.6 days, so any real design would build a backbone of km-scale apertures. Spreading 24 hubs evenly (a snub cube) brings the worst-case broadcast to 6.6 days, against 7.8 days for the same hubs placed at random.
- Even with the backbone, the global bandwidth is only about 10⁻⁸ of the local bit rate. The cloud behaves less like one computer and more like 10⁸-way loosely coupled cells. A globally coherent step takes days.

### 2. Waste heat can't show what the cloud is computing
- Starlight in equals IR out, and the thermal time constant is at most 0.36 s. Thermal emission therefore follows the absorbed starlight whether nodes are busy or idle. Bolometric IR is set by the star, not by the workload.
- **Corrects the earlier "unnaturally narrow blackbody" test.** Once natural dust has a ν^β emissivity (β = 1), its mid-IR SED is about as narrow as a single-T blackbody with the same peak: 0.51–0.58 dex against 0.54 dex (fig 1). The robust discriminant is the **sub-mm spectral index**. A gray engineered shell gives α₄₅₀–₈₅₀ ≈ 1.9, while natural dust gives α ≈ 2.6–2.9. That test assumes the structure is gray at sub-mm wavelengths, which a thin carbon layer generally isn't unless it is engineered to be. The alternative is spectral features such as SiC at 11.3 µm or silicate at 10 µm.
- The time-domain discriminant from the StarSETI angle holds only for one design (fig 2). A passive engineered shell (T ∝ L^¼) varies chromatically, almost exactly like a natural envelope: d ln F/d ln L = 1.69 vs. 1.49 at 12 µm, and 0.95 vs. 0.96 at 22 µm. A **thermostatted** shell, which holds T fixed for device reliability and modulates its radiating area instead, varies **achromatically**. Its color stays constant while W4 swings by ±0.4 mag, which no passively heated dust does. No design can buffer the Mira cycle thermally: smoothing ±40% of L over 200 days would need ΔT of about 10⁹ K across an Earth mass.

### 3. The communication band is the only channel that carries computational state
- If the links run efficiently (low SNR, near the Shannon limit), the comm band is optically thick and its surface radiates the traffic. From outside, this looks like a flat-topped, noise-like plateau at roughly 2× T_cloud, with sharp edges (fig 7). For the fiducial 10 GHz band at 100 GHz, the excess equals the thermal continuum: about 0.19 Jy at 1 kpc and about 3 mJy at 8 kpc. That is easily within ALMA's reach anywhere in the Galaxy. The leaked power is only about 3 × 10⁻⁷ L. Well-encrypted traffic is statistically thermal, so the signature is the **band shape**, not the content.
- The band edges are Doppler-smeared by about ν v_exp/c ≈ 10 MHz. Because only the approaching hemisphere of an opaque shell is visible, the profile is one-sided (blueshifted). Its edge width therefore measures the outflow speed, just as CO line profiles do for natural AGB winds.
- A natural plateau 30 GHz wide with vertical edges and no molecular identification would be very strange. SiO and H₂O masers in AGB envelopes are bright and variable, but they are narrow lines.

### 4. Light travel blurs the cloud's timekeeping for an unresolved observer
- By Archimedes' hat-box theorem, the light delays across a thin shell are uniformly distributed over 2R/c. Anything that happens simultaneously in the cloud's frame reaches us convolved with a 2.54-day boxcar, a transfer of |sinc(2fR/c)| (fig 6, drawn by `run_cells.py` beside the 5.3–26 day global superstep cadence). A 1-day shell-wide modulation keeps about 12% of its amplitude. Activity that is independent from cell to cell, such as a population of minds, is not filtered but averages down as N_cells^-½.
- A single broadcast wave shows up in the comm band as a burst lasting about 6–12 days over the mesh, or about 1.3–4 days with a backbone of 24 hubs at the vertices of a snub cube (fig 5). Its shape depends on where the wave starts relative to our line of sight. Bursts longer than 2R/c but shorter than the Mira period form a time-domain window that only computation fills. Natural envelopes have nothing that varies coherently on day timescales across 10 GHz.

### 5. Computing cells (`blackcloud/cells.py`, `python run_cells.py` → figs 6 and 8–11, `cells.json`)
- Suppose nodes intercept starlight with radial covering τ, carry one Fresnel-matched aperture per lattice neighbor, and use links of fractional bandwidth f_B. Then a region balances its volume compute against its surface bandwidth at **ℓ_cell = 2τ f_B c kT ln2 / (β F λ²)**. This is independent of the lattice and of the shell thickness. β is the number of bits exchanged per bit-op, and F is the absorbed flux.
- The 3 mm links in the Laughlin et al. design give ℓ_cell ≈ 0.2 mm with nodes 120 km apart, so the network is bandwidth-starved. The global bisection bandwidth is only 2πR τ f_B c/λ².
- For supercomputer-like traffic (β ~ 10⁻⁶) the optimum is 1 µm links, about 1.3 cm rhombic-dodecahedral nodes 182 m apart on an FCC lattice, and a 0.05 AU shell. That gives about 10²⁸ cells, each 1.8 km across and ticking at about 50 kHz. FCC, the high-radix analog of Tofu-style meshes, routes with a stretch of 1.21, against 1.49 for a 3D torus.
- The comm leakage of the optimum is a sharp-edged 10% band adding about 12% to the starlight escaping at 1 µm. The 39% diffraction spill of a Fresnel-matched link is derived from the prolate-spheroidal eigenvalue λ₀(π/2)² = 0.61.
- One fiducial cell draws 43 W and has a mass of 0.34 g. It performs 2.5×10²² bit-ops/s, about 0.85 EFLOPS-equivalent at an assumed 3×10⁴ bit-ops per flop, or roughly half of El Capitan (TOP500, June 2026). The cloud is assumed to run at the Landauer limit, while El Capitan spends about 2×10⁵ kT ln2 per bit-op. `cells.json → terrestrial_comparison` has the numbers.
- Workloads by level are in `cells.json → workloads`:
  - **Nodes** (2×10⁻⁵ bits per op) handle dense kernels, search and sampling.
  - **Cells** (10⁻⁶) handle stencils, linear algebra and model-parallel ML. A cloud-wide stencil is latency-bound at about 10⁸ points per node, and simulated signals move at about 250 km/s.
  - **The whole cloud** is a bulk-synchronous machine with 9–26 d global supersteps (5 d at the light-crossing bound of a dense backbone), about 4×10⁵ of them per AGB lifetime. That favors wide, shallow work: ensembles, about 2¹⁹³-trial searches, island evolution, map-reduce. The superstep cadence survives light-travel smoothing (66–98%).

### 6. Minds as a workload (`blackcloud/minds.py`, `python run_minds.py` → `minds.json`)
- A synapse-level, real-time human-brain emulation costs about 2.5×10¹⁸ bit-ops/s (range 1.6×10¹⁷–8×10¹⁸). Inputs are 8.6×10¹⁰ neurons, 1.6×10¹⁴ neocortical synapses, a mean rate ≤ 0.5 Hz, and 3×10⁴ bit-ops per synaptic event. At the Landauer limit that is 4 mW, against the brain's 20 W.
- Its spike traffic is about 6×10⁻⁷ bits per op, essentially the fiducial β = 10⁻⁶, so minds are a natural cell-level workload. A node hosts about 7 minds, a cell about 10⁴, and the cloud about 2×10³² (2×10³⁶ mind-years over the AGB lifetime).
- Relay latency caps a mind's speed-up near 600× real time. At that speed a global superstep lasts about 16 subjective years, so a mind civilization is an archipelago of town-sized cells. Its activity is cell-local and leaves no time-domain signature.

## Observing strategy suggested by the toy model
1. Select WISE red, ~175 K sources with high luminosity (already saturated at 22 µm within about 1 kpc).
2. Get sub-mm continuum at two frequencies. α ≈ 2 against α ≈ 3 tests whether the source is gray.
3. Take a broadband ALMA spectral scan to look for flat-topped plateaus of tens of GHz with sharp edges that match no molecular transition.
4. Monitor that band over days to weeks for bursts between 2R/c and the Mira period. In the mid-IR, look for achromatic variability across the Mira cycle using two-band MIRI or ground N/Q-band monitoring.

## Caveats (it's a toy)
- The shell is one thin, spherically symmetric, isothermal layer. There is no Matryoshka layering, clumping or partial covering beyond a single covering fraction.
- The natural-envelope comparison is optically thin with a single β. Optically thick natural shells broaden the SED further.
- The link model idealises nodes as apertures with Friis/Fresnel scaling, sets the noise floor to T_cloud, and charges about 1 kT ln2 per relay. Reversible or ballistic relaying would change the energy numbers but not the latencies.
- The sensitivities in `observe.SENSITIVITY_JY` are order-of-magnitude only.
- The design choices (band, SNR, thermostat or not, backbone or not) are the dominant uncertainty. The model tells you what each choice would leak, not which choice an engineer would make.
