"""Shared figure style: every figure is drawn at the manuscript's text width, so the
fonts below are the sizes that appear on the printed page (main text is 12 pt)."""
import matplotlib.pyplot as plt

W = 5.4            # elsarticle preprint \textwidth = 390 pt = 5.4 in
WC = 3.48          # one column of the two-column (5p) layout, 252 pt: single-panel figures with axes
FONT = 10          # axis labels and tick labels
SMALL = 9          # legends and in-plot text
TITLE = 10

# Text that has to sit on data gets the same 80% white background as the legends.
BOX = dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.15)


def apply():
    plt.rcParams.update({
        'figure.dpi': 200, 'savefig.dpi': 300, 'font.size': FONT,
        'axes.titlesize': TITLE, 'axes.labelsize': FONT,
        'xtick.labelsize': SMALL, 'ytick.labelsize': SMALL,
        'axes.grid': True, 'grid.alpha': 0.3,
        'legend.frameon': True, 'legend.facecolor': 'white', 'legend.framealpha': 0.8,
        'legend.edgecolor': 'none', 'legend.fontsize': SMALL, 'legend.labelspacing': 0.32,
    })


def sci(x, digits=1):
    """Mathtext scientific notation, e.g. 1.8e19 -> $1.8\\times10^{19}$ (digits=0 drops the mantissa if 1)."""
    m, e = f'{x:.{digits}e}'.split('e')
    e = int(e)
    return rf'$10^{{{e}}}$' if m in ('1', '1.0') else rf'${m}\times10^{{{e}}}$'


def save(fig, path, pad=8):
    """Crop to the drawn content so the figure runs margin to margin in the paper.

    bbox_inches='tight' trims around axes but not the empty interior of 3D axes, so the
    saved image is also cropped to its non-white pixels (plus `pad` pixels)."""
    import numpy as np
    from PIL import Image
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    im = Image.open(path)
    rgb = np.asarray(im.convert('RGB'))
    ys, xs = np.nonzero((rgb < 250).any(axis=2))
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad + 1, rgb.shape[1]), min(ys.max() + pad + 1, rgb.shape[0]))
    im.crop(box).save(path, dpi=im.info.get('dpi', (300, 300)))
