"""pub_style.py — Publication-quality matplotlib settings."""
import matplotlib as mpl

DPI = 300

def apply():
    mpl.rcParams.update({
        'figure.dpi':          DPI,
        'savefig.dpi':         DPI,
        'font.family':         'sans-serif',
        'font.sans-serif':     ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size':           11,
        'axes.titlesize':      13,
        'axes.titleweight':    'bold',
        'axes.labelsize':      12,
        'axes.linewidth':      0.8,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'xtick.major.width':   0.8,
        'ytick.major.width':   0.8,
        'xtick.minor.width':   0.5,
        'ytick.minor.width':   0.5,
        'lines.linewidth':     1.5,
        'legend.fontsize':     10,
        'legend.framealpha':   0.9,
        'legend.edgecolor':    '0.8',
        'figure.facecolor':    'white',
        'axes.facecolor':      'white',
        'axes.grid':           False,
        'grid.alpha':          0.3,
        'grid.linewidth':      0.5,
        'image.interpolation': 'nearest',
        'pdf.fonttype':        42,   # embed fonts in PDF
        'ps.fonttype':         42,
    })
