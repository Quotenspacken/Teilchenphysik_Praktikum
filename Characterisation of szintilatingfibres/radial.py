import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

h_values = np.arange(-18, 35, 3.5)
v_values = np.arange(-6, 35, 2.7)

I_map = np.zeros((len(h_values), len(v_values)))

for i, h in enumerate(h_values):
    for j, v in enumerate(v_values):
        
        filename = f"radial_Sebastian_Robin/Attenuation_h={h:g}deg_v={v:g}deg_x=0mm.txt"
        dark_filename = f"radial_Sebastian_Robin/DarkCounts_h={h:g}deg_v={v:g}deg.txt"
        _, counts = np.genfromtxt(filename, unpack=True)

        _, d_counts = np.genfromtxt(dark_filename, unpack=True)
        counts_fixed = counts - d_counts

        I_total = np.sum(counts_fixed)

        I_map[i, j] = I_total

H, V = np.meshgrid(h_values, v_values, indexing="ij")

plt.pcolormesh(H, V, I_map, shading="auto")
plt.xlabel("horizontal angle h / deg")
plt.ylabel("vertical angle v / deg")
plt.colorbar(label="Intensität")
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/radial.pdf')
 


