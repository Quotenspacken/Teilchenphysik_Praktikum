import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

h0 = 10
def exp_model(x, I0, Lambda):
    return I0 * np.exp(-x / Lambda)

folder = "intensity_Sebastian_Robin"

h = 10
v_values = np.arange(0, 36 + 4, 4)
x_values = np.arange(0, 2280 + 120, 120)

plt.figure(figsize=(8,6))

# Colormap (rot-Verlauf)
cmap = plt.cm.Reds
colors = cmap(np.linspace(0.3, 1, len(v_values)))

for idx, v in enumerate(v_values):
    I_values = []

    for x in x_values:
        filename = f"{folder}/Attenuation_h={h:g}deg_v={v:g}deg_x={x:g}mm.txt"
        dark_filename = f"{folder}/DarkCounts_h={h:g}deg_v={v:g}deg.txt"

        _, counts = np.genfromtxt(filename, unpack=True)
        _, dark_counts = np.genfromtxt(dark_filename, unpack=True)

        I_clean = counts - dark_counts
        I_total = np.sum(I_clean)

        I_values.append(I_total)

    I_values = np.array(I_values)

    # Fit
    popt, _ = curve_fit(exp_model, x_values, I_values, p0=[np.max(I_values), 1000])

    # Plot Daten
    plt.scatter(x_values, I_values, s=20, color=colors[idx])

    # Plot Fit
    x_fit = np.linspace(min(x_values), max(x_values), 300)
    plt.plot(x_fit, exp_model(x_fit, *popt),
             color=colors[idx],
             label=fr"$v={v}^\circ,\ \Lambda={popt[1]:.0f}\,\mathrm{{mm}}$")

# Achsen & Layout
plt.xlabel(r"$x \,/\, \mathrm{mm}$")
plt.ylabel("Intensität")
plt.legend(loc="upper right", fontsize=9)
plt.title(rf"Messung bei $h={h0}^\circ$")

# Layout fix (wie bei dir)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

# Speichern
plt.savefig('build/intensity.pdf')