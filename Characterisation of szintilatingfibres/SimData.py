import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

#Simulierte Daten importieren
df = pd.read_pickle("../../SimData.pkl")
print(df.columns)

# Radius Faser in mm definieren 
r_fiber = 0.25/2
 
# Brechungsindex
n_core = 1.6
n_clad1 = 1.49
n_clad2 = 1.42

#Kritische Winkel
theta_core_krit = np.degrees(np.arccos(n_clad1/n_core))
theta_clad_krit = np.degrees(np.arccos(n_clad2/n_clad1))

#Radialen Exit definieren
df["r_exit"] = np.sqrt(df["# y_exit"]**2+df["z_exit"]**2)

#unphysikalische Photonen außerhalb entfernen + filtern von Rayleigh Streuung
df_phy = df[(df["r_exit"]<=r_fiber) & (df["rayleighScatterings"]==0)].copy() #um CopywWarning zu beseitigen

df_phy["theta"] = np.degrees(np.arccos(np.clip(df_phy["px_start"],-1,1)))

core = df_phy[df_phy["length_clad"]==0].copy()
clad = df_phy[df_phy["length_clad"]>0].copy()

# Histogramm
plt.figure(figsize=(7, 5))
plt.hist(
    core["theta"],
    bins=100,
    histtype="step",
    label="Core Photons"
)

plt.hist(
    clad["theta"],
    bins=100,
    histtype="step",
    color= "orange",
    label="Cladding Photons"
)

# kritische Winkel einzeichnen
plt.axvline(
    theta_core_krit,
    linestyle="--",
    label=r"$\theta_{\mathrm{crit,core}}$"
)

plt.axvline(
    theta_clad_krit,
    linestyle="--",
    color='orange',
    label=r"$\theta_{\mathrm{crit,clad}}$"
)
plt.xlabel(r"$\theta \mathbin{/} \si{\degree}$")
plt.ylabel("Anzahl der Photonen")
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/SimData.pdf')


#minimaler Abstand zum Faserzentrum
def r_min_to_x_axis(data):
    y0 = data["y_start"].to_numpy()
    z0 = data["z_start"].to_numpy()
    py = data["py_start"].to_numpy()
    pz = data["pz_start"].to_numpy()

    numerator = np.abs(y0 * pz - z0 * py)
    denominator = np.sqrt(py**2 + pz**2)

    # Division durch 0 vermeiden
    r_min = np.zeros_like(numerator)

    mask = denominator > 0
    r_min[mask] = numerator[mask] / denominator[mask]

    r_min[~mask] = np.sqrt(y0[~mask]**2 + z0[~mask]**2)

    return r_min


# r_min berechnen
core["r_min"] = r_min_to_x_axis(core)
clad["r_min"] = r_min_to_x_axis(clad)

fig, axs = plt.subplots(1, 2, figsize=(14,5))

# --- Core ---
h0 = axs[0].hist2d(core["theta"], core["r_min"], bins=100)
axs[0].set_xlabel(r"$\theta$ (deg)")
axs[0].set_ylabel(r"$r_{\min}$ (mm)")
axs[0].set_title("Core")

cbar0 = fig.colorbar(h0[3], ax=axs[0])
cbar0.set_label("Anzahl der Counts")

# --- Cladding ---
h1 = axs[1].hist2d(clad["theta"], clad["r_min"], bins=100)
axs[1].set_xlabel(r"$\theta$ (deg)")
axs[1].set_ylabel(r"$r_{\min}$ (mm)")
axs[1].set_title("Cladding")

cbar1 = fig.colorbar(h1[3], ax=axs[1])
cbar1.set_label("Anzahl der Counts")

plt.tight_layout()
plt.savefig('build/hist2d_core_clad.pdf')

######## Attenuation length ##########
# Winkel berechnen
df_phy["h"] = np.degrees(np.arctan2(df_phy["py_start"], df_phy["px_start"]))
df_phy["v"] = np.degrees(np.arctan2(df_phy["pz_start"], df_phy["px_start"]))

# fixes h
h0 = 10
dh = 2

# v-Werte
v_values = np.arange(0, 37, 4)
dv = 2

# Fit-Funktion
def exp_model(x, I0, Lambda):
    return I0 * np.exp(-x / Lambda)

# Colormap (rot-Verlauf)
cmap = plt.cm.Reds
colors = cmap(np.linspace(0.3, 1, len(v_values)))

plt.figure(figsize=(8,6))

for v0, color in zip(v_values, colors):

    # Daten filtern
    data = df_phy[
        (df_phy["h"] >= h0 - dh) & (df_phy["h"] < h0 + dh) &
        (df_phy["v"] >= v0 - dv) & (df_phy["v"] < v0 + dv)
    ]

    if len(data) < 10:
        continue

    # Counts bestimmen
    counts = data.groupby("gpsPosX").size()

    x = counts.index.to_numpy()
    y = counts.to_numpy()

    # sortieren
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # nur sinnvolle Punkte (kein log-Fit nötig, aber stabiler)
    if len(x) < 5:
        continue

    # Fit
    try:
        popt, pcov = curve_fit(
            exp_model,
            x,
            y,
            p0=[np.max(y), 3000],
            maxfev=10000
        )

        I0_fit, Lambda_fit = popt
        Lambda_err = np.sqrt(np.diag(pcov))[1]

    except:
        continue

    # Fitkurve
    x_fit = np.linspace(np.min(x), np.max(x), 300)
    y_fit = exp_model(x_fit, I0_fit, Lambda_fit)

    # Plot: Punkte + Linie gleiche Farbe
    plt.plot(x, y, "o", color=color)
    plt.plot(
        x_fit,
        y_fit,
        "-",
        color=color,
        label=rf"$v={v0}^\circ$: $\Lambda={Lambda_fit:.0f}\,\mathrm{{mm}}$"
    )

plt.xlabel(r"$x = \mathrm{gpsPosX} \,/\, \mathrm{mm}$")
plt.ylabel("Counts")
plt.title(rf"Simulation für $h={h0}^\circ \pm {dh}^\circ$")

plt.legend(loc="upper right", fontsize=9)

plt.tight_layout()

plt.savefig("build/atten.pdf")




