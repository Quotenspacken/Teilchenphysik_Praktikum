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
theta_clad_krit = np.degrees(np.arccos(n_clad2/n_core))

print(theta_clad_krit)
print(theta_core_krit)


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
    label="Core Photonen"
)

plt.hist(
    clad["theta"],
    bins=100,
    histtype="step",
    color= "orange",
    label="Cladding Photonen"
)

# kritische Winkel einzeichnen
plt.axvline(
    theta_core_krit,
    linestyle="--",
    label=r"$\theta_{\mathrm{krit,core}}$"
)

plt.axvline(
    theta_clad_krit,
    linestyle="--",
    color='orange',
    label=r"$\theta_{\mathrm{krit,clad}}$"
)
plt.xlabel(r"$\theta \mathbin{/} \si{\degree}$")
plt.ylabel(r"Intensität $\mathbin{/}$ Counts")
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

r_core= 0.22/2 #Core radius 
r_clad = r_core+0.0075

fig, axs = plt.subplots(2, 1, figsize=(7,10), sharex=False)

# Funktion für Grenzlinie
def r_min_boundary(theta_deg, theta_krit_deg, r):
    theta = np.radians(theta_deg)
    theta_krit = np.radians(theta_krit_deg)

    argument = 1 - (np.sin(theta_krit)**2 / np.sin(theta)**2)

    # unphysikalische Werte vermeiden
    argument = np.where(argument >= 0, argument, np.nan)

    return r * np.sqrt(argument)


# Winkelbereiche für Grenzlinien
theta_core_line = np.linspace(theta_core_krit, core["theta"].max(), 500)
theta_clad_line = np.linspace(theta_clad_krit, clad["theta"].max(), 500)


r_core_line = r_min_boundary(theta_core_line, theta_core_krit, r_core)
r_clad_line = r_min_boundary(theta_clad_line, theta_clad_krit, r_clad)



# --- Core ---
h0 = axs[0].hist2d(core["theta"], core["r_min"], bins=100)
axs[0].plot(
    theta_core_line,
    r_core_line,
    "r--",
    linewidth=2,
    label=r"Grenzlinie bei $\theta_{\text{refl}}=\theta_{\text{krit, core}}$"
)

axs[0].set_xlabel(r"$\theta \mathbin{/} \si{\degree}$")
axs[0].set_ylabel(r"$r_{\min} \mathbin{/} \si{\milli\meter}$")
axs[0].set_title("Core")
axs[0].legend(loc="upper left")

cbar0 = fig.colorbar(h0[3], ax=axs[0])
cbar0.set_label(r"Intensität $\mathbin{/}$ Counts")


# --- Cladding ---
h1 = axs[1].hist2d(clad["theta"], clad["r_min"], bins=100)
axs[1].plot(
    theta_clad_line,
    r_clad_line,
    "r--",
    linewidth=2,
    label=r"Grenzlinie bei $\theta_{\text{refl}}=\theta_{\text{krit, clad}}$"
)
axs[1].set_xlabel(r"$\theta \mathbin{/} \si{\degree}$")
axs[1].set_ylabel(r"$r_{\min} \mathbin{/} \si{\milli\meter}$")
axs[1].set_title("Cladding")
axs[1].legend(loc="upper left")

cbar1 = fig.colorbar(h1[3], ax=axs[1])
cbar1.set_label(r"Intensität $\mathbin{/}$ Counts")

plt.tight_layout()
plt.savefig('build/hist2d_core_clad.pdf')

######## Attenuation length ##########
# Winkel berechnen
# theta-Werte
theta_values = np.arange(0, 37, 4)
dtheta = 2

# Fit-Funktion
def exp_model(x, I0, Lambda):
    return I0 * np.exp(-x / Lambda)

# Colormap
cmap = plt.cm.Reds
colors = cmap(np.linspace(0.3, 1, len(theta_values)))

plt.figure(figsize=(8, 6))

for theta0, color in zip(theta_values, colors):

    # Daten filtern
    data = df_phy[
        (df_phy["theta"] >= theta0 - dtheta) &
        (df_phy["theta"] <  theta0 + dtheta)
    ]

    if len(data) < 10:
        continue

    # Counts pro Anregungsort bestimmen
    counts = data.groupby("gpsPosX").size()

    x = counts.index.to_numpy()
    y = counts.to_numpy()

    # nach x sortieren
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # genügend Punkte für Fit?
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

    except RuntimeError:
        continue

    # Fitkurve
    x_fit = np.linspace(np.min(x), np.max(x), 300)
    y_fit = exp_model(x_fit, I0_fit, Lambda_fit)

    # Plot: Punkte + Fitlinie
    plt.plot(x, y, "o", color=color)
    plt.plot(
        x_fit,
        y_fit,
        "-",
        color=color,
        label=rf"$\Theta={theta0}^\circ \pm {dtheta}^\circ$: "
              rf"$\Lambda={Lambda_fit:.0f}\,\mathrm{{mm}}$"
    )

plt.xlabel(r"$x = \mathrm{gpsPosX} \,/\, \mathrm{mm}$")
plt.ylabel(r"Intensität $\mathbin{/}$ Counts")
plt.title(r"Simulation für verschiedene Winkel $\Theta$ zur Faserachse")

plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()

plt.savefig("build/atten.pdf")

