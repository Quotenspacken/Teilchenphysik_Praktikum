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

# neues Histogramm

plt.figure(figsize=(7,5))

plt.hist2d(core["theta"], core["r_min"], bins=100)
plt.xlabel(r"$\theta$ (deg)")
plt.ylabel(r"$r_{\min}$ (mm)")
cbar = plt.colorbar()
cbar.set_label("Anzahl der Counts")
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/hist2d_core.pdf')

plt.figure(figsize=(7,5))

plt.hist2d(clad["theta"], clad["r_min"], bins=100)
plt.xlabel(r"$\theta$ (deg)")
plt.ylabel(r"$r_{\min}$ (mm)")
cbar = plt.colorbar()
cbar.set_label("Anzahl der Counts")
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/hist2d_clad.pdf')

theta_bins = [(i, i+5) for i in range(0, 40, 5)]


theta_centers = []
lambdas = []
lambda_errors = []

for theta_min, theta_max in theta_bins:
    data = df_phy[
        (df_phy["theta"] >= theta_min) &
        (df_phy["theta"] < theta_max)
    ]

    counts = data.groupby("gpsPosX").size()

    # genug Punkte?
    if len(counts) < 5:
        print(f"Skipping {theta_min}-{theta_max} (zu wenig Daten)")
        continue

    x = counts.index.to_numpy()
    y = counts.to_numpy()

    # nur positive Werte (wegen log)
    mask = y > 0
    x = x[mask]
    y = y[mask]

    if len(y) < 5:
        print(f"Skipping {theta_min}-{theta_max} (zu wenig valide Punkte)")
        continue

    # Logarithmieren
    log_y = np.log(y)

    # Fehler: Poisson → sigma_y = sqrt(y)
    sigma_y = np.sqrt(y)
    sigma_log = sigma_y / y   # Fehlerfortpflanzung

    # lineares Modell
    def lin_model(x, a, b):
        return a + b * x

    try:
        popt, pcov = curve_fit(
            lin_model,
            x,
            log_y,
            sigma=sigma_log,
            absolute_sigma=True
        )

        a, b = popt

        Lambda = -1 / b
        Lambda_err = np.sqrt(pcov[1,1]) / (b**2)

        theta_center = 0.5 * (theta_min + theta_max)

        theta_centers.append(theta_center)
        lambdas.append(Lambda)
        lambda_errors.append(Lambda_err)

        print(f"{theta_min:2d}-{theta_max:2d}° : Lambda = {Lambda:.2f} ± {Lambda_err:.2f} mm")

    except RuntimeError:
        print(f"Fit failed for {theta_min}-{theta_max}")


plt.figure(figsize=(7,5))

plt.errorbar(theta_centers, lambdas, yerr=lambda_errors, fmt='o', capsize=4)

plt.xlabel(r"$\theta$ (deg)")
plt.ylabel(r"$\Lambda_{\mathrm{eff}}$ (mm)")
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/sim.pdf')



