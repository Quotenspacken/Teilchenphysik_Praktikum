#test
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

#Simulierte Daten importieren
df = pd.read_pickle("SimData.pkl")
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