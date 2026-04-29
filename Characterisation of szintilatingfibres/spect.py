import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

_, dc_off, c1_off = np.genfromtxt("light_off.txt", unpack=True)
lam, dc_on, c1_on = np.genfromtxt("light_on.txt", unpack=True)

fig, axs = plt.subplots(2, 1, figsize=(6,8), sharex=False)

axs[0].plot(lam, c1_off, "r,", label="ohne Raumlicht")
axs[0].set_ylabel("Intensität")
axs[0].set_xlabel(r"$\lambda \,/\, \mathrm{nm}$")
axs[0].legend()

axs[1].plot(lam, c1_on, "b,", label="mit Raumlicht")
axs[1].set_xlabel(r"$\lambda \,/\, \mathrm{nm}$")
axs[1].set_ylabel("Intensität")
axs[1].legend()

# Layout
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/spectra.pdf')
plt.close()

I_off = c1_off - dc_off
I_on = c1_on - dc_on

fig, axs = plt.subplots(2, 1, figsize=(6,8), sharex=False)

axs[0].plot(lam, I_off, "r,", label="ohne Raumlicht, ohne DC")
axs[0].set_xlabel(r"$\lambda \,/\, \mathrm{nm}$")
axs[0].set_ylabel("Intensität ohne Dark Counts")
axs[0].legend()

axs[1].plot(lam, I_on, "b,", label="mit Raumlicht, ohne DC")
axs[1].set_xlabel(r"$\lambda \,/\, \mathrm{nm}$")
axs[1].set_ylabel("Intensität ohne Dark Counts")
axs[1].legend()
# Layout
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/spectra_dc.pdf') 
plt.close()