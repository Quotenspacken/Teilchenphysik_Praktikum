import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
V, I = np.genfromtxt('IV.txt',unpack = True)
linearArea = (V>=70)&(V<=150)
m, b = np.polyfit(V[linearArea], I[linearArea], 1)
V_fit = np.linspace(70,150,200)
I_fit = m*V_fit+b
plt.axvline(70, linestyle='--' ,alpha=0.4,color = 'grey', label = 'Cut of linearity')
plt.axvline(150, linestyle='--' ,alpha=0.4,color = 'grey')
plt.plot(V,I,'x',label = 'Measured values')
plt.plot(V_fit,I_fit,'-',alpha=0.4,label = 'linear regression')
plt.ylabel(r'$I \mathbin{/} \si{\micro\ampere}$')
plt.xlabel(r'$V \mathbin{/} \si{\volt}$')
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/IV.pdf')