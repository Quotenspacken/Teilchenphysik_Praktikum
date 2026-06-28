#test
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.ylabel(r'$I \mathbin{/} \mathrm{(counts\ s^{-1})}$')
plt.xlabel(r'$x \mathbin{/} \mathrm{mm}$')
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/x_scan.pdf')