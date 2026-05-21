import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Achsen & Layout
plt.xlabel(r"$x \mathbin{/} \si{\milli\meter}$")
plt.ylabel(r"y $\mathbin{/}$ \si{\milli\meter}")
plt.legend(loc="best")


# Layout fix (wie bei dir)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

# Speichern
plt.savefig('build/TCT.pdf')