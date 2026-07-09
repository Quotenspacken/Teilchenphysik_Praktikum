import numpy as np
import matplotlib.pyplot as plt

P = np.genfromtxt(
    "measurement/Pedestal.txt",
    delimiter=";"
)

print(P.shape)  # should be (1000, 128)

n_events, n_strips = P.shape
strips = np.arange(n_strips)

# Pedestal per strip: mean over all events
pedestal = np.mean(P, axis=0)

# Noise per strip: standard deviation over all events
noise = np.std(P, axis=0, ddof=1)

fig, ax = plt.subplots(figsize=(10, 5))

ax.errorbar(
    strips,
    pedestal,
    yerr=noise,
    fmt=".",
    capsize=2,
    label="Pedestal ± noise"
)

ax.set_xlabel("Strip number")
ax.set_ylabel("ADC counts")
ax.legend()

fig.tight_layout()
fig.savefig("build/pedestal.pdf")
