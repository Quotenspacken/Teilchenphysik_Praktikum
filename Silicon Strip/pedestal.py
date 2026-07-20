import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

header = Path(__file__).parent / "header-matplotlib.tex"

mpl.rcParams["pgf.preamble"] = rf"\input{{{header.as_posix()}}}"

adc = np.genfromtxt(
    "measurement/Pedestal.txt",
    delimiter=";"
)

if adc.shape == (1000, 128):
    adc = adc.T
elif adc.shape != (128, 1000):
    raise ValueError(f"Unexpected pedestal data shape: {adc.shape}")

n_strips, n_events = adc.shape
strips = np.arange(n_strips)

pedestal = np.mean(adc, axis=1)
common_mode = np.mean(adc - pedestal[:, np.newaxis], axis=0)
noise = np.sqrt(
    np.sum((adc - pedestal[:, np.newaxis] - common_mode[np.newaxis, :])**2, axis=1)
    / (n_events - 1)
)

print(f"Pedestal data: {n_strips} strips, {n_events} events")
print(
    "Pedestal / ADC counts: "
    f"mean = {np.mean(pedestal):.2f}, "
    f"min = {np.min(pedestal):.2f}, "
    f"max = {np.max(pedestal):.2f}"
)
print(
    "Noise / ADC counts: "
    f"mean = {np.mean(noise):.2f}, "
    f"min = {np.min(noise):.2f}, "
    f"max = {np.max(noise):.2f}"
)
print(
    "Common mode / ADC counts: "
    f"mean = {np.mean(common_mode):.2f}, "
    f"std = {np.std(common_mode, ddof=1):.2f}"
)

fig, (ax_pedestal, ax_common_mode) = plt.subplots(
    2,
    1,
    figsize=(10, 6),
    height_ratios=[1.3, 1],
)

ax_pedestal.errorbar(
    strips,
    pedestal,
    yerr=noise,
    fmt=".",
    capsize=2,
    label="Pedestal ± noise"
)
ax_pedestal.set_xlabel("Strip number")
ax_pedestal.set_ylabel("ADC counts")
ax_pedestal.legend(loc="best")

ax_common_mode.hist(common_mode, bins=35, histtype="stepfilled", alpha=0.7)
ax_common_mode.axvline(0, linestyle="--", alpha=0.5, color="grey")
ax_common_mode.set_xlabel("Common mode / ADC counts")
ax_common_mode.set_ylabel("Number of events")

fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig("build/pedestal.pdf")
