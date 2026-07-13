import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from scipy.signal import find_peaks


def load_txt_with_decimal_comma(filename, **kwargs):
    with open(filename, "r") as f:
        text = f.read().replace(",", ".")

    return np.genfromtxt(StringIO(text), **kwargs)


laser_scan = np.genfromtxt("measurement/Laserscan.txt")
positions = np.arange(10, 10 * (laser_scan.shape[0] + 1), 10)
strips = np.arange(laser_scan.shape[1])

sync = load_txt_with_decimal_comma("measurement/Laser/Sync_1.txt", skip_header=2)
delay = sync[:, 0]
sync_signal = sync[:, 1]
optimal_delay = delay[np.nanargmax(sync_signal)]

strip_max = np.max(laser_scan, axis=0)
signal_strips = strips[strip_max > 80]
positive_signal = np.clip(laser_scan[:, signal_strips], 0, None)
summed_signal = np.sum(positive_signal, axis=1)

peak_indices, _ = find_peaks(summed_signal, distance=2, height=20)
peak_positions = positions[peak_indices]
pitch = np.mean(np.diff(peak_positions))
pitch_std = np.std(np.diff(peak_positions), ddof=1)

laser_widths = []
for strip in signal_strips:
    signal = laser_scan[:, strip]
    if np.max(signal) < 80:
        continue
    half_max = np.max(signal) / 2
    above_half = positions[signal >= half_max]
    if len(above_half) > 1:
        laser_widths.append(above_half[-1] - above_half[0])
laser_width = np.mean(laser_widths)

print(f"Laser scan data: {laser_scan.shape[0]} positions, {laser_scan.shape[1]} strips")
print(f"Optimal laser delay: {optimal_delay:.0f} ns")
print(f"Relevant strips: {signal_strips.tolist()}")
print(f"Signal maxima spacing: ({pitch:.1f} +- {pitch_std:.1f}) um")
print(f"Laser FWHM from focused strip peaks: about {laser_width:.0f} um")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for strip in signal_strips:
    axes[0].plot(positions, laser_scan[:, strip], "-", linewidth=0.8, label=f"{strip}")
axes[0].set_xlabel(r"Laser position $\mathbin{/} \si{\micro\meter}$")
axes[0].set_ylabel(r"$\text{ADC}$")
axes[0].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    fontsize="x-small",
    title="Strip",
)

axes[1].plot(
    positions,
    summed_signal,
    "o-",
    markersize=3,
    linewidth=0.8,
    label="Summed signal",
)
axes[1].plot(
    peak_positions,
    summed_signal[peak_indices],
    "x",
    color="tab:red",
    label="Detected maxima",
)
axes[1].set_xlabel(r"Laser position $\mathbin{/} \si{\micro\meter}$")
axes[1].set_ylabel(r"Summed signal $\mathbin{/} \text{ADC}$")
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

# in matplotlibrc leider (noch) nicht möglich
fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig("build/laser.pdf")
plt.close(fig)
