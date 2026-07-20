import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from pathlib import Path

header = Path(__file__).parent / "header-matplotlib.tex"

mpl.rcParams["pgf.preamble"] = rf"\input{{{header.as_posix()}}}"

print("Backend:", plt.get_backend())
print("usetex:", plt.rcParams["text.usetex"])
print("preamble:", plt.rcParams["pgf.preamble"])

voltages = []
cce_signals = []
hit_strip = 84

for voltage in range(0, 201, 10):
    signal = np.genfromtxt(f"measurement/CCEL/{voltage:02d}CCEL.txt")
    voltages.append(voltage)
    cce_signals.append(signal[hit_strip])

voltages = np.array(voltages, dtype=float)
cce_signals = np.array(cce_signals)

plateau_signal = np.mean(cce_signals[voltages >= 120])
cce = cce_signals / plateau_signal

sensor_thickness = 300.0
depletion_voltage = 70.0


def cce_model(voltage, a, plateau_height, plateau_slope):
    voltage = np.asarray(voltage)
    depletion_depth = sensor_thickness * np.sqrt(
        np.minimum(voltage, depletion_voltage) / depletion_voltage
    )
    below_depletion = plateau_height * (
        1 - np.exp(-depletion_depth / a)
    ) / (
        1 - np.exp(-sensor_thickness / a)
    )
    above_depletion = plateau_height + plateau_slope * (voltage - depletion_voltage)
    return np.where(voltage <= depletion_voltage, below_depletion, above_depletion)


fit_mask = voltages > 0
fit_params, fit_cov = curve_fit(
    cce_model,
    voltages[fit_mask],
    cce[fit_mask],
    p0=[300, 0.95, 5e-4],
    bounds=([1, 0, -1], [5000, 2, 1]),
)
a = fit_params[0]
plateau_height = fit_params[1]
plateau_slope = fit_params[2]
a_err, plateau_height_err, plateau_slope_err = np.sqrt(np.diag(fit_cov))
voltage_fit = np.linspace(0, np.max(voltages), 500)

print(f"CCE hit strip: {hit_strip}")
print(f"CCE plateau signal: {plateau_signal:.2f} ADC")
print(f"Fit parameter a: ({a:.0f} +- {a_err:.0f}) um")
print(f"Fitted plateau height: {plateau_height:.3f} +- {plateau_height_err:.3f}")
print(f"Fitted plateau slope: ({plateau_slope:.4f} +- {plateau_slope_err:.4f}) / V")

fig, ax = plt.subplots()
ax.plot(voltages, cce, "o", markersize=3, label="Measured CCE")
ax.plot(
    voltage_fit,
    cce_model(voltage_fit, a, plateau_height, plateau_slope),
    "-",
    label="CCE fit",
)
ax.axvline(
    depletion_voltage,
    linestyle="--",
    alpha=0.5,
    color="grey",
    label=r"$U_\text{dep}$ cut",
)
ax.set_xlabel(r"$U \mathbin{/} \si{\volt}$")
ax.set_ylabel("Relative CCE")
ax.legend(loc="lower right")

# in matplotlibrc leider (noch) nicht möglich
fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig("build/CCE.pdf")
plt.close(fig)
