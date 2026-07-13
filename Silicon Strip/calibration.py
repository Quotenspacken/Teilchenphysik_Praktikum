import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

calibration_strips = [28, 52, 76, 100, 124]
used_calibration_strips = [28, 52, 76, 100]
excluded_calibration_strips = [124]
delay_strips = [28, 100]


def load_calibration_txt(filename):
    with open(filename, "r") as f:
        text = f.read().replace(",", ".")

    return np.genfromtxt(StringIO(text))


delay_data = {}
best_delays = {}
for strip in delay_strips:
    data = load_calibration_txt(f"measurement/Calibration/{strip}_t.txt")
    delay_data[strip] = data

fig, ax = plt.subplots()
for strip, data in delay_data.items():
    delay = data[:, 0]
    adc = data[:, 1]
    best_delay = delay[np.argmax(adc)]
    best_delays[strip] = best_delay
    print(f"Optimal delay for strip {strip}: {best_delay:.0f} ns")
    ax.plot(
        delay,
        adc,
        "o",
        markersize=3,
        alpha=0.5,
        label=rf"Strip {strip}, $t_\text{{delay}} = \SI{{{best_delay:.0f}}}{{\nano\second}}$",
    )
    ax.axvline(best_delay, linestyle="--", alpha=0.4)

ax.set_xlabel(r"$t \mathbin{/} \si{\nano\second}$")
ax.set_ylabel(r"$\text{ADC}$")
ax.legend(loc="best")
fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig("build/calibration_delay.pdf")
plt.close(fig)

charges = []
adcs = []

for strip in calibration_strips:
    data = load_calibration_txt(f"measurement/Calibration/{strip}.txt")

    charge = data[:, 0]
    adc = data[:, 1]

    charges.append(charge)
    adcs.append(adc)

charges = np.array(charges)
adcs = np.array(adcs)
used_indices = [
    calibration_strips.index(strip)
    for strip in used_calibration_strips
]
charge = charges[0]
adc_mean = np.mean(adcs[used_indices], axis=0)

fig, ax = plt.subplots()
for i, strip in enumerate(calibration_strips):
    if strip in excluded_calibration_strips:
        label = f"Strip {strip} (excluded)"
        alpha = 0.12
    else:
        label = f"Strip {strip}"
        alpha = 0.25
    ax.plot(charges[i], adcs[i], "o", markersize=3, alpha=alpha, label=label)
ax.plot(charge, adc_mean, "-", color="tab:red", label="Mean without strip 124")
plt.ylabel(r'$\text{ADC}$')
plt.xlabel(r'$Q_{\text{ind}} \mathbin{/} e$')
plt.legend(loc="best")
# in matplotlibrc leider (noch) nicht möglich
fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig("build/calibration.pdf")
plt.close(fig)

charge_fit = np.linspace(np.min(charge), np.max(charge), 500)
adc_fit = {}
coefficients = {}

fig, ax = plt.subplots()
for strip in used_calibration_strips:
    i = calibration_strips.index(strip)
    coeffs = np.polyfit(charges[i], adcs[i], 4)
    coefficients[strip] = coeffs
    adc_fit[strip] = np.polyval(coeffs, charge_fit)
    ax.plot(charges[i], adcs[i], "o", markersize=3, alpha=0.15, label=f"Strip {strip}")
    ax.plot(charge_fit, adc_fit[strip], "-", alpha=0.6)
    print(f"Values for ADC(Q) calibration curve {strip}:")
    print(f"a_4 = {coeffs[0]}")
    print(f"a_3 = {coeffs[1]}")
    print(f"a_2 = {coeffs[2]}")
    print(f"a_1 = {coeffs[3]}")
    print(f"a_0 = {coeffs[4]}")

mean_coeffs = np.polyfit(charge, adc_mean, 4)
mean_adc_fit = np.polyval(mean_coeffs, charge_fit)
ax.plot(charge, adc_mean, "o", markersize=3, color="tab:red", alpha=0.35, label="Mean without strip 124")
ax.plot(charge_fit, mean_adc_fit, "-", color="tab:red", label="Mean fit")
print("Values for mean ADC(Q) calibration curve without strip 124:")
print(f"a_4 = {mean_coeffs[0]}")
print(f"a_3 = {mean_coeffs[1]}")
print(f"a_2 = {mean_coeffs[2]}")
print(f"a_1 = {mean_coeffs[3]}")
print(f"a_0 = {mean_coeffs[4]}")

ax.set_xlabel(r"$Q_{\text{ind}} \mathbin{/} e$")
ax.set_ylabel(r"$\text{ADC}$")
ax.legend(loc="best")
fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig("build/calibration_fix.pdf")
plt.close(fig)

data_0v = load_calibration_txt("measurement/Calibration/0V_Bias.txt")
charge_0v = data_0v[:, 0]
adc_0v = data_0v[:, 1]
reference_strip = 100
reference_index = calibration_strips.index(reference_strip)

fig, ax = plt.subplots()
ax.plot(
    charges[reference_index],
    adcs[reference_index],
    "o",
    markersize=3,
    alpha=0.25,
    label=fr"Strip {reference_strip}, above $U_\text{{dep}}$",
)
ax.plot(charge_0v, adc_0v, "o", markersize=3, alpha=0.25, label=r"$0\,\si{\volt}$ bias")
ax.set_ylabel(r"$\text{ADC}$")
ax.set_xlabel(r"$Q_{\text{ind}} \mathbin{/} e$")
ax.legend(loc="best")
# in matplotlibrc leider (noch) nicht möglich
fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig("build/calibration_0.pdf")
plt.close(fig)
