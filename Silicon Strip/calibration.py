import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

charges = []
adcs = []
files = [28,52,76,100,124]

for i in files:  
    filename = f"measurement/Calibration/{i}.txt"

    with open(filename, "r") as f:
        text = f.read().replace(",", ".")

    data = np.genfromtxt(StringIO(text))

    charge = data[:, 0]
    adc = data[:, 0]

    charges.append(charge)
    adcs.append(adc)

for i in range(0,5):
    strip= 28+i*24
    plt.plot(charges[i],adcs[i],'o', markersize=3,alpha=0.2, label=f"Strip {strip}")
plt.ylabel(r'$\text{ADC}$')
plt.xlabel(r'$Q_{\text{ind}} \mathbin{/} \si{\coulomb}$')
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/calibration.pdf')
plt.close()

adc_mean = adcs[0:4]
adcs_mean = np.mean(adc_mean,axis=0)
for i in range(0,4):
    strip= 28+i*24
    plt.plot(charges[i],adcs[i],'o', markersize=3,alpha=0.2, label=f"Strip {strip}")
plt.plot(charges[0],adcs_mean,linestyle='-',color='tab:red',label='Mean')
plt.ylabel(r'$\text{ADC}$')
plt.xlabel(r'$Q_{\text{ind}} \mathbin{/} \si{\coulomb}$')
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/calibration_fix.pdf')
plt.close()
a_4 = []
a_3 = []
a_2 = []
a_1 = []
a_0 = []
for i in range(4):
    coeffs = np.polyfit(charges[i], adcs[i], 4)

    a_4.append(coeffs[0])
    a_3.append(coeffs[1])
    a_2.append(coeffs[2])
    a_1.append(coeffs[3])
    a_0.append(coeffs[4])

    strip = 28+i*24
    print(f"Values for calibration curve {strip}:")
    print(f"a_4 = {a_4[i]}")
    print(f"a_3 = {a_3[i]}")
    print(f"a_2 = {a_2[i]}")
    print(f"a_1 = {a_1[i]}")
    print(f"a_0 = {a_0[i]}")
plt.close()


charges = []
adcs = []
files = [28,52,76,100,124]

filename = f"measurement/Calibration/0V_Bias.txt"

with open(filename, "r") as f:
    text = f.read().replace(",", ".")

data = np.genfromtxt(StringIO(text))

charge = data[:, 0]
adc = data[:, 1]
charges.append(charge)
adcs.append(adc)

plt.plot(charge,adc,'o', markersize=3,alpha=0.2, label="0 V bias")
plt.plot(charges[0],adcs_mean,linestyle='-',color='tab:red',label='Mean curve')
plt.ylabel(r'$\text{ADC}$')
plt.xlabel(r'$Q_{\text{ind}} \mathbin{/} \si{\coulomb}$')
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/calibration_0.pdf')
plt.close()