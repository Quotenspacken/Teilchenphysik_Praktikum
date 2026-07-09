import numpy as np
import matplotlib.pyplot as plt

adc = np.genfromtxt('measurement/Laserscan.txt', unpack = True)

print(adc.shape)
laserStep = np.arange(10,360,10)
strip = np.arange(adc.shape[0])
meanAdc = []
for i in range(0,35):
    mean = np.mean(adc[:,i], axis=0)
    meanAdc.append(mean)
for i in range(0,128):
    plt.plot(laserStep, adc[i,:], '-')
plt.ylabel(r'$\text{ADC}$')
plt.xlabel(r'$Q_{\text{ind}} \mathbin{/} \si{\coulomb}$')
plt.legend(loc='best')
# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/laser.pdf')