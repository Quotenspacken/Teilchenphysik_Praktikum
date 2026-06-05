import struct
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit



def read_drs4(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    pos = 0

    assert data[pos:pos+4] == b"DRS2", "Keine gueltige DRS4-Datei (Magic 'DRS2' fehlt)"
    pos += 4

    assert data[pos:pos+4] == b"TIME", "TIME-Block nicht gefunden"
    pos += 4
    pos += 4

    time_bins = {}
    while pos < len(data) and data[pos:pos+1] == b"C":
        ch_id = data[pos:pos+4].decode("ascii")
        pos += 4
        widths = np.array(struct.unpack_from("<1024f", data, pos))
        time_bins[ch_id] = np.concatenate([[0.0], np.cumsum(widths[:-1])])
        pos += 4096

    records = []
    event_num = 0

    while pos < len(data):
        if data[pos:pos+4] != b"EHDR":
            break
        pos += 4
        pos += 4
        pos += 16
        pos += 8

        event_num += 1

        while pos < len(data) and data[pos:pos+1] == b"C":
            ch_id = data[pos:pos+4].decode("ascii")
            pos += 4
            pos += 4

            raw = np.array(struct.unpack_from("<1024H", data, pos), dtype=np.float32)
            pos += 2048

            voltage = (raw / 65536.0 - 0.5) * 1.0
            t = time_bins.get(ch_id, np.arange(1024, dtype=float))

            records.append(pd.DataFrame({
                "event":     event_num,
                "channel":   ch_id,
                "sample":    np.arange(1024),
                "time_ns":   t,
                "voltage_V": voltage,
            }))

    df = pd.concat(records, ignore_index=True)
    return df

def extract_signal_window(grp, baseline_samples=1000, threshold_factor=500):
    v = grp["voltage_V"].values

    baseline = np.mean(v[:baseline_samples])
    noise    = np.std(v[:baseline_samples])

    peak_idx = np.argmax(np.abs(v - baseline))
    polarity = np.sign(v[peak_idx] - baseline)

    start = peak_idx
    while start > 0 and polarity * (v[start] - baseline) > noise:
        start -= 1

    end = peak_idx
    while end < len(v) - 1 and polarity * (v[end] - baseline) > noise:
        end += 1

    return grp.iloc[start:end + 1]

def integrate_charge(grp):
    t = grp["time_ns"].values * 1e-9  # ns -> s
    v = grp["voltage_V"].values
    return np.trapz(v, t) / R         # Q = integral(V/R dt) in Coulomb



daten_dir = Path("daten")

################################
### Bias Vergleich Rückseite ###
################################
dfs = []
for f in sorted(daten_dir.glob("3_*.dat")):
    # Bias-Spannung aus Dateiname extrahieren: "3_100V.dat" -> 100
    bias_str = f.stem.split("_")[1]
    bias_v   = int(bias_str.replace("V", ""))
    df_tmp = read_drs4(f)
    df_tmp["bias_V"] = bias_v
    dfs.append(df_tmp)
    print(f"Datei {f.name} mit Bias {bias_v} V eingelesen, {len(df_tmp)} Zeilen")

if not dfs:
    raise FileNotFoundError("Keine passenden 3_*.dat-Dateien in 'daten/' gefunden")

df = pd.concat(dfs, ignore_index=True)

# Daten sortieren und glätten
df = df.sort_values(["bias_V", "event", "channel", "sample"]).reset_index(drop=True)
df['voltage_V'] = (
    df.groupby(["bias_V", "event", "channel"])["voltage_V"]
    .transform(lambda x: gaussian_filter(x, sigma=2))
)
print("Daten sortiert und gefiltert")

fig, ax = plt.subplots(figsize=(10, 4))

selection = [60,100,200,300,400,500]
for volt in selection:
    df_sel = df[
    (df["bias_V"]  == volt) &
    (df["event"]   == 1) &
    (df["channel"] == "C001")
    ]
    sns.lineplot(data=df_sel, x="time_ns", y="voltage_V", ax=ax, linewidth=1.2, label=f"{volt} V")
    ax.set_xlim(40, 60)
    ax.set_title("Bias Vergleich — Rückseite")
    ax.set_xlabel("Zeit (ns)")
    ax.set_ylabel("Spannung (V)")
    ax.legend(title="Bias Spannung")
    print(f"Graph {volt} V geplottet")

ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/bias_rueckseite.png", dpi=150)

###################################
### Mittlere Waveforms C001     ###
###################################
df_raw = df[(df["channel"] == "C001") & (df["event"] <= 1000)].copy()

fig, ax = plt.subplots(figsize=(10, 4))
for volt in selection:
    grp = df_raw[df_raw["bias_V"] == volt]
    t_common = np.linspace(grp["time_ns"].min(), grp["time_ns"].max(), 1024)
    mean_v = np.zeros_like(t_common)
    n_ev = 0
    for ev, ev_df in grp.groupby("event"):
        if len(ev_df) < 5:
            continue
        mean_v += np.interp(t_common, ev_df["time_ns"].values, ev_df["voltage_V"].values)
        n_ev += 1
    if n_ev > 0:
        mean_v /= n_ev
        mean_v = gaussian_filter(mean_v, sigma=3)
        ax.plot(t_common, mean_v, linewidth=1.2, label=f"{volt} V")

ax.set_xlim(40, 60)
ax.set_xlabel("Zeit (ns)")
ax.set_ylabel("Spannung (V)")
ax.set_title("Signal Rückseite")
ax.legend(title="Bias-Spannung")
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/mean_waveforms_rueckseite.png", dpi=150)
print("Mittlere Waveforms Rückseite gespeichert")

#########################
### Charge berechnen ###
#########################
df = (
    df.groupby(["bias_V", "event", "channel"], group_keys=False)
    .apply(extract_signal_window)
    .reset_index(drop=True)
)
print(f"Signalextraktion abgeschlossen, {len(df)} Samples verbleiben")

R = 50.0  # Abschlusswiderstand in Ohm

df_100 = df[(df["event"] <= 1000) & (df["channel"] == "C001")]

charge = (
    df_100.groupby(["bias_V", "event"])
    .apply(integrate_charge)
    .reset_index(name="charge_C")
)
charge["charge_pC"] = charge["charge_C"] * 1e12

charge_mean = (
    charge.groupby("bias_V")["charge_pC"]
    .agg(mean="mean", std="std")
    .reset_index()
    .sort_values("bias_V")
)
print(charge_mean)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    charge_mean["bias_V"], charge_mean["mean"], yerr=charge_mean["std"],
    marker="o", linewidth=1.5, capsize=4, color="steelblue"
)
ax.set_xlabel("Bias-Spannung (V)")
ax.set_ylabel("Ladung (pC)")
ax.set_title("Gemittelte Ladung vs. Bias-Spannung — Rückseite")
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/charge_vs_bias_rueckseite.png", dpi=150)
print("Ladungsplot gespeichert")

##################################
### 3x3 Zufällige Signale C001 ###
##################################
df_c001 = df[df["channel"] == "C001"]
bias_options = df_c001["bias_V"].unique()

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=False, sharey=True)

for ax in axes.flat:
    bias_val = np.random.choice(bias_options)
    ev       = np.random.choice(df_c001[df_c001["bias_V"] == bias_val]["event"].unique())
    trace = df_c001[(df_c001["bias_V"] == bias_val) & (df_c001["event"] == ev)]
    ax.plot(trace["time_ns"], trace["voltage_V"], linewidth=1.0, color="steelblue")
    ax.set_title(f"Event {ev} | {bias_val} V", fontsize=8)
    ax.set_xlabel("Zeit (ns)", fontsize=7)
    ax.set_ylabel("U (V)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True)

plt.suptitle("9 zufällige Signale — Kanal C001 — Rückseite", fontsize=11)
plt.tight_layout()
plt.savefig("Auswertung/random_signals_rueckseite.png", dpi=150)
print("3x3 Signalplot gespeichert")

################################
### Signalzeit vs. Bias      ###
################################
df_c001_1000 = df[(df["channel"] == "C001") & (df["event"] <= 1000)]

signal_time = (
    df_c001_1000.groupby(["bias_V", "event"])["time_ns"]
    .agg(duration=lambda t: t.max() - t.min())
    .reset_index()
)

signal_time_mean = (
    signal_time.groupby("bias_V")["duration"]
    .agg(mean="mean", std="std")
    .reset_index()
    .sort_values("bias_V")
)
print(signal_time_mean)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    signal_time_mean["bias_V"], signal_time_mean["mean"], yerr=signal_time_mean["std"],
    marker="o", linewidth=1.5, capsize=4, color="steelblue"
)
ax.set_xlabel("Bias-Spannung (V)")
ax.set_ylabel("Signaldauer (ns)")
ax.set_title("Gemittelte Signaldauer vs. Bias-Spannung — Rückseite")
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/signalzeit_vs_bias_rueckseite.png", dpi=150)
print("Signalzeitplot gespeichert")

################################
### Mobilitäts-Fit (Gl. 5.77) ##
### Ladungsträger: Löcher      ##
### (Rückseite, x_0 ≈ d)      ##
################################
d_m   = 300e-6   # Detektordicke in m
V_dep = 14.83    # Depletionsspannung in V (Betrag)

# Nur Punkte mit |V| > V_dep verwenden (Formel nur dann definiert)
fit_df = signal_time_mean[signal_time_mean["bias_V"] > V_dep].copy()
V_abs  = fit_df["bias_V"].values.astype(float)
T_ns   = fit_df["mean"].values
T_s    = T_ns * 1e-9
T_err  = fit_df["std"].values * 1e-9

def collection_time_holes(V_abs, mu_h):
    """
    Sammelzeit für Löcher nach Gl. (5.77), x_0 = d:
    T⁺ = tau_h * ln((|V| + V_dep) / (|V| - V_dep))
    mit tau_h = d² / (2 * mu_h * V_dep)
    """
    tau = d_m**2 / (2.0 * mu_h * V_dep)
    return tau * np.log((V_abs + V_dep) / (V_abs - V_dep))

# Startwert: mu_h ≈ 0.045 m²/(V·s) = 450 cm²/(V·s)
p0 = [0.045]
popt, pcov = curve_fit(
    collection_time_holes, V_abs, T_s,
    p0=p0, sigma=T_err, absolute_sigma=True, maxfev=10000
)
mu_h_fit = popt[0]
mu_h_err = np.sqrt(pcov[0, 0])

print(f"\n=== Mobilitäts-Fit Löcher (Rückseite) ===")
print(f"μ_h = ({mu_h_fit * 1e4:.1f} ± {mu_h_err * 1e4:.1f}) cm²/(V·s)")

# Plot
V_plot = np.linspace(V_abs.min(), V_abs.max(), 500)
T_plot = collection_time_holes(V_plot, mu_h_fit) * 1e9  # in ns

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    V_abs, T_ns, yerr=fit_df["std"].values,
    fmt="o", capsize=4, color="steelblue", label="Messung", zorder=3
)
ax.plot(
    V_plot, T_plot, "-", color="tomato", linewidth=2,
    label=rf"Fit Gl.(5.77): $\mu_h$ = ({mu_h_fit * 1e4:.0f} $\pm$ {mu_h_err * 1e4:.0f}) cm²/(V·s)"
)
ax.set_xlabel("|Bias-Spannung| (V)")
ax.set_ylabel("Signaldauer (ns)")
ax.set_title("Mobilitäts-Fit — Löcher (Rückseite)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/mobilitaet_loecher_rueckseite.png", dpi=150)
print("Mobilitätsplot gespeichert")
