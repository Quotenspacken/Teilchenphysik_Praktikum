import struct
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter



def read_drs4(filepath):
    """
    Liest eine DRS4-Binaerdatei (.dat) und gibt einen pandas DataFrame zurueck.

    DataFrame-Spalten:
        event     : Event-Nummer (int, 1-basiert)
        channel   : Kanal-ID (str, z.B. 'C001', 'C004')
        sample    : Sample-Index (0-1023)
        time_ns   : Zeitpunkt des Samples in ns
        voltage_V : gemessene Spannung in V

    Parameter
    ----------
    filepath : str  Pfad zur .dat-Datei
    """
    with open(filepath, "rb") as f:
        data = f.read()

    pos = 0

    # --- Datei-Magic ---
    assert data[pos:pos+4] == b"DRS2", "Keine gueltige DRS4-Datei (Magic 'DRS2' fehlt)"
    pos += 4

    # --- TIME-Kalibrierungsblock ---
    assert data[pos:pos+4] == b"TIME", "TIME-Block nicht gefunden"
    pos += 4
    pos += 4  # Board-Seriennummer ueberspringen

    # Zeitbin-Breiten pro Kanal
    time_bins = {}
    while pos < len(data) and data[pos:pos+1] == b"C":
        ch_id = data[pos:pos+4].decode("ascii")
        pos += 4
        widths = np.array(struct.unpack_from("<1024f", data, pos))
        time_bins[ch_id] = np.concatenate([[0.0], np.cumsum(widths[:-1])])
        pos += 4096  # 1024 x 4 Bytes

    # --- Events einlesen ---
    records = []
    event_num = 0

    while pos < len(data):
        if data[pos:pos+4] != b"EHDR":
            break
        pos += 4
        pos += 4   # Event-Seriennummer
        pos += 16  # Timestamp (7x uint16) + Range (2 Bytes)
        pos += 8   # Board-Seriennummer + Trigger-Cell-Marker

        event_num += 1

        # Kanaldaten
        while pos < len(data) and data[pos:pos+1] == b"C":
            ch_id = data[pos:pos+4].decode("ascii")
            pos += 4
            pos += 4  # Scaler

            raw = np.array(struct.unpack_from("<1024H", data, pos), dtype=np.float32)
            pos += 2048  # 1024 x 2 Bytes

            voltage = (raw / 65536.0 - 0.5) * 1.0  # Vollausschlag +/-0.5 V
            t = time_bins.get(ch_id, np.arange(1024, dtype=float))

            df_chunk = pd.DataFrame({
                "event":     event_num,
                "channel":   ch_id,
                "sample":    np.arange(1024),
                "time_ns":   t,
                "voltage_V": voltage,
            })
            records.append(df_chunk)

    df = pd.concat(records, ignore_index=True)
    return df

def extract_signal_window(grp, baseline_samples=1000, threshold_factor=500):
    """
    Gibt nur die Samples zurück, die zum Signal gehören.
    Das Signal wird als jener Puls definiert, der den höchsten Peak hat.
    Start und Ende werden über einen Threshold relativ zum Baseline-Rauschen bestimmt.
    """
    v = grp["voltage_V"].values

    # Baseline aus den ersten ruhigen Samples schätzen
    baseline = np.mean(v[:baseline_samples])
    noise    = np.std(v[:baseline_samples])
    threshold = baseline + threshold_factor * noise

    # Peak: Sample mit größter Abweichung von der Baseline
    peak_idx = np.argmax(np.abs(v - baseline))

    # Signal-Polarität: positiv oder negativ
    polarity = np.sign(v[peak_idx] - baseline)

    # Vom Peak rückwärts laufen bis unter Threshold
    start = peak_idx
    while start > 0 and polarity * (v[start] - baseline) > noise:
        start -= 1

    # Vom Peak vorwärts laufen bis unter Threshold
    end = peak_idx
    while end < len(v) - 1 and polarity * (v[end] - baseline) > noise:
        end += 1

    return grp.iloc[start:end + 1]

def integrate_charge(grp):
    t = grp["time_ns"].values * 1e-9  # ns -> s
    v = grp["voltage_V"].values
    return np.trapz(v, t) / R         # Q = integral(V/R dt) in Coulomb



daten_dir = Path("daten")

#################################
### Bias Vergleich Vorderseite###
#################################
dfs = []
for f in sorted(daten_dir.glob("1_*.dat")):
    # Bias-Spannung aus Dateiname extrahieren: "1_100V.dat" -> 100
    bias_str = f.stem.split("_")[1]          # "100V"
    bias_v   = int(bias_str.replace("V", "")) # 100
    df_tmp = read_drs4(f)
    df_tmp["bias_V"] = bias_v
    dfs.append(df_tmp)
    print(f"Datei {f.name} mit Bias {bias_v} V eingelesen, {len(df_tmp)} Zeilen")
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
    
    sns.lineplot(data=df_sel, x="time_ns", y="voltage_V",   ax=ax, linewidth=1.2, label=f"{volt} V")
    ax.set_xlim(40, 60)
    ax.set_title(f"Bias Vergleich")
    ax.set_xlabel("Zeit (ns)")
    ax.set_ylabel("Spannung (V)")
    ax.legend(title="Bias Spannung")
    print(f"Graph {volt} V geplottet")

ax.grid(True)
plt.tight_layout()
plt.savefig(f"Auswertung/bias_vorderseite.png", dpi=150)

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

# Ladung pro Event berechnen, nur Kanal C001, erste 1000 Events
df_100 = df[(df["event"] <= 1000) & (df["channel"] == "C001")]

charge = (
    df_100.groupby(["bias_V", "event"])
    .apply(integrate_charge)
    .reset_index(name="charge_C")
)
charge["charge_pC"] = charge["charge_C"] * 1e12  # Coulomb -> Picocoulomb

# Über die ersten 1000 Events mitteln
charge_mean = (
    charge.groupby("bias_V")["charge_pC"]
    .agg(mean="mean", std="std")
    .reset_index()
    .sort_values("bias_V")
)
print(charge_mean)

# Ladung vs. Bias-Spannung plotten
fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    charge_mean["bias_V"], charge_mean["mean"], yerr=charge_mean["std"],
    marker="o", linewidth=1.5, capsize=4, color="steelblue"
)

ax.set_xlabel("Bias-Spannung (V)")
ax.set_ylabel("Ladung (pC)")
ax.set_title("Gemittelte Ladung vs. Bias-Spannung ")
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/charge_vs_bias.png", dpi=150)
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

plt.suptitle("9 zufällige Signale — Kanal C001", fontsize=11)
plt.tight_layout()
plt.savefig("Auswertung/random_signals.png", dpi=150)
print("3x3 Signalplot gespeichert")

###############################
### Signalzeit vs. Bias ###
###############################
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
ax.set_title("Gemittelte Signaldauer vs. Bias-Spannung ")
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/signalzeit_vs_bias.png", dpi=150)
print("Signalzeitplot gespeichert")

