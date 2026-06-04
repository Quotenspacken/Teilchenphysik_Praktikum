import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter


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

    return pd.concat(records, ignore_index=True)


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


R = 50.0  # Abschlusswiderstand in Ohm

def integrate_charge(grp):
    t = grp["time_ns"].values * 1e-9
    v = grp["voltage_V"].values
    return np.trapz(v, t) / R


daten_dir = Path("daten")

##########################################
### Laser-Intensität Vergleich Rückseite##
##########################################
dfs = []
for f in sorted(daten_dir.glob("4_*.dat")):
    # Intensitätsreduktion aus Dateiname extrahieren: "4_50p.dat" -> 50
    intensity_str = f.stem.split("_")[1]
    reduction_pct = int(intensity_str.replace("p", ""))
    df_tmp = read_drs4(f)
    df_tmp["reduction_pct"] = reduction_pct
    dfs.append(df_tmp)
    print(f"Datei {f.name} mit {reduction_pct}% Reduktion eingelesen, {len(df_tmp)} Zeilen")

if not dfs:
    raise FileNotFoundError("Keine passenden 4_*.dat-Dateien in 'daten/' gefunden")

df = pd.concat(dfs, ignore_index=True)

# Daten sortieren und glätten
df = df.sort_values(["reduction_pct", "event", "channel", "sample"]).reset_index(drop=True)
df['voltage_V'] = (
    df.groupby(["reduction_pct", "event", "channel"])["voltage_V"]
    .transform(lambda x: gaussian_filter(x, sigma=2))
)
print("Daten sortiert und gefiltert")

# Übersichtsplot: verschiedene Intensitäten, Event 1, Kanal C001
fig, ax = plt.subplots(figsize=(10, 4))

selection = sorted(df["reduction_pct"].unique())
for red in selection:
    df_sel = df[
        (df["reduction_pct"] == red) &
        (df["event"]         == 1) &
        (df["channel"]       == "C001")
    ]
    sns.lineplot(data=df_sel, x="time_ns", y="voltage_V", ax=ax, linewidth=1.2, label=f"{red}%")

ax.set_xlim(40, 60)
ax.set_xlabel("Zeit (ns)")
ax.set_ylabel("Spannung (V)")
ax.set_title("Laser-Intensitätsvergleich — Rückseite")
ax.legend(title="Reduktion")
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/laser_intensitaet_vergleich_rueckseite.png", dpi=150)
print("Übersichtsplot gespeichert")

#########################
### Charge berechnen  ###
#########################
df = (
    df.groupby(["reduction_pct", "event", "channel"], group_keys=False)
    .apply(extract_signal_window)
    .reset_index(drop=True)
)
print(f"Signalextraktion abgeschlossen, {len(df)} Samples verbleiben")

df_c001 = df[(df["channel"] == "C001") & (df["event"] <= 1000)]

charge = (
    df_c001.groupby(["reduction_pct", "event"])
    .apply(integrate_charge)
    .reset_index(name="charge_C")
)
charge["charge_pC"] = charge["charge_C"] * 1e12

charge_mean = (
    charge.groupby("reduction_pct")["charge_pC"]
    .agg(mean="mean", std="std")
    .reset_index()
    .sort_values("reduction_pct")
)
print(charge_mean)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    charge_mean["reduction_pct"], charge_mean["mean"], yerr=charge_mean["std"],
    marker="o", linewidth=1.5, capsize=4, color="steelblue"
)
ax.set_xlabel("Intensitätsreduktion (%)")
ax.set_ylabel("Ladung (pC)")
ax.set_title("Gemittelte Ladung vs. Laser-Intensität — Rückseite")
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/charge_vs_laser_rueckseite.png", dpi=150)
print("Ladungsplot gespeichert")

##################################
### 3x3 Zufällige Signale C001 ###
##################################
intensity_options = df[df["channel"] == "C001"]["reduction_pct"].unique()

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=False, sharey=True)

for ax in axes.flat:
    red_val = np.random.choice(intensity_options)
    ev      = np.random.choice(df[(df["channel"] == "C001") & (df["reduction_pct"] == red_val)]["event"].unique())
    trace   = df[(df["channel"] == "C001") & (df["reduction_pct"] == red_val) & (df["event"] == ev)]
    ax.plot(trace["time_ns"], trace["voltage_V"], linewidth=1.0, color="steelblue")
    ax.set_title(f"Event {ev} | {red_val}% Reduktion", fontsize=8)
    ax.set_xlabel("Zeit (ns)", fontsize=7)
    ax.set_ylabel("U (V)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True)

plt.suptitle("9 zufällige Signale — Rückseite", fontsize=11)
plt.tight_layout()
plt.savefig("Auswertung/laser_random_signals_rueckseite.png", dpi=150)
print("3x3 Signalplot gespeichert")

####################################
### Signalzeit vs. Intensität    ###
####################################
df_c001_1000 = df[(df["channel"] == "C001") & (df["event"] <= 1000)]

signal_time = (
    df_c001_1000.groupby(["reduction_pct", "event"])["time_ns"]
    .agg(duration=lambda t: t.max() - t.min())
    .reset_index()
)

signal_time_mean = (
    signal_time.groupby("reduction_pct")["duration"]
    .agg(mean="mean", std="std")
    .reset_index()
    .sort_values("reduction_pct")
)
print(signal_time_mean)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    signal_time_mean["reduction_pct"], signal_time_mean["mean"], yerr=signal_time_mean["std"],
    marker="o", linewidth=1.5, capsize=4, color="steelblue"
)
ax.set_xlabel("Intensitätsreduktion (%)")
ax.set_ylabel("Signaldauer (ns)")
ax.set_title("Gemittelte Signaldauer vs. Laser-Intensität — Rückseite")
ax.grid(True)
plt.tight_layout()
plt.savefig("Auswertung/signalzeit_vs_laser_rueckseite.png", dpi=150)
print("Signalzeitplot gespeichert")
