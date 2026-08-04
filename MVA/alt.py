
###Alt distanz

def weighted_ks_distance(x1, x2, w1=None, w2=None):
    """
    Berechnet die maximale Distanz zwischen zwei gewichteten
    kumulativen Verteilungen.

    Kleines D  -> gute Übereinstimmung
    Großes D  -> schlechte Übereinstimmung
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if w1 is None:
        w1 = np.ones(len(x1))

    if w2 is None:
        w2 = np.ones(len(x2))

    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)

    # NaN- und unendliche Werte entfernen
    mask1 = np.isfinite(x1) & np.isfinite(w1)
    mask2 = np.isfinite(x2) & np.isfinite(w2)

    x1 = x1[mask1]
    w1 = w1[mask1]

    x2 = x2[mask2]
    w2 = w2[mask2]

    # Ungültige Spalten abfangen
    if len(x1) == 0 or len(x2) == 0:
        return np.nan

    if np.sum(w1) == 0 or np.sum(w2) == 0:
        return np.nan

    # Werte sortieren
    order1 = np.argsort(x1)
    order2 = np.argsort(x2)

    x1 = x1[order1]
    w1 = w1[order1]

    x2 = x2[order2]
    w2 = w2[order2]

    # Gewichtete kumulative Verteilungen
    cdf1 = np.cumsum(w1) / np.sum(w1)
    cdf2 = np.cumsum(w2) / np.sum(w2)

    # Gemeinsame x-Werte
    x_all = np.sort(
        np.unique(
            np.concatenate([x1, x2])
        )
    )

    # Position in den beiden sortierten Arrays
    idx1 = np.searchsorted(x1, x_all, side="right") - 1
    idx2 = np.searchsorted(x2, x_all, side="right") - 1

    # Kumulative Werte an den gemeinsamen x-Stellen
    F1 = np.where(
        idx1 >= 0,
        cdf1[np.maximum(idx1, 0)],
        0.0
    )

    F2 = np.where(
        idx2 >= 0,
        cdf2[np.maximum(idx2, 0)],
        0.0
    )

    # Größte Distanz
    return np.max(np.abs(F1 - F2))


# Nur gemeinsame numerische Spalten auswählen
common_cols = (
    data.select_dtypes(include=np.number).columns
    .intersection(
        sim_ctrl.select_dtypes(include=np.number).columns
    )
)

# Gewichtsspalten selbst nicht testen
exclude = {
    "kinematic_weights",
    "sweights_sig"
}

variables = [
    var for var in common_cols
    if var not in exclude
]

results = []
bad_vars = []

n = len(variables)

for i, var in enumerate(variables, start=1):

    D = weighted_ks_distance(
        sim_ctrl[var],
        data[var],
        sim_ctrl["kinematic_weights"],
        data["sweights_sig"]
    )

    # Ungültige Ergebnisse überspringen
    if np.isnan(D):
        print(f"[{i}/{n}] {var}: übersprungen")
        continue

    results.append([var, D])

    if D > 0.01:
        bad_vars.append(var)
        print(f"[{i}/{n}] {var}: D = {D:.4f} -> entfernen")
    else:
        print(f"[{i}/{n}] {var}: D = {D:.4f}")


# Schlechte Variablen entfernen
data_clear = data.drop(
    columns=bad_vars,
    errors="ignore"
)

sim_ctrl_clear = sim_ctrl.drop(
    columns=bad_vars,
    errors="ignore"
)


# Ergebnisse sortieren
results_df = pd.DataFrame(
    results,
    columns=["Variable", "D"]
)

results_df = results_df.sort_values(
    "D",
    ascending=True
).reset_index(drop=True)


print("\nBeste 20 Variablen:")
print(results_df.head(20))

print(f"\nAnzahl entfernter Variablen: {len(bad_vars)}")
print(f"Anzahl verbleibender Variablen: {len(data_clear.columns)}")








############### alt korrelation
######Alt




# Ex. 3.3: Korrelation der Features mit der invarianten Bs-Masse

mass_col = "B_FitDaughtersConst_M_flat"
correlation_limit = 0.01   # Grenze nach Bedarf anpassen

# Bisher akzeptierte Variablen aus Ex. 3.1
features = results_df.loc[
    results_df["D"] <= 0.01,
    "Variable"
].tolist()

# Nur Variablen verwenden, die auch in der Bs-Simulation existieren
features = [
    var for var in features
    if var in sim.columns
    and var != mass_col
    and pd.api.types.is_numeric_dtype(sim[var])
]

mass_correlations = []

for var in features:

    # Nur Zeilen verwenden, in denen Masse und Feature gültig sind
    valid = sim[[var, mass_col]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Konstante Variablen besitzen keine sinnvolle Korrelation
    if valid[var].nunique() <= 1:
        continue

    correlation = valid[var].corr(
        valid[mass_col],
        method="pearson"
    )

    mass_correlations.append([
        var,
        correlation,
        abs(correlation)
    ])

# Übersichtstabelle
mass_corr_df = pd.DataFrame(
    mass_correlations,
    columns=["Variable", "Correlation", "AbsCorrelation"]
)

# Stärkste Korrelationen zuerst anzeigen
mass_corr_df = mass_corr_df.sort_values(
    "AbsCorrelation",
    ascending=False
).reset_index(drop=True)

print("Stärkste Korrelationen mit der Bs-Masse:")
print(mass_corr_df.head(20))



mass_correlated_vars = mass_corr_df.loc[
    mass_corr_df["AbsCorrelation"] > correlation_limit,
    "Variable"
].tolist()

final_features = [
    var for var in features
    if var not in mass_correlated_vars
]

print("\nEntfernte massenkorrelierte Variablen:")
print(mass_correlated_vars)

print(f"\nAnzahl Features vorher: {len(features)}")
print(f"Anzahl Features danach: {len(final_features)}")

# Schlechte Variablen entfernen
data_clear_2 = data_clear.drop(
    columns=mass_correlated_vars,
    errors="ignore"
)

sim_ctrl_clear_2 = sim_ctrl_clear.drop(
    columns=mass_correlated_vars,
    errors="ignore"
)



####### Altes k-Values
def train_kfold_models(X, y, weights, n_splits=5, **xgb_params):
    kf = StratifiedKFold(...)     # ← Splitter für 5 Folds
    bdts = []                      # ← Liste für 5 Modelle
    results = []                   # ← Liste für 5 Ergebnisse
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        # Jede Iteration ist ein FOLD
        
        # 1. EXTRAHIEREN
        X_train = X.iloc[train_idx]  # ← Die Training-Samples für DIESEN Fold
        y_train = y[train_idx]
        w_train = weights[train_idx]
        
        # 2. TRAINIEREN
        bdt = XGBClassifier(**xgb_params)  # ← Neues Modell!
        bdt.fit(X_train, y_train, sample_weight=w_train)
        
        # 3. EVALUIEREN
        y_pred = bdt.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], y_pred, sample_weight=weights[test_idx])
        
        # 4. SPEICHERN
        bdts.append(bdt)            # ← Modell speichern
        results.append({
            'fold': fold,
            'auc': auc,
            'pred': y_pred
        })
    
    return bdts, results            # ← 5 Modelle + 5 Ergebnisse





### 4.4
all_outputs = np.concatenate([
    result["output"] for result in fold_results
])

all_labels = np.concatenate([
    result["labels"] for result in fold_results
])

all_weights = np.concatenate([
    result["weights"] for result in fold_results
])

signal_mask = all_labels == 1
background_mask = all_labels == 0
bins = np.linspace(0, 1, 51)

plt.hist(
    all_outputs[signal_mask],
    bins=bins,
    weights=all_weights[signal_mask],
    histtype="step",
    density=True,
    label="Signal MC"
)

plt.hist(
    all_outputs[background_mask],
    bins=bins,
    weights=all_weights[background_mask],
    histtype="step",
    density=True,
    label="Background"
)

plt.xlabel("BDT output")
plt.ylabel("Normalized events")
plt.yscale("log")
plt.legend()
plt.show()


####1
import zfit
zfit.core.parameter.ZfitParameterMixin._existing_names.clear()



# --- BDT auf das Signal-MC Sample anwenden ---
sim_features = sim[vvg_features]
sim_predictions = np.mean([
    bdt.predict_proba(sim_features)[:, 1] for bdt in bdts
], axis=0)

sim_selected = sim[sim_predictions > best_cut]
mass_data = sim_selected['B_FitDaughtersConst_M_flat'].values

# Check auf NaN/Inf in den Daten (häufige Fehlerquelle!)
print("NaNs in mass_data:", np.isnan(mass_data).sum())
print("Infs in mass_data:", np.isinf(mass_data).sum())
mass_data = mass_data[np.isfinite(mass_data)]

# --- Datengetriebene Startwerte ---
mean_guess = np.mean(mass_data)
std_guess = np.std(mass_data)
print(f"Mean guess: {mean_guess:.1f}, Std guess: {std_guess:.1f}")

# --- zfit Setup ---
obs = zfit.Space('B_FitDaughtersConst_M_flat', limits=(mass_data.min(), mass_data.max()))
data_sim = zfit.Data.from_numpy(obs=obs, array=mass_data)

# --- Parameter mit sinnvollen, datengetriebenen Startwerten ---
mu = zfit.Parameter('mu_sim', mean_guess, mean_guess - 50, mean_guess + 50)

# Sigma-Bounds NICHT bei 0 beginnen lassen (Instabilität!)
sigma1 = zfit.Parameter('sigma1_sim', std_guess * 0.7, 2, std_guess * 3)
sigma2 = zfit.Parameter('sigma2_sim', std_guess * 1.5, 2, std_guess * 5)

gauss1 = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs)
gauss2 = zfit.pdf.Gauss(mu=mu, sigma=sigma2, obs=obs)

# frac NICHT direkt an der Grenze starten (nicht 0 oder 1!)
frac = zfit.Parameter('frac_sim', 0.5, 0.01, 0.99)

signal_pdf = zfit.pdf.SumPDF([gauss1, gauss2], fracs=frac)

# --- Fit ---
nll = zfit.loss.UnbinnedNLL(model=signal_pdf, data=data_sim)
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)

print(result)
print(result.params)


mu.floating = False
sigma1.floating = False
sigma2.floating = False
frac.floating = False


plt.figure(figsize=(8, 6))
plt.hist(mass_data, bins=100, range=(mass_data.min(), mass_data.max()), density=True,
         histtype='step', color='black', label='Simulation')

x_plot = np.linspace(mass_data.min(), mass_data.max(), 1000)
y_plot = zfit.run(signal_pdf.pdf(x_plot))
plt.plot(x_plot, y_plot, color='red', label='Double Gaussian Fit')

plt.xlabel('Invariant Mass [MeV/$c^2$]')
plt.ylabel('Normalized Events')
plt.legend()
plt.show()



#2
# --- BDT auf das B0-Signal-MC Sample anwenden ---
sim_B0_features = sim_ctrl[vvg_features]
sim_B0_predictions = np.mean([
    bdt.predict_proba(sim_B0_features)[:, 1] for bdt in bdts
], axis=0)

sim_B0_selected = sim_ctrl[sim_B0_predictions > best_cut]
mass_data_B0 = sim_B0_selected['B_FitDaughtersConst_M_flat'].values

# Check auf NaN/Inf
print("NaNs in mass_data_B0:", np.isnan(mass_data_B0).sum())
print("Infs in mass_data_B0:", np.isinf(mass_data_B0).sum())
mass_data_B0 = mass_data_B0[np.isfinite(mass_data_B0)]

mean_guess_B0 = np.mean(mass_data_B0)
std_guess_B0 = np.std(mass_data_B0)
print(f"Mean guess: {mean_guess_B0:.1f}, Std guess: {std_guess_B0:.1f}")

# --- zfit Setup ---
obs_B0 = zfit.Space('B_M', limits=(mass_data_B0.min(), mass_data_B0.max()))
data_sim_B0 = zfit.Data.from_numpy(obs=obs_B0, array=mass_data_B0)

# --- Parameter (eindeutige Namen, damit kein NameAlreadyTakenError) ---
mu_B0 = zfit.Parameter('mu_B0', mean_guess_B0, mean_guess_B0 - 50, mean_guess_B0 + 50)
sigma1_B0 = zfit.Parameter('sigma1_B0', std_guess_B0 * 0.7, 2, std_guess_B0 * 3)
sigma2_B0 = zfit.Parameter('sigma2_B0', std_guess_B0 * 1.5, 2, std_guess_B0 * 5)

gauss1_B0 = zfit.pdf.Gauss(mu=mu_B0, sigma=sigma1_B0, obs=obs_B0)
gauss2_B0 = zfit.pdf.Gauss(mu=mu_B0, sigma=sigma2_B0, obs=obs_B0)

frac_B0 = zfit.Parameter('frac_B0', 0.5, 0.01, 0.99)

signal_pdf_B0 = zfit.pdf.SumPDF([gauss1_B0, gauss2_B0], fracs=frac_B0)

# --- Fit ---
nll_B0 = zfit.loss.UnbinnedNLL(model=signal_pdf_B0, data=data_sim_B0)
minimizer_B0 = zfit.minimize.Minuit()
result_B0 = minimizer_B0.minimize(nll_B0)

print(result_B0)
print(result_B0.params)

# --- Parameter fixieren ---
mu_B0.floating = False
sigma1_B0.floating = False
sigma2_B0.floating = False
frac_B0.floating = False

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.hist(mass_data_B0, bins=100, range=(mass_data_B0.min(), mass_data_B0.max()), density=True,
         histtype='step', color='black', label='Simulation ($B^0$)')

x_plot_B0 = np.linspace(mass_data_B0.min(), mass_data_B0.max(), 1000)
y_plot_B0 = zfit.run(signal_pdf_B0.pdf(x_plot_B0))
plt.plot(x_plot_B0, y_plot_B0, color='blue', label='Double Gaussian Fit ($B^0$)')

plt.xlabel('Invariant Mass [MeV/$c^2$]')
plt.ylabel('Normalized Events')
plt.legend()
plt.show()

#3
# --- BDT-Cut auf die echten Daten anwenden 
data_features = data[vvg_features]
data_predictions = np.mean([
    bdt.predict_proba(data_features)[:, 1] for bdt in bdts
], axis=0)

data_selected = data[data_predictions > best_cut]
mass_data_full = data_selected['B_FitDaughtersConst_M_flat'].values
mass_data_full = mass_data_full[np.isfinite(mass_data_full)]

n_total = len(mass_data_full)
print(f"Anzahl selektierter Daten-Events: {n_total}")

# --- zfit Setup für den vollen Massenbereich ---
obs_full = zfit.Space('B_FitDaughtersConst_M_flat', limits=(mass_data_full.min(), mass_data_full.max()))
data_full = zfit.Data.from_numpy(obs=obs_full, array=mass_data_full)

# --- Bs Signal PDF mit fixierten Shape-Parametern (aus Ex 6.2) ---
gauss1_data = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs_full)
gauss2_data = zfit.pdf.Gauss(mu=mu, sigma=sigma2, obs=obs_full)
signal_shape_Bs = zfit.pdf.SumPDF([gauss1_data, gauss2_data], fracs=frac)

# --- Skalierungsfaktor + Yield für Bs ---
scale_Bs = zfit.Parameter('scale_Bs', 0.3, 0, 1)
n_Bs = zfit.ComposedParameter('n_Bs', scale_Bs * n_total)
signal_pdf_Bs_ext = signal_shape_Bs.create_extended(n_Bs)

# --- B0 Signal PDF mit fixierten Shape-Parametern (aus Ex 6.3) ---
gauss1_B0_data = zfit.pdf.Gauss(mu=mu_B0, sigma=sigma1_B0, obs=obs_full)
gauss2_B0_data = zfit.pdf.Gauss(mu=mu_B0, sigma=sigma2_B0, obs=obs_full)
signal_shape_B0 = zfit.pdf.SumPDF([gauss1_B0_data, gauss2_B0_data], fracs=frac_B0)

# --- Skalierungsfaktor + Yield für B0 ---
scale_B0 = zfit.Parameter('scale_B0', 0.1, 0, 1)
n_B0 = zfit.ComposedParameter('n_B0', scale_B0 * n_total)
signal_pdf_B0_ext = signal_shape_B0.create_extended(n_B0)

# --- Exponentieller Background ---
lambda_bkg = zfit.Parameter('lambda_bkg', -0.001, -0.1, -0.00001)
background_pdf = zfit.pdf.Exponential(lambda_bkg, obs=obs_full)

# --- Skalierungsfaktor + Yield für Background ---
scale_bkg = zfit.Parameter('scale_bkg', 0.6, 0, 1)
n_bkg = zfit.ComposedParameter('n_bkg', scale_bkg * n_total)
background_pdf_ext = background_pdf.create_extended(n_bkg)

# --- Gesamtmodell: Summe aller extended PDFs ---
full_model = zfit.pdf.SumPDF([signal_pdf_Bs_ext, signal_pdf_B0_ext, background_pdf_ext])

# --- Fit (Extended NLL, da wir Yields fitten) ---
nll_full = zfit.loss.ExtendedUnbinnedNLL(model=full_model, data=data_full)
minimizer_full = zfit.minimize.Minuit()
result_full = minimizer_full.minimize(nll_full)

print(result_full)
print(result_full.params)

# --- Plot ---
plt.figure(figsize=(8, 6))
n_bins = 100
counts, bin_edges, _ = plt.hist(mass_data_full, bins=n_bins,
                                  range=(mass_data_full.min(), mass_data_full.max()),
                                  histtype='step', color='black', label='Data')

x_plot_full = np.linspace(mass_data_full.min(), mass_data_full.max(), 1000)
bin_width = (bin_edges[1] - bin_edges[0])

y_total = zfit.run(full_model.pdf(x_plot_full)) * bin_width * n_total
plt.plot(x_plot_full, y_total, color='red', label='Full Fit')

y_Bs = zfit.run(signal_pdf_Bs_ext.pdf(x_plot_full)) * bin_width * zfit.run(n_Bs)
y_B0 = zfit.run(signal_pdf_B0_ext.pdf(x_plot_full)) * bin_width * zfit.run(n_B0)
y_bkg = zfit.run(background_pdf_ext.pdf(x_plot_full)) * bin_width * zfit.run(n_bkg)

plt.plot(x_plot_full, y_Bs, '--', color='blue', label='$B_s$ Signal')
plt.plot(x_plot_full, y_B0, '--', color='green', label='$B^0$ Signal')
plt.plot(x_plot_full, y_bkg, '--', color='orange', label='Combinatorial Background')

plt.xlabel('Invariant Mass [MeV/$c^2$]')
plt.ylabel('Events')
#plt.yscale('log')
plt.legend()
plt.show()

print(f"\nFitted scale factors:")
print(f"  scale_Bs  = {zfit.run(scale_Bs):.4f}  →  N(Bs)  = {zfit.run(n_Bs):.1f}")
print(f"  scale_B0  = {zfit.run(scale_B0):.4f}  →  N(B0)  = {zfit.run(n_B0):.1f}")
print(f"  scale_bkg = {zfit.run(scale_bkg):.4f}  →  N(bkg) = {zfit.run(n_bkg):.1f}")


#4
plt.figure(figsize=(10, 7))

# --- Daten als Histogramm ---
n_bins = 100
counts, bin_edges, _ = plt.hist(
    mass_data_full, bins=n_bins,
    range=(mass_data_full.min(), mass_data_full.max()),
    histtype='step', color='black', label='Data'
)
bin_width = bin_edges[1] - bin_edges[0]

# --- x-Werte für glatte Kurven ---
x_plot_full = np.linspace(mass_data_full.min(), mass_data_full.max(), 1000)

# --- Gesamtmodell (Summe aller 3 Komponenten) ---
y_total = zfit.run(full_model.pdf(x_plot_full)) * bin_width * n_total
plt.plot(x_plot_full, y_total, color='red', linewidth=2, label='Full Model')

# --- Einzelkomponenten ---
y_Bs = zfit.run(signal_pdf_Bs_ext.pdf(x_plot_full)) * bin_width * zfit.run(n_Bs)
y_B0 = zfit.run(signal_pdf_B0_ext.pdf(x_plot_full)) * bin_width * zfit.run(n_B0)
y_bkg = zfit.run(background_pdf_ext.pdf(x_plot_full)) * bin_width * zfit.run(n_bkg)

plt.plot(x_plot_full, y_Bs, '--', color='blue', label='$B_s$ Signal')
#plt.plot(x_plot_full, y_B0, '--', color='green', label='$B^0$ Signal')
plt.plot(x_plot_full, y_bkg, '--', color='orange', label='Combinatorial Background')

plt.xlabel('Invariant Mass [MeV/$c^2$]')
plt.ylabel('Events')
plt.yscale('log')
plt.legend()
plt.title('Full Fit: Data vs. Model')
plt.show()

# --- Yields zur Kontrolle ---
print(f"N(Bs)  = {zfit.run(n_Bs):.1f}")
print(f"N(B0)  = {zfit.run(n_B0):.1f}")
print(f"N(bkg) = {zfit.run(n_bkg):.1f}")
print(f"Summe  = {zfit.run(n_Bs) + zfit.run(n_B0) + zfit.run(n_bkg):.1f}  (sollte ≈ {n_total} sein)")


#4
# --- Signal-Region definieren: mu_Bs ± 3 sigma (effektive Breite) ---
mu_val = zfit.run(mu)

# Effektive Breite als gewichteter Mittelwert der beiden Gaussians (grobe Schätzung)
sigma1_val = zfit.run(sigma1)
sigma2_val = zfit.run(sigma2)
frac_val = zfit.run(frac)
sigma_eff = frac_val * sigma1_val + (1 - frac_val) * sigma2_val

n_sigma_window = 3
signal_region_low = mu_val - n_sigma_window * sigma_eff
signal_region_high = mu_val + n_sigma_window * sigma_eff

print(f"Signal-Region: [{signal_region_low:.1f}, {signal_region_high:.1f}] MeV/c^2")
print(f"(mu = {mu_val:.1f}, sigma_eff = {sigma_eff:.1f})")

# --- Signal-Space für die Integration definieren ---
signal_region_obs = zfit.Space('B_FitDaughtersConst_M_flat', limits=(signal_region_low, signal_region_high))

# --- Integral der PDF-Formen über die Signal-Region (liefert Anteil zwischen 0 und 1) ---
frac_Bs_in_region = zfit.run(signal_shape_Bs.integrate(limits=signal_region_obs, norm_range=obs_full))
frac_B0_in_region = zfit.run(signal_shape_B0.integrate(limits=signal_region_obs, norm_range=obs_full))
frac_bkg_in_region = zfit.run(background_pdf.integrate(limits=signal_region_obs, norm_range=obs_full))

# --- Anzahl Events in der Signal-Region = Integral-Anteil * Gesamt-Yield ---
n_sig = frac_Bs_in_region * zfit.run(n_Bs)
n_B0_in_region = frac_B0_in_region * zfit.run(n_B0)
n_bkg_in_region = frac_bkg_in_region * zfit.run(n_bkg)

# Background in der Signal-Region: hier zählt kombinatorischer Background 
# UND ggf. der B0-Anteil, der in die Bs-Region "hineinragt" (Cross-feed)
n_bkg = n_bkg_in_region + n_B0_in_region

print(f"\nn_sig (Bs in Signal-Region)  = {n_sig:.1f}")
print(f"n_bkg (combinatorial)         = {n_bkg_in_region:.1f}")
print(f"n_bkg (B0 cross-feed)         = {n_B0_in_region:.1f}")
print(f"n_bkg (total)                 = {n_bkg:.1f}")

# --- Signifikanz-Proxy ---
m_significance = n_sig / np.sqrt(n_sig + n_bkg)

print(f"\nSignificance proxy m = {m_significance:.2f}")