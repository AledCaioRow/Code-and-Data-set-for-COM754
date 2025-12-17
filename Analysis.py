import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path

# ======================
# CONFIG
# ======================
INPUT_CSV = "TheDataset.csv"
DV = "Z_composite"

OUT_DIR = Path("hypothesis_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(INPUT_CSV)

df = df[df["Complexity"].isin(["ID", "OOD"]) & df["Valence"].isin(["Positive", "Negative"])].copy()

df["Complexity"] = pd.Categorical(df["Complexity"], ["ID", "OOD"])
df["Valence"] = pd.Categorical(df["Valence"], ["Positive", "Negative"])

# ======================
# Helper: eta squared
# ======================
def eta_squared(anova_table, effect_name):
    ss_effect = anova_table.loc[effect_name, "sum_sq"]
    ss_total = anova_table["sum_sq"].sum()
    return ss_effect / ss_total

# ======================
# H1 — Complexity
# ======================
model_H1 = smf.ols(f"{DV} ~ C(Complexity)", data=df).fit()
anova_H1 = sm.stats.anova_lm(model_H1, typ=2)
eta_H1 = eta_squared(anova_H1, "C(Complexity)")

anova_H1["eta_sq"] = [eta_H1, None]
anova_H1.to_csv(OUT_DIR / "H1_complexity_anova.csv")

# ======================
# H2 — Valence
# ======================
model_H2 = smf.ols(f"{DV} ~ C(Valence)", data=df).fit()
anova_H2 = sm.stats.anova_lm(model_H2, typ=2)
eta_H2 = eta_squared(anova_H2, "C(Valence)")

anova_H2["eta_sq"] = [eta_H2, None]
anova_H2.to_csv(OUT_DIR / "H2_valence_anova.csv")

# ======================
# H3 — Interaction
# ======================
model_H3 = smf.ols(f"{DV} ~ C(Complexity) * C(Valence)", data=df).fit()
anova_H3 = sm.stats.anova_lm(model_H3, typ=2)

eta_H3_complexity = eta_squared(anova_H3, "C(Complexity)")
eta_H3_valence = eta_squared(anova_H3, "C(Valence)")
eta_H3_interaction = eta_squared(anova_H3, "C(Complexity):C(Valence)")

anova_H3["eta_sq"] = [
    eta_H3_complexity,
    eta_H3_valence,
    eta_H3_interaction,
    None
]
anova_H3.to_csv(OUT_DIR / "H3_interaction_anova.csv")

# ======================
# WRITE REPORT
# ======================
with open(OUT_DIR / "hypothesis_results_report.txt", "w", encoding="utf-8") as f:
    f.write("HYPOTHESIS TEST RESULTS (Z_composite)\n")
    f.write("=" * 45 + "\n\n")

    f.write("H1 — Emotional Complexity\n")
    f.write(anova_H1.to_string())
    f.write(f"\n\nEffect size (η²): {eta_H1:.4f}\n\n")

    f.write("H2 — Emotional Valence\n")
    f.write(anova_H2.to_string())
    f.write(f"\n\nEffect size (η²): {eta_H2:.4f}\n\n")

    f.write("H3 — Complexity × Valence Interaction\n")
    f.write(anova_H3.to_string())
    f.write(
        f"\n\nEffect sizes (η²):\n"
        f"  Complexity: {eta_H3_complexity:.4f}\n"
        f"  Valence: {eta_H3_valence:.4f}\n"
        f"  Interaction: {eta_H3_interaction:.4f}\n"
    )

print("✅ Analysis complete.")
print(f"Outputs saved to: {OUT_DIR.resolve()}")
# ======================