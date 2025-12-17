import pandas as pd
from pathlib import Path

# ======================
# CONFIG
# ======================
INPUT_CSV = "TheDataset.csv"
OUT_DIR = Path("descriptives_balanced_conversations")
OUT_DIR.mkdir(exist_ok=True)

DV = "Z_composite"

# ======================
# LOAD
# ======================
df = pd.read_csv(INPUT_CSV)

required = {"conv_id", "emotion", DV}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ======================
# BASIC INFO
# ======================
N_rows = len(df)
N_convs = df["conv_id"].nunique()

# ======================
# CONVERSATION-LEVEL TABLE
# ======================
conv_tbl = (
    df.groupby("conv_id")[["Complexity", "Valence"]]
    .first()
    .reset_index()
)

conv_counts = (
    conv_tbl
    .groupby(["Complexity", "Valence"])
    .size()
    .reset_index(name="n_conversations")
)

total_convs = conv_counts["n_conversations"].sum()
conv_counts["pct_conversations"] = (
    conv_counts["n_conversations"] / total_convs * 100
)

conv_counts.to_csv(OUT_DIR / "conversation_counts_2x2.csv", index=False)

# ======================
# UTTERANCE-LEVEL COUNTS
# ======================
utt_counts = (
    df.groupby(["Complexity", "Valence"])
    .size()
    .reset_index(name="n_utterances")
)

utt_counts["pct_utterances"] = (
    utt_counts["n_utterances"] / N_rows * 100
)

utt_counts.to_csv(OUT_DIR / "utterance_counts_2x2.csv", index=False)

# ======================
# DV DESCRIPTIVES (UTTERANCE LEVEL)
# ======================
dv_stats = (
    df.groupby(["Complexity", "Valence"])[DV]
    .agg(
        n_utterances="count",
        mean="mean",
        sd="std",
        median="median"
    )
    .reset_index()
)

dv_stats.to_csv(OUT_DIR / "dv_descriptives_2x2.csv", index=False)

# ======================
# WRITE CLEAN TEXT SUMMARY
# ======================
report_path = OUT_DIR / "descriptives_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("DESCRIPTIVE STATISTICS (Conversation-balanced dataset)\n")
    f.write("="*55 + "\n\n")

    f.write(f"Total conversations: {N_convs}\n")
    f.write(f"Total utterances: {N_rows}\n\n")

    f.write("Conversation-level distribution (PRIMARY):\n")
    f.write(conv_counts.to_string(index=False))
    f.write("\n\n")

    f.write("Utterance-level distribution (SECONDARY):\n")
    f.write(utt_counts.to_string(index=False))
    f.write("\n\n")

    f.write("Z_composite descriptives by cell (utterance-level):\n")
    f.write(dv_stats.to_string(index=False))

print("\n✅ Descriptives complete (conversation-balanced).")
print(f"Saved to: {OUT_DIR.resolve()}")
print(f"- conversation_counts_2x2.csv")
print(f"- utterance_counts_2x2.csv")
print(f"- dv_descriptives_2x2.csv")
print(f"- descriptives_report.txt")
