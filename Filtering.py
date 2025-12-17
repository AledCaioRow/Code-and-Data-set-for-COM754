import pandas as pd
from pathlib import Path

INPUT_CSV = "TheDataset.csv"
OUT_CSV   = "dataset_balanced_by_conversation.csv"
SEED = 42

# ---- emotion sets (lowercase) ----
ID_POS = {"surprised","excited","grateful","hopeful","joyful"}
ID_NEG = {"angry","annoyed","sad","afraid","lonely"}
OOD_POS = {"proud","nostalgic","sentimental"}
OOD_NEG = {"guilty","disgusted","furious","jealous","devastated","embarrassed","ashamed"}

def complexity(e):
    e = str(e).strip().lower()
    if e in (ID_POS | ID_NEG): return "ID"
    if e in (OOD_POS | OOD_NEG): return "OOD"
    return "Other"

def valence(e):
    e = str(e).strip().lower()
    if e in (ID_POS | OOD_POS): return "Positive"
    if e in (ID_NEG | OOD_NEG): return "Negative"
    return "Other"

df = pd.read_csv(INPUT_CSV)

# required cols
for c in ["conv_id","emotion"]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
df["Complexity"] = df["emotion"].apply(complexity)
df["Valence"] = df["emotion"].apply(valence)

# keep only 2x2 eligible rows
df_2x2 = df[(df["Complexity"].isin(["ID","OOD"])) & (df["Valence"].isin(["Positive","Negative"]))].copy()

# conversation-level table (one row per conv_id)
conv_tbl = (
    df_2x2.groupby("conv_id")[["Complexity","Valence"]]
    .first()
    .reset_index()
)

# count conversations per cell
cell_counts = conv_tbl.groupby(["Complexity","Valence"]).size()
print("\nConversation counts per cell:")
print(cell_counts)

min_conv = int(cell_counts.min())
print(f"\nBalancing to {min_conv} conversations per cell")

# sample conversations per cell
sampled_conv_ids = (
    conv_tbl.groupby(["Complexity","Valence"], group_keys=False)
    .apply(lambda x: x.sample(n=min_conv, random_state=SEED))["conv_id"]
    .tolist()
)

# keep ALL utterances for sampled conversations
df_bal = df_2x2[df_2x2["conv_id"].isin(sampled_conv_ids)].copy()

# sanity checks
print("\nBalanced conversation counts per cell:")
print(df_bal.groupby(["Complexity","Valence"])["conv_id"].nunique())

print("\nRow counts per cell (will not be equal; convo lengths vary):")
print(df_bal.groupby(["Complexity","Valence"]).size())

df_bal.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved: {OUT_CSV}")
