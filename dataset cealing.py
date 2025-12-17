# dataset_cleaning.py
# Minimal cleaning: remove real NLP noise + replace EmpatheticDialogues placeholders.
# Auto-detects the input CSV in the same folder (prefers filenames containing "unclean").

import os
import re
import unicodedata
from collections import Counter, defaultdict
import pandas as pd

# =========================
# OUTPUT FILES (same folder)
# =========================
OUTPUT_FILE = "Clean_TheDataset.csv"
REPORT_FILE = "Clean_TheDataset_report.txt"

# =========================
# YOUR REAL TEXT COLUMNS
# =========================
TARGET_COLS = ["speaker_last_message", "gold_listener", "gpt_listener"]
# If you also want to clean the ED "situation" text, add it:
# TARGET_COLS = ["situation", "speaker_last_message", "gold_listener", "gpt_listener"]

# =========================
# NOISE DEFINITIONS
# =========================
ZERO_WIDTH_CHARS = [
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\ufeff",  # BYTE ORDER MARK (BOM)
]

CONTROL_REGEX = re.compile(r"[\x00-\x1f\x7f-\x9f]")

ED_PLACEHOLDERS = {
    "_comma_": ",",
    "_period_": ".",
    "_exclamation_": "!",
    "_question_": "?",
    "_ellipsis_": "..."
}


def char_info(ch: str) -> str:
    return f"{repr(ch)} U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN')} cat={unicodedata.category(ch)}"


def find_input_csv():
    files = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError("No .csv files found in this folder.")

    unclean = [f for f in files if "unclean" in f.lower()]
    if unclean:
        return sorted(unclean, key=len, reverse=True)[0]

    if "TheDataset.csv" in files:
        return "TheDataset.csv"

    return files[0]


def clean_real_noise(text, stats=None, colname=""):
    if not isinstance(text, str):
        return text

    original = text

    # Canonical unicode (safe)
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible zero-width chars (pure noise)
    for zw in ZERO_WIDTH_CHARS:
        if zw in text:
            n = text.count(zw)
            if stats is not None:
                stats["zero_width_removed"][zw] += n
                stats["rows_with_zero_width"][colname] += 1
            text = text.replace(zw, "")

    # Remove control chars (pure noise)
    if CONTROL_REGEX.search(text):
        removed = CONTROL_REGEX.findall(text)
        if stats is not None:
            stats["control_removed_total"] += len(removed)
            stats["rows_with_control"][colname] += 1
        text = CONTROL_REGEX.sub("", text)

    # Replace ED placeholders (_comma_ etc.)
    for tok, repl in ED_PLACEHOLDERS.items():
        if tok in text:
            n = text.count(tok)
            if stats is not None:
                stats["placeholders_replaced"][tok] += n
                stats["rows_with_placeholders"][colname] += 1
            text = text.replace(tok, repl)

    # Collapse whitespace introduced by removals/replacements
    text = re.sub(r"\s+", " ", text).strip()

    # Keep a few before/after examples
    if stats is not None and text != original and len(stats["examples"][colname]) < 5:
        stats["examples"][colname].append((original[:220], text[:220]))

    return text


def main():
    input_file = find_input_csv()
    print("Using input file:", input_file)
    print("Loading dataset...")

    df = pd.read_csv(input_file)

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        print("\nERROR: Missing required columns:", missing)
        print("Columns found:\n", list(df.columns))
        raise SystemExit(1)

    stats = {
        "zero_width_removed": Counter(),
        "rows_with_zero_width": Counter(),
        "control_removed_total": 0,
        "rows_with_control": Counter(),
        "placeholders_replaced": Counter(),
        "rows_with_placeholders": Counter(),
        "examples": defaultdict(list),
    }

    print("Cleaning columns:", TARGET_COLS)
    for col in TARGET_COLS:
        df[col] = df[col].apply(lambda x: clean_real_noise(x, stats=stats, colname=col))

    print("Saving:", OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Writing report:", REPORT_FILE)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("CLEANING REPORT (minimal: real noise + ED placeholders)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"INPUT FILE:  {input_file}\n")
        f.write(f"OUTPUT CSV:  {OUTPUT_FILE}\n")
        f.write(f"OUTPUT TXT:  {REPORT_FILE}\n\n")

        f.write("COLUMNS CLEANED:\n")
        for c in TARGET_COLS:
            f.write(f" - {c}\n")
        f.write("\n")

        f.write("REMOVED (real NLP noise):\n")
        f.write(" - Zero-width / invisible chars:\n")
        for zw in ZERO_WIDTH_CHARS:
            f.write(f"    * {char_info(zw)}\n")
        f.write(" - Control chars: U+0000–U+001F and U+007F–U+009F\n\n")

        f.write("REPLACED (EmpatheticDialogues placeholders):\n")
        for tok, repl in ED_PLACEHOLDERS.items():
            f.write(f" - {tok}  ->  {repr(repl)}\n")
        f.write("\n")

        f.write("TOTAL COUNTS (all cleaned columns combined):\n")
        f.write("\nZero-width removed occurrences:\n")
        if stats["zero_width_removed"]:
            for ch, n in stats["zero_width_removed"].most_common():
                f.write(f" - {char_info(ch)} : {n}\n")
        else:
            f.write(" - none\n")

        f.write(f"\nControl chars removed total occurrences: {stats['control_removed_total']}\n")

        f.write("\nPlaceholders replaced occurrences:\n")
        if stats["placeholders_replaced"]:
            for tok, n in stats["placeholders_replaced"].most_common():
                f.write(f" - {tok} : {n}\n")
        else:
            f.write(" - none\n")

        f.write("\nROWS AFFECTED (per column):\n")
        for col in TARGET_COLS:
            f.write(f"\n[{col}]\n")
            f.write(f" - rows with zero-width chars: {stats['rows_with_zero_width'][col]}\n")
            f.write(f" - rows with control chars:    {stats['rows_with_control'][col]}\n")
            f.write(f" - rows with placeholders:     {stats['rows_with_placeholders'][col]}\n")

        f.write("\nEXAMPLES (before -> after), first 5 per column:\n")
        for col in TARGET_COLS:
            f.write(f"\n[{col}]\n")
            if stats["examples"][col]:
                for i, (before, after) in enumerate(stats["examples"][col], 1):
                    f.write(f"{i}. BEFORE: {before}\n")
                    f.write(f"   AFTER : {after}\n\n")
            else:
                f.write(" - no changes\n")

        f.write("\nNOT CHANGED:\n")
        f.write(" - smart quotes / em dashes / emojis / accents / casing / wording\n")
        f.write(" - no lowercasing, lemmatisation, or 'beautifying'\n")

    print("Done.")
    print("Cleaned file:", OUTPUT_FILE)
    print("Report file:", REPORT_FILE)


if __name__ == "__main__":
    main()