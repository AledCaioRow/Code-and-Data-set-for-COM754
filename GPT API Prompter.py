import csv
import time
from collections import defaultdict
from openai import OpenAI, RateLimitError, APIError

# ====== CONFIG ======
INPUT_CSV = "listener_generation_prompts.csv"                 # your test file
OUTPUT_CSV = "TheDataset.csv"       # output with prompts + gpt_listener
MAX_ROWS = None                            # set to e.g. 50 for a small test
MAX_RETRIES = 5
SPEAKER_HISTORY_LIMIT = 4                  # how many previous speaker messages to show
USE_API = True                             # set False to just build prompts without calling GPT
# ====================

client = OpenAI()  # or OpenAI(api_key="YOUR_KEY_HERE")


def build_prompt_from_speaker_history(speaker_history, current_speaker_msg):
    """
    speaker_history: list of previous SPEAKER messages (strings) in THIS conversation.
    current_speaker_msg: the new message we want to respond to.
    """
    if speaker_history:
        # cap history length
        recent = speaker_history[-SPEAKER_HISTORY_LIMIT:]
        history_lines = "\n".join(f"- {utt}" for utt in recent)

        prompt = f"""You are an empathetic listener in a short emotional conversation.

Earlier in this conversation, the speaker said:
{history_lines}

Now the speaker says something new.

SPEAKER-MESSAGE:
"{current_speaker_msg}"

Please respond empathetically and naturally to this new SPEAKER-MESSAGE in under 30 words.
Only respond to this new message, but you may use the history as background context."""
    else:
        prompt = f"""You are an empathetic listener in a short emotional conversation.

SPEAKER-MESSAGE:
"{current_speaker_msg}"

Please respond empathetically and naturally to this SPEAKER-MESSAGE in under 30 words."""
    return prompt.strip()


def call_gpt(prompt: str) -> str:
    """Call GPT once with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            wait = 5 * attempt
            print(f"[429 RateLimit] attempt {attempt}/{MAX_RETRIES}, waiting {wait}s")
            time.sleep(wait)
        except APIError:
            wait = 3 * attempt
            print(f"[APIError] attempt {attempt}/{MAX_RETRIES}, waiting {wait}s")
            time.sleep(wait)

    print("ERROR: GPT failed after retries; returning empty string.")
    return ""


def main():
    # Load rows and group by conversation id
    convs = defaultdict(list)
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            convs[row["conv_id"]].append(row)

    all_rows_out = []
    total_rows_used = 0
    stop = False

    for conv_id, rows in convs.items():
        if stop:
            break

        # ensure sorted by utterance_idx
        rows.sort(key=lambda r: int(r["utterance_idx"]))

        speaker_history = []  # previous speaker_last_message for THIS conv

        for row in rows:
            # ensure new columns exist
            if "gpt_prompt" not in row:
                row["gpt_prompt"] = ""
            if "gpt_listener" not in row:
                row["gpt_listener"] = ""

            current_speaker_msg = row["speaker_last_message"]

            # build prompt using prior history (if any)
            prompt = build_prompt_from_speaker_history(speaker_history, current_speaker_msg)
            row["gpt_prompt"] = prompt

            gpt_reply = ""
            if USE_API:
                print(f"[{total_rows_used+1}] conv_id={row['conv_id']} utt={row['utterance_idx']}")
                gpt_reply = call_gpt(prompt)

            row["gpt_listener"] = gpt_reply
            all_rows_out.append(row)

            # update history AFTER using this row
            speaker_history.append(current_speaker_msg)
            total_rows_used += 1

            if MAX_ROWS is not None and total_rows_used >= MAX_ROWS:
                stop = True
                break

    if not all_rows_out:
        print("No rows processed.")
        return

    fieldnames = list(all_rows_out[0].keys())
    if "gpt_prompt" not in fieldnames:
        fieldnames.append("gpt_prompt")
    if "gpt_listener" not in fieldnames:
        fieldnames.append("gpt_listener")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows_out)

    print(f"Done. Processed {total_rows_used} rows.")
    print(f"Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
