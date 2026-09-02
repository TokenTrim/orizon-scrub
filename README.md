# orizon-scrub

Strip PII from AI agent traces **locally** using [OpenAI's Privacy Filter](https://github.com/openai/privacy-filter)
(`openai/privacy-filter`, Apache-2.0). Traces come from a local JSONL file or are
pulled from your own PostHog LLM-analytics instance. Nothing leaves your machine
except the model download (Hugging Face, first run) and your own PostHog queries.

## Install

Clone and run with [`uv`](https://docs.astral.sh/uv/) — it resolves every
dependency (including the model package `opf` and Torch) into an isolated env on
first run:

```bash
git clone https://github.com/Lyadalachanchu/orizon-scrub.git
cd orizon-scrub
uv run orizon-scrub            # launches the wizard; first run installs deps
```

Or install into your own environment with `pip install -e .`. The Privacy Filter
model (~2.8 GB) downloads to `~/.opf/` on the first scrub.

## Quickstart — interactive wizard

Run with **no arguments** and it walks you through everything:

```text
$ orizon-scrub
orizon-scrub — interactive setup (Ctrl-C to cancel)

Where are the traces?
  1) Local JSONL file
  2) PostHog (pull traces)
Choose [1]: 1
Path to JSONL file: traces.jsonl
Output file [traces.scrubbed.jsonl]:
Device (auto/cpu/cuda) [auto]:
```

Choose PostHog instead and it prompts for the host, project id, personal API key
(entered hidden), and time window — no environment variables to set up by hand.

## Scripting — flags

```bash
# A) scrub a local JSONL file (one conversation per line, OpenAI chat format)
orizon-scrub traces.jsonl                     # -> traces.scrubbed.jsonl

# B) pull from PostHog, then scrub identically
export POSTHOG_HOST=https://us.posthog.com    # or eu / self-hosted app host
export POSTHOG_API_KEY=phx_...                # PERSONAL key with query:read scope
export POSTHOG_PROJECT_ID=12345
orizon-scrub --posthog --window 30d           # -> traces.scrubbed.jsonl
```

The PostHog pull paginates with a keyset cursor persisted to
`.orizon-scrub-cursor.json`, advanced **only after** the scrubbed output is
durably written — so an interrupted or failed run never loses data. A fresh pull
(no cursor file) writes the export atomically; while the cursor exists, later runs
append newly-arrived events to the same export and advance the cursor, so
resuming an interrupted pull accumulates rather than overwrites. A run that finds
nothing new leaves the export untouched. Delete `.orizon-scrub-cursor.json` (or
use a different `--cursor-file`) to start a new export from a full window. The
public project key (`phc_...`) is write-only and **cannot** read events — use a
personal API key.

## What it does

- Scrubs only content: message text, tool-call arguments (JSON parsed, string
  values scrubbed, re-serialized), and tool results. Roles, tool names, ids,
  ordering, and token counts are never touched.
- Consistent placeholders per conversation: the same entity → the same token
  (`[PERSON_1]`, `[EMAIL_1]`, `[ACCOUNT_1]`, ...), so traces stay analyzable.
- Base64 images / other non-text content become `[ATTACHMENT REMOVED]`.
- After scrubbing, a regex leak check (emails, credit cards, secret prefixes)
  runs over the output; any hit fails the run (exit 1) with the offending trace
  ids and **no** output file is written (temp file, renamed on success only).

## Caveat: redaction ≠ anonymization

The Privacy Filter is a data-minimization aid, not an anonymization or compliance
guarantee. It under-detects rare/non-English names and novel secret formats and
can over-redact. Placeholder consistency is by surface form: the same string maps
to the same token within a conversation, but the model may segment a multi-token
entity (e.g. an address) differently in prose vs. an isolated JSON value, so those
can get different tokens. Emails, credit cards, and secret keys are additionally
caught by deterministic patterns (the same ones the leak check uses). Keep human
review for high-sensitivity data.
