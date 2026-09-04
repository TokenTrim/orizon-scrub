# orizon-scrub

Strip PII from your AI agent traces, **fully locally**, using
[OpenAI's Privacy Filter](https://github.com/openai/privacy-filter).

## Get PII-stripped traces

```bash
git clone https://github.com/TokenTrim/orizon-scrub.git
cd orizon-scrub
uv run orizon-scrub
```

That's it. A short wizard asks where your traces are — a local JSONL file, or
pulled from PostHog — and writes `*.scrubbed.jsonl` with every name, email,
phone, address, account number, and secret replaced by consistent placeholders
(`[PERSON_1]`, `[EMAIL_1]`, `[ACCOUNT_1]`, …).

The first run installs dependencies and downloads the ~2.8 GB model to `~/.opf/`;
after that startup is instant.

## What you need

- [`uv`](https://docs.astral.sh/uv/) and Python 3.11+. `uv run` handles the rest.
- **No OpenAI or Anthropic API key.** The Privacy Filter model runs entirely on
  your machine; nothing is sent to an LLM API.
- The model is downloaded once from the **public** `openai/privacy-filter`
  repository on Hugging Face, so **no Hugging Face token is required**. You may
  optionally set `HF_TOKEN` to get higher rate limits and a faster first download.
- **Only for PostHog mode:** a PostHog **personal** API key (`phx_...`) with the
  `query:read` scope. That is the single credential the tool needs. The public
  project key (`phc_...`) is write-only and cannot read traces. Scrubbing a local
  JSONL file needs no credentials at all.

### Skip the wizard

```bash
# a local file (one conversation per line, OpenAI chat format)
uv run orizon-scrub traces.jsonl                 # -> traces.scrubbed.jsonl

# or pull from PostHog (personal phx_ key with the query:read scope)
export POSTHOG_HOST=https://us.posthog.com       # use https://eu.posthog.com for EU
export POSTHOG_API_KEY=phx_...
export POSTHOG_PROJECT_ID=12345
uv run orizon-scrub --posthog --window 30d       # -> traces.scrubbed.jsonl
```

Your PostHog project must be **capturing LLM inputs and outputs** (the
`$ai_input` and `$ai_output_choices` properties). If it only records metadata
(token counts, cost), orizon-scrub prints `N trace(s) had no message content`
and there is nothing to scrub. Turn on input/output capture in your PostHog LLM
analytics settings, or export the traces to a JSONL file and scrub that instead.

## Options

```bash
# Redact instead of pseudonymize: [PERSON] with no numbering, so the same value
# is indistinguishable from any other (zero within-trace linkage).
uv run orizon-scrub traces.jsonl --mode redact

# Add your own recognizers (employee ids, internal account formats, ...).
# They feed both detection and the leak check. See examples/patterns.example.json.
uv run orizon-scrub traces.jsonl --patterns patterns.json

# Write an audit report of what was stripped (counts only, never values).
uv run orizon-scrub traces.jsonl --report report.json
```

## Good to know

- **Local only** — nothing leaves your machine except the model download and your
  own PostHog queries. See [COMPLIANCE.md](COMPLIANCE.md) for the full data-handling
  and reviewer notes.
- **Pseudonymize or redact** — the default keeps consistent numbered placeholders
  (`[PERSON_1]`) so traces stay analyzable; `--mode redact` drops the numbering for
  zero linkability.
- **Extend it** — add domain-specific identifiers with `--patterns`; get an audit
  trail with `--report`.
- **Structure is preserved** — roles, tool names, ids, ordering, and token counts
  are untouched; only content is scrubbed, so tool-call JSON still parses and the
  same entity maps to the same placeholder across a conversation.
- **Fails closed** — a regex check runs over the output; if any email, card, or
  secret slips through, the run exits non-zero and writes nothing.
- **Redaction ≠ anonymization** — it's a data-minimization aid, not a compliance
  guarantee. Keep human review for high-sensitivity data.
