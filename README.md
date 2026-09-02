# orizon-scrub

Strip PII from your AI agent traces, **fully locally**, using
[OpenAI's Privacy Filter](https://github.com/openai/privacy-filter).

## Get PII-stripped traces

```bash
git clone https://github.com/Lyadalachanchu/orizon-scrub.git
cd orizon-scrub
uv run orizon-scrub
```

That's it. A short wizard asks where your traces are — a local JSONL file, or
pulled from PostHog — and writes `*.scrubbed.jsonl` with every name, email,
phone, address, account number, and secret replaced by consistent placeholders
(`[PERSON_1]`, `[EMAIL_1]`, `[ACCOUNT_1]`, …).

The first run installs dependencies and downloads the ~2.8 GB model to `~/.opf/`;
after that startup is instant.

### Skip the wizard

```bash
# a local file (one conversation per line, OpenAI chat format)
uv run orizon-scrub traces.jsonl                 # -> traces.scrubbed.jsonl

# or pull from PostHog (personal phx_ key with the query:read scope)
export POSTHOG_HOST=https://us.posthog.com
export POSTHOG_API_KEY=phx_...
export POSTHOG_PROJECT_ID=12345
uv run orizon-scrub --posthog --window 30d       # -> traces.scrubbed.jsonl
```

## Good to know

- **Local only** — nothing leaves your machine except the model download and your
  own PostHog queries.
- **Structure is preserved** — roles, tool names, ids, ordering, and token counts
  are untouched; only content is scrubbed, so tool-call JSON still parses and the
  same entity maps to the same placeholder across a conversation.
- **Fails closed** — a regex check runs over the output; if any email, card, or
  secret slips through, the run exits non-zero and writes nothing.
- **Redaction ≠ anonymization** — it's a data-minimization aid, not a compliance
  guarantee. Keep human review for high-sensitivity data.
