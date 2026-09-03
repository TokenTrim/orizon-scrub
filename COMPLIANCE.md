# Compliance and data handling

This document describes how `orizon-scrub` handles data so your security and
privacy reviewers can assess it. It is not legal advice. Whether sharing scrubbed
traces with a third party is permitted for your data depends on your obligations
and contracts. Involve your own counsel.

## What leaves your machine

`orizon-scrub` runs entirely on the machine you run it on. The only outbound
network traffic is:

1. **The model download.** On first run the ~2.8 GB Privacy Filter model is fetched
   from Hugging Face to `~/.opf/`. After that the tool runs offline. You can
   pre-stage this model and run the scrub with no network at all (set
   `OPF_CHECKPOINT` to a local path).
2. **Your own PostHog queries.** Only in `--posthog` mode, and only to the PostHog
   host you supply, using your own key, to read your own events.

Nothing else is sent anywhere. Your raw traces are never transmitted. Only the
redacted output file, which you produce and control, is suitable for sharing.

To verify: run the local-file path (`orizon-scrub traces.jsonl`) on a machine with
networking disabled after the model is cached. It completes with no network.

## What the output is

The output is **pseudonymized**, not anonymized. Direct identifiers are replaced,
but under regimes such as GDPR pseudonymized data is still personal data, because
it can in principle be re-identified (for example by linkage with the surrounding,
un-redacted context). Treat the scrubbed output as sensitive and keep it under the
same access controls and retention limits as other personal data. A data
processing agreement between you and the recipient is still required.

`orizon-scrub` does **not** store the mapping from a placeholder back to its
original value. The alias table lives in memory for the duration of a single
conversation and is discarded. So the output on its own is not reversible by the
recipient.

## Replacement modes

- **`--mode pseudonymize`** (default): each distinct value becomes a consistent
  numbered token, `[PERSON_1]`, `[EMAIL_1]`, and so on. The same value maps to the
  same token within a conversation, so the trace stays coherent and analyzable.
  This keeps within-trace linkage (you can tell two spans were the same person).
- **`--mode redact`**: each value becomes an unnumbered token, `[PERSON]`,
  `[EMAIL]`. Every value of a category is indistinguishable from every other, which
  removes within-trace linkage for teams that want zero linkability. Traces are
  less analyzable in this mode.

## Detection

Detection is a hybrid, which is the recommended approach because no single machine
learning model reliably catches structured identifiers:

- **OpenAI Privacy Filter** (the `opf` model), run at a high-recall operating point,
  detects contextual PII: names, addresses, phones, dates, URLs.
- **Deterministic regular expressions plus a Luhn check** detect emails, credit
  cards, and common secret and API-key formats, and take precedence on overlap.
- **Custom recognizers** (`--patterns file.json`) let you add your own formats,
  such as internal employee or account identifiers. They feed both detection and
  the final leak check, so anything they match is redacted and, if it ever
  survives, fails the run. See `examples/patterns.example.json`.

Detection is deliberately biased toward over-redaction (higher recall). It is a
data-minimization aid, not a guarantee. Keep human review for high-sensitivity
data.

## Fail-closed leak check

After scrubbing, a regex leak check runs over the written output. If any email,
credit card, secret, or custom-pattern match survives, the run exits non-zero and
writes no output file. In `--posthog` mode the resume cursor is not advanced, so
the affected data can be re-fetched and re-scrubbed.

## Redaction report

`--report report.json` writes a machine-readable report containing counts only,
never any redacted values: conversations processed, incomplete and no-content
counts, attachments removed, and spans redacted per category. Use it to audit what
was stripped without exposing anything sensitive.

## What is removed versus kept

- **Text** (message content, tool-call arguments, tool results) is scrubbed and
  kept as placeholders. Message structure, roles, tool names, ids, ordering, and
  token counts are preserved unchanged.
- **Binary media** (images, audio, video, files, and large `data:` or URL blobs)
  is removed entirely and replaced with `[ATTACHMENT REMOVED]`. The tool does not
  transcribe or OCR media, so any content inside an attachment is dropped, not
  redacted in place.

## Recommended checklist before sharing traces

1. Choose the mode that matches your linkability requirement.
2. Add custom recognizers for any identifiers specific to your domain.
3. Run with `--report` and review the counts.
4. Spot-check the output, especially for high-sensitivity data and non-English
   text (the model is English-first).
5. Confirm a data processing agreement and a lawful basis are in place with the
   recipient.
