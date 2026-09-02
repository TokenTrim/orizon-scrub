"""orizon-scrub CLI: strip PII from AI agent traces locally.

Two input paths, one scrubbing pipeline:

    orizon-scrub traces.jsonl                 # scrub a local JSONL file
    orizon-scrub --posthog --window 30d       # pull from PostHog, then scrub

Both write ``<input>.scrubbed.jsonl`` (temp file, renamed on success only) and
print a summary. A regex leak check runs over the scrubbed output; any residual
email/card/secret fails the run with exit code 1 and no output file written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter

from . import posthog
from .posthog import Rec
from .scrub import CATEGORY_PREFIX, PrivacyFilterDetector, Scrubber, find_leaks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orizon-scrub",
        description="Strip PII from AI agent traces locally using OpenAI's Privacy Filter model.",
    )
    p.add_argument("input", nargs="?", help="Path to a JSONL file (one conversation per line).")
    p.add_argument("--posthog", action="store_true", help="Pull traces from PostHog instead of a file.")
    p.add_argument("--window", default="30d", help="PostHog time window, e.g. 30d, 12h, 4w (default: 30d).")
    p.add_argument("-o", "--output", help="Output path (default: <input>.scrubbed.jsonl).")
    p.add_argument("--host", default=os.environ.get("POSTHOG_HOST"), help="PostHog host, e.g. https://us.posthog.com (env: POSTHOG_HOST).")
    p.add_argument("--api-key", default=os.environ.get("POSTHOG_API_KEY"), help="PostHog personal API key, phx_... with query:read (env: POSTHOG_API_KEY).")
    p.add_argument("--project-id", default=os.environ.get("POSTHOG_PROJECT_ID"), help="PostHog project id (env: POSTHOG_PROJECT_ID).")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Inference device (default: auto-detect).")
    p.add_argument("--cursor-file", default=".orizon-scrub-cursor.json", help="PostHog resume cursor file (default: .orizon-scrub-cursor.json).")
    p.add_argument("--wizard", action="store_true", help="Interactive setup (also launched when run with no arguments in a terminal).")
    return p


def default_output_path(input_path: str) -> str:
    if input_path.endswith(".jsonl"):
        return input_path[: -len(".jsonl")] + ".scrubbed.jsonl"
    return input_path + ".scrubbed.jsonl"


def read_jsonl(path: str):
    """Yield one :class:`Rec` per line; malformed lines are marked skipped."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                yield Rec(id=f"line-{lineno}", conv={}, skipped=True, reason=str(exc))
                continue
            if not isinstance(obj, dict):
                yield Rec(id=f"line-{lineno}", conv={}, skipped=True, reason="line is not a JSON object")
                continue
            tid = obj.get("trace_id") or obj.get("id") or f"line-{lineno}"
            yield Rec(id=str(tid), conv=obj)


def _source(args):
    """Return ``(records, default_output_path, commit, append)``.

    ``commit`` is called only after the scrubbed output is durably written; for
    PostHog it advances the resume cursor so the cursor never moves past data that
    was not committed. ``append`` is True when resuming an existing PostHog pull
    (a cursor file already exists), so the new conversations are appended to the
    accumulating export rather than overwriting it. For file input, ``commit`` is a
    no-op and ``append`` is False.
    """
    if args.posthog:
        missing = [
            name
            for name, val in (("--host/POSTHOG_HOST", args.host),
                              ("--api-key/POSTHOG_API_KEY", args.api_key),
                              ("--project-id/POSTHOG_PROJECT_ID", args.project_id))
            if not val
        ]
        if missing:
            raise SystemExit(f"error: PostHog mode requires: {', '.join(missing)}")
        resuming = os.path.exists(args.cursor_file)
        start = posthog.load_cursor(args.cursor_file)
        records, final_cursor = posthog.pull(
            args.host, args.project_id, args.api_key, args.window, start
        )

        def commit():
            if final_cursor:
                posthog.save_cursor(args.cursor_file, *final_cursor)

        return records, "traces.scrubbed.jsonl", commit, resuming
    if not args.input:
        raise SystemExit("error: provide an input JSONL file, or use --posthog")
    if not os.path.exists(args.input):
        raise SystemExit(f"error: input file not found: {args.input}")
    return read_jsonl(args.input), default_output_path(args.input), (lambda: None), False


def _prompt(label, default=None, required=False, validate=None):
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{label}{suffix}: ").strip()
        if not raw and default is not None:
            raw = default
        if not raw:
            if required:
                print("  (required)")
                continue
            return ""
        if validate and not validate(raw):
            print("  (not found / invalid, try again)")
            continue
        return raw


def _prompt_secret(label):
    import getpass

    while True:
        val = getpass.getpass(f"{label}: ").strip()
        if val:
            return val
        print("  (required)")


def _prompt_choice(label, options):
    print(label)
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        raw = input("Choose [1]: ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw)
        print("  (enter a number)")


def _model_cached() -> bool:
    path = os.environ.get("OPF_CHECKPOINT") or os.path.expanduser("~/.opf/privacy_filter")
    return os.path.isdir(path)


def run_wizard(args):
    """Fill ``args`` interactively. Returns the same namespace, ready for the pipeline."""
    print("orizon-scrub — interactive setup (Ctrl-C to cancel)\n")
    if _prompt_choice("Where are the traces?", ["Local JSONL file", "PostHog (pull traces)"]) == 1:
        args.posthog = False
        args.input = _prompt("Path to JSONL file", required=True, validate=os.path.exists)
        args.output = args.output or _prompt("Output file", default=default_output_path(args.input))
    else:
        args.posthog = True
        args.host = _prompt("PostHog host", default=args.host or "https://us.posthog.com")
        args.project_id = _prompt("PostHog project id", default=args.project_id, required=True)
        args.api_key = args.api_key or _prompt_secret("PostHog personal API key (phx_..., query:read scope)")
        args.window = _prompt("Time window (e.g. 30d, 12h, 4w)", default=args.window or "30d")
        args.output = args.output or _prompt("Output file", default="traces.scrubbed.jsonl")
    dev = _prompt("Device (auto/cpu/cuda)", default=args.device or "auto")
    args.device = None if dev == "auto" else dev
    if not _model_cached():
        print("\nNote: the first scrub downloads the ~2.8GB Privacy Filter model to ~/.opf/.")
    print()
    return args


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.wizard or (not args.input and not args.posthog and sys.stdin.isatty()):
        try:
            args = run_wizard(args)
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.", file=sys.stderr)
            return 130
    records, default_out, commit, append = _source(args)
    out_path = args.output or default_out

    scrubber = Scrubber(PrivacyFilterDetector(device=args.device))
    counts: Counter = Counter()
    n_conv = n_incomplete = n_skipped = 0
    leaks: dict[str, list] = {}

    out_dir = os.path.dirname(os.path.abspath(out_path))
    fd, tmp_path = tempfile.mkstemp(prefix=".orizon-scrub-", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for rec in records:
                if rec.skipped:
                    n_skipped += 1
                    print(f"  skipped {rec.id}: {rec.reason}", file=sys.stderr)
                    continue
                if rec.incomplete:
                    n_incomplete += 1
                scrubbed = scrubber.scrub_conversation(rec.conv, counts)
                payload = json.dumps(scrubbed, ensure_ascii=False)
                hits = find_leaks(payload)
                if hits:
                    leaks[rec.id] = hits
                out.write(payload + "\n")
                n_conv += 1
    except BaseException:
        _quiet_remove(tmp_path)
        raise

    if leaks:
        # Fail loud; do NOT commit the cursor, so the pulled data can be re-fetched.
        _quiet_remove(tmp_path)
        print(
            f"\nLEAK CHECK FAILED: residual PII in {len(leaks)} conversation(s); "
            "no output written.",
            file=sys.stderr,
        )
        for tid, hits in leaks.items():
            kinds = ", ".join(sorted({k for k, _ in hits}))
            print(f"  {tid}: {kinds}", file=sys.stderr)
        return 1

    if n_conv == 0:
        # Never overwrite an existing good output with an empty file, and do not
        # advance the cursor (nothing was committed).
        _quiet_remove(tmp_path)
        existing = " (existing output left unchanged)" if os.path.exists(out_path) else ""
        print(f"\nNo conversations to scrub; nothing written{existing}.")
        return 0

    try:
        if append:
            # Resuming an existing pull: accumulate onto the prior export, durably,
            # before advancing the cursor. Roll back a partial append on failure so
            # the export never ends with a torn JSONL line.
            pre_len = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            try:
                with open(tmp_path, encoding="utf-8") as src, open(out_path, "a", encoding="utf-8") as dst:
                    dst.write(src.read())
                    dst.flush()
                    os.fsync(dst.fileno())
            except BaseException:
                try:
                    os.truncate(out_path, pre_len)
                except OSError:
                    pass
                raise
            _quiet_remove(tmp_path)
        else:
            os.replace(tmp_path, out_path)
    except BaseException:
        _quiet_remove(tmp_path)
        raise
    commit()  # advance the resume cursor only after output is durable
    _print_summary(out_path, n_conv, n_incomplete, n_skipped, counts)
    return 0


def _print_summary(out_path, n_conv, n_incomplete, n_skipped, counts) -> None:
    attachments = counts.pop("__attachments__", 0)
    total_spans = sum(counts.values())
    print("\norizon-scrub summary")
    print(f"  conversations processed : {n_conv}")
    print(f"  incomplete/partial      : {n_incomplete}")
    print(f"  skipped (malformed)     : {n_skipped}")
    print(f"  attachments removed     : {attachments}")
    print(f"  spans redacted          : {total_spans}")
    for category in sorted(counts):
        label = CATEGORY_PREFIX.get(category, category)
        print(f"      {label:<9} {counts[category]}")
    print(f"  output                  : {out_path}")


def _quiet_remove(path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
