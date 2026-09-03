"""PII span detection and consistent-placeholder replacement.

Detection is done by OpenAI's Privacy Filter model via the official ``opf``
package (https://github.com/openai/privacy-filter), run at a high-recall
operating point. The model returns character-offset spans + categories; we do
our own replacement so that the same entity maps to the same numbered token
within a conversation (PERSON_1, EMAIL_1, ...), which keeps traces analyzable.

The replacement/structure logic (:class:`Scrubber`) is independent of the model:
it takes any detector exposing ``detect(text) -> (base_text, [Span, ...])``,
which makes it fast and deterministic to unit-test without downloading weights.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass

# Model category label -> friendly placeholder prefix. The v2 taxonomy shipped by
# openai/privacy-filter is exactly these eight labels.
CATEGORY_PREFIX = {
    "private_person": "PERSON",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_address": "ADDRESS",
    "private_url": "URL",
    "private_date": "DATE",
    "account_number": "ACCOUNT",
    "secret": "SECRET",
}

ATTACHMENT_MARKER = "[ATTACHMENT REMOVED]"


def placeholder_prefix(category: str) -> str:
    """The token prefix used for a category in placeholders and summaries.

    ``private_email`` -> ``EMAIL`` (via the built-in map); an unmapped custom
    category like ``employee_id`` -> ``EMPLOYEE_ID``.
    """
    prefix = CATEGORY_PREFIX.get(category)
    if prefix is None:
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", category.upper()).strip("_") or "PII"
    return prefix

# Content-part types whose payload is binary/media (redact the blob, keep structure).
# Any other part type (text, thinking, reasoning, unknown) has its string values
# scrubbed so text-bearing PII is never passed through untouched.
_MEDIA_PART_TYPES = frozenset(
    {"image", "image_url", "input_image", "audio", "input_audio", "video", "file", "document"}
)

# High-recall Viterbi calibration for the opf CRF decoder. The shipped calibration
# is all-zeros; opf validates this artifact has EXACTLY these keys (see
# opf._core.decoding.resolve_viterbi_biases_from_calibration_path). Directions come
# from opf._core.decoding._transition_bias: leaving/avoiding the background state and
# extending/chaining spans maximizes recall (we would rather over-redact than leak).
HIGH_RECALL_BIASES = {
    "transition_bias_background_stay": -2.5,
    "transition_bias_background_to_start": 2.5,
    "transition_bias_inside_to_continue": 1.0,
    "transition_bias_inside_to_end": 0.0,
    "transition_bias_end_to_background": -1.0,
    "transition_bias_end_to_start": 1.0,
}


@dataclass(frozen=True)
class Span:
    """A detected PII span: character offsets into the detector's base text."""

    start: int
    end: int
    category: str


class PrivacyFilterDetector:
    """Lazy, reusable wrapper around the ``opf`` model at a high-recall operating point.

    The model (~2.8GB) is downloaded and loaded on first ``detect`` call, then
    reused for the whole run. ``opf``'s ``OPF()`` constructor defaults to CUDA and
    crashes on CPU-only machines, so the device is always set explicitly.
    """

    def __init__(self, device: str | None = None, biases: dict | None = None,
                 extra_patterns=()):
        self._device = device
        self._biases = dict(biases or HIGH_RECALL_BIASES)
        self._extra_patterns = tuple(extra_patterns)
        self._redactor = None
        self._calib_path: str | None = None

    def _ensure_loaded(self) -> None:
        if self._redactor is not None:
            return
        try:
            import torch
            from opf import OPF
        except ImportError as exc:  # pragma: no cover - exercised only without opf
            raise RuntimeError(
                "The 'opf' package (openai/privacy-filter) is required for scrubbing. "
                "Install it with:\n"
                "  pip install 'git+https://github.com/openai/privacy-filter.git'"
            ) from exc

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        fd, path = tempfile.mkstemp(prefix="orizon_scrub_viterbi_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"operating_points": {"default": {"biases": self._biases}}}, fh)
        self._calib_path = path
        redactor = OPF(
            output_mode="typed",
            decode_mode="viterbi",
            device=device,
            discard_overlapping_predicted_spans=False,
        )
        redactor.set_viterbi_decoder(calibration_path=path)
        self._redactor = redactor

    def detect(self, text: str) -> tuple[str, list[Span]]:
        """Return ``(base_text, spans)``.

        ``base_text`` is the string the spans index into. It equals ``text``
        unless the tokenizer round-trip did not reproduce the input exactly
        (``result.warning`` is set); in that case opf's spans refer to the
        decoded text, so we return that as the base to keep offsets valid.

        The model's spans are augmented with high-precision regex spans for
        emails, credit cards, and secret keys. The model is unreliable on these
        structured types (e.g. it mislabels or misses formatted card numbers),
        so the deterministic patterns — the same ones the leak check uses — take
        precedence on overlap. This guarantees regex-catchable PII is removed and
        maps consistently (a card is always ACCOUNT, never PHONE in one turn and
        missed in the next), leaving the leak check as a true redundant net.
        """
        if not text:
            return text, []
        self._ensure_loaded()
        result = self._redactor.redact(text)
        base = result.text
        model_spans = [Span(s.start, s.end, s.label) for s in result.detected_spans]
        return base, merge_spans(regex_pii_spans(base, self._extra_patterns), model_spans)


class Scrubber:
    """Replace detected PII with consistent, per-conversation numbered placeholders.

    Only content is modified: message ``content`` (string or multimodal parts),
    tool-call ``arguments`` (JSON parsed, string values scrubbed, re-serialized),
    tool-result content, and legacy ``function_call`` arguments. Roles, tool
    names, ids, ordering, token counts, and every other field are left untouched.
    """

    def __init__(self, detector, mode: str = "pseudonymize"):
        self.detector = detector
        # "pseudonymize": consistent numbered tokens ([PERSON_1]) that stay linked
        # within a conversation. "redact": unnumbered tokens ([PERSON]) so the same
        # value is indistinguishable from any other, removing within-trace linkage
        # for teams that want zero linkability.
        if mode not in ("pseudonymize", "redact"):
            raise ValueError(f"unknown mode {mode!r}; use 'pseudonymize' or 'redact'")
        self.mode = mode

    def scrub_conversation(self, conv: dict, counts: Counter) -> dict:
        """Return a scrubbed deep copy of one OpenAI-chat conversation dict.

        ``counts`` is updated in place with per-category span counts and an
        ``__attachments__`` tally.
        """
        conv = copy.deepcopy(conv)
        alias: dict[tuple[str, str], str] = {}
        per_cat: dict[str, int] = {}
        messages = conv.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    self._scrub_message(msg, alias, per_cat, counts)
        return conv

    # -- message-level ----------------------------------------------------

    def _scrub_message(self, msg, alias, per_cat, counts) -> None:
        role = msg.get("role")
        if "content" in msg:
            if role == "tool":
                # Tool results are usually JSON; scrub values, preserve keys/structure.
                msg["content"] = self._scrub_json_or_text(
                    msg["content"], alias, per_cat, counts
                )
            else:
                msg["content"] = self._scrub_content(
                    msg["content"], alias, per_cat, counts
                )
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    self._scrub_function_obj(tc.get("function"), alias, per_cat, counts)
        # Legacy single function_call shape.
        self._scrub_function_obj(msg.get("function_call"), alias, per_cat, counts)

    def _scrub_function_obj(self, fn, alias, per_cat, counts) -> None:
        if isinstance(fn, dict) and "arguments" in fn:
            fn["arguments"] = self._scrub_json_or_text(
                fn["arguments"], alias, per_cat, counts
            )

    # -- content parts ----------------------------------------------------

    def _scrub_content(self, content, alias, per_cat, counts):
        if isinstance(content, str):
            return self._scrub_text(content, alias, per_cat, counts)
        if isinstance(content, list):
            return [self._scrub_part(p, alias, per_cat, counts) for p in content]
        if isinstance(content, dict):
            # Non-standard object-shaped content: scrub its string values (keys kept).
            return self._scrub_json_value(content, alias, per_cat, counts)
        return content

    def _scrub_part(self, part, alias, per_cat, counts):
        if isinstance(part, str):
            return self._scrub_text(part, alias, per_cat, counts)
        if not isinstance(part, dict):
            return part
        ptype = part.get("type")
        ptype_l = ptype.lower() if isinstance(ptype, str) else ptype
        if ptype_l in ("function", "tool_call") or isinstance(part.get("function"), dict):
            part = dict(part)
            fn = part.get("function")
            if isinstance(fn, dict):
                fn = dict(fn)
                self._scrub_function_obj(fn, alias, per_cat, counts)
                part["function"] = fn
            return part
        if ptype_l in _MEDIA_PART_TYPES:
            return self._redact_attachment_part(part, counts)
        # text / thinking / reasoning / unknown: scrub every string value (keys kept)
        # so PII in any text-bearing field is removed, never passed through.
        return self._scrub_json_value(part, alias, per_cat, counts)

    def _redact_attachment_part(self, part, counts):
        """Blank an attachment's payload (base64/URI/large blob) while keeping its
        structure and small metadata enums (``detail``, ``format``, ``mime_type``)."""
        counts["__attachments__"] += 1

        def blank(value):
            if isinstance(value, str):
                return ATTACHMENT_MARKER if _looks_like_payload(value) else value
            if isinstance(value, dict):
                return {k: blank(v) for k, v in value.items()}
            if isinstance(value, list):
                return [blank(v) for v in value]
            return value

        return {k: (v if k == "type" else blank(v)) for k, v in part.items()}

    # -- JSON-aware and text scrubbing -----------------------------------

    def _scrub_json_or_text(self, value, alias, per_cat, counts):
        """Scrub a tool-call/tool-result payload, preserving JSON structure if present."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped[:1] in ("{", "["):
                try:
                    parsed = json.loads(value)
                except (ValueError, TypeError):
                    return self._scrub_text(value, alias, per_cat, counts)
                scrubbed = self._scrub_json_value(parsed, alias, per_cat, counts)
                return json.dumps(scrubbed, ensure_ascii=False)
            return self._scrub_text(value, alias, per_cat, counts)
        # Already-structured value (e.g. a function part with an object arguments).
        return self._scrub_json_value(value, alias, per_cat, counts)

    def _scrub_json_value(self, obj, alias, per_cat, counts):
        if isinstance(obj, dict):
            return {k: self._scrub_json_value(v, alias, per_cat, counts) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._scrub_json_value(v, alias, per_cat, counts) for v in obj]
        if isinstance(obj, str):
            return self._scrub_text(obj, alias, per_cat, counts)
        return obj

    def _scrub_text(self, text, alias, per_cat, counts):
        if not isinstance(text, str) or not text:
            return text
        if _is_attachment_payload(text):
            counts["__attachments__"] += 1
            return ATTACHMENT_MARKER
        base, spans = self.detector.detect(text)
        valid = [sp for sp in spans if 0 <= sp.start < sp.end <= len(base)]
        if not valid:
            return base
        # Assign placeholders in document order so numbering reads 1, 2, 3 ...
        plan = []
        for sp in sorted(valid, key=lambda s: s.start):
            token = self._placeholder(sp.category, base[sp.start : sp.end], alias, per_cat)
            counts[sp.category] += 1
            plan.append((sp, token))
        # Splice right-to-left so earlier offsets stay valid.
        out = base
        for sp, token in sorted(plan, key=lambda t: t[0].start, reverse=True):
            out = out[: sp.start] + token + out[sp.end :]
        return out

    def _placeholder(self, category, value, alias, per_cat) -> str:
        prefix = placeholder_prefix(category)
        if self.mode == "redact":
            # Unnumbered, unlinkable: every value of a category collapses to the
            # same token, so you cannot tell whether two spans were the same value.
            return f"[{prefix}]"
        # Normalize to alphanumerics so one entity written in different formats
        # (e.g. a card as "4242 4242..." / "4242-4242..." / "4242...") maps to a
        # single token within the conversation. Fall back to the raw form if the
        # value has no alphanumerics.
        norm = re.sub(r"[^a-z0-9]+", "", value.lower()) or value.strip().lower()
        key = (category, norm)
        token = alias.get(key)
        if token is None:
            per_cat[category] = per_cat.get(category, 0) + 1
            token = f"[{prefix}_{per_cat[category]}]"
            alias[key] = token
        return token


def _is_attachment_payload(text: str) -> bool:
    """True for a string that is a base64/URI attachment blob rather than prose."""
    return isinstance(text, str) and text.strip().lower().startswith("data:")


_PAYLOAD_PREFIXES = ("data:", "http://", "https://", "blob:")


def _looks_like_payload(text: str) -> bool:
    """True for an attachment payload (URI or large blob), not small metadata."""
    stripped = text.strip()
    return stripped.lower().startswith(_PAYLOAD_PREFIXES) or len(stripped) > 200


# ---------------------------------------------------------------------------
# High-precision structured-PII patterns. These serve two roles: (1) a detection
# supplement (the model is unreliable on these), and (2) the final leak safety-net
# over the scrubbed output. Using the SAME patterns for both means detection is a
# superset of the leak check, so a leak-check hit can only mean something slipped
# past replacement — a real reason to fail loudly.
# ---------------------------------------------------------------------------

# Bounded quantifiers keep this linear: an unbounded domain run before the TLD
# causes O(n^2) backtracking on pathological input (a long run after '@' with no
# valid TLD), which would hang both detection and the leak check.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9][A-Za-z0-9.\-]{0,253}\.[A-Za-z]{2,24}")
# A run of >=13 digits, optionally single-separated by spaces/hyphens. We scan
# windows inside each run for a Luhn-valid card, so a card fused to adjacent
# digits (which a single greedy match would over-consume and fail Luhn on) is
# still found.
_DIGIT_RUN_RE = re.compile(r"\d(?:[ \-]?\d){12,}")
_SECRET_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(?:sk|rk|pk|api)[_\-](?:live|test|prod|proj)[_\-][A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _luhn_ok(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_ascii_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _looks_like_identifier(text: str, start: int, end: int, run: str) -> bool:
    """True when the digit run at ``[start, end)`` is a fragment of an identifier
    (a hex trace id, a base64 token) rather than a card number.

    Only runs directly touching an ASCII letter are candidates: a card in prose or
    JSON is bounded by whitespace, quotes, or punctuation. A run written with
    internal separators (``4242 4242 ...``) is a formatted card, never an id. For a
    contiguous run glued to letters, inspect the enclosing alphanumeric token: if,
    after removing a single leading and trailing alphabetic run, letters remain
    interspersed among the digits, it is hex/base64 (e.g. ``039626639469462ca...``)
    and skipped; a clean word wrapped around a pure digit block (``card4242...4242``)
    is left for the Luhn check so real cards are never dropped.
    """
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if not (_is_ascii_letter(before) or _is_ascii_letter(after)):
        return False
    if any(c in " -" for c in run):
        return False
    left = start
    while left > 0 and text[left - 1].isascii() and text[left - 1].isalnum():
        left -= 1
    right = end
    while right < len(text) and text[right].isascii() and text[right].isalnum():
        right += 1
    token = text[left:right]
    i = 0
    while i < len(token) and _is_ascii_letter(token[i]):
        i += 1
    j = len(token)
    while j > i and _is_ascii_letter(token[j - 1]):
        j -= 1
    return any(_is_ascii_letter(c) for c in token[i:j])


def _card_char_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of Luhn-valid 13-19 digit card numbers within ``text``.

    Scans each maximal digit run for the longest, earliest Luhn-valid window so a
    card adjacent to other digits is still located precisely. Runs that are a
    fragment of an alphanumeric identifier are skipped (see
    :func:`_looks_like_identifier`): a hex trace id such as
    ``039626639469462ca...`` holds a Luhn-valid substring but is not a card, and it
    is a structural field the scrubber never touches, so matching it would both
    over-redact and trip the leak check on every conversation.
    """
    spans: list[tuple[int, int]] = []
    for m in _DIGIT_RUN_RE.finditer(text):
        run = m.group()
        if _looks_like_identifier(text, m.start(), m.end(), run):
            continue
        idx = [i for i, ch in enumerate(run) if ch.isdigit()]
        digits = "".join(run[i] for i in idx)
        n = len(digits)
        for length in range(min(19, n), 12, -1):
            hit = None
            for start in range(n - length + 1):
                if _luhn_ok(digits[start : start + length]):
                    hit = (m.start() + idx[start], m.start() + idx[start + length - 1] + 1)
                    break
            if hit:
                spans.append(hit)
                break
    return spans


def merge_spans(primary: list[Span], secondary: list[Span]) -> list[Span]:
    """Merge two span lists into a non-overlapping set; ``primary`` wins on overlap.

    Within each list, longer spans win. Used to give the deterministic regex spans
    precedence over the model's (which can mislabel structured PII), matching opf's
    own left-to-right, longest-preferred non-overlap policy.
    """
    kept: list[Span] = []
    occupied: list[tuple[int, int]] = []
    for group in (primary, secondary):  # primary placed first => wins on overlap
        for sp in sorted(group, key=lambda s: (-(s.end - s.start), s.start)):
            if any(sp.start < e and sp.end > s for s, e in occupied):
                continue
            occupied.append((sp.start, sp.end))
            kept.append(sp)
    return kept


@dataclass(frozen=True)
class CustomPattern:
    """A customer-supplied high-precision recognizer: a category and a compiled regex."""

    category: str
    regex: "re.Pattern"


def load_custom_patterns(path: str) -> list[CustomPattern]:
    """Load extra recognizers from a JSON file.

    Format: a JSON list of objects, each ``{"category": "<name>", "pattern":
    "<regex>", "ignore_case": <bool, optional>}``. The category names the
    placeholder prefix (``employee_id`` -> ``[EMPLOYEE_ID_1]``) and the leak-check
    kind. Because these deterministic patterns feed BOTH detection and the leak
    check, anything a pattern matches is redacted and, if it ever survives, fails
    the run. This lets a customer add their own ID or secret formats without code.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("custom patterns file must be a JSON list of objects")
    patterns: list[CustomPattern] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "category" not in item or "pattern" not in item:
            raise ValueError(f"pattern {i} must have 'category' and 'pattern'")
        category, pattern = item["category"], item["pattern"]
        if not isinstance(category, str) or not isinstance(pattern, str):
            raise ValueError(f"pattern {i}: 'category' and 'pattern' must be strings")
        try:
            flags = re.IGNORECASE if item.get("ignore_case") else 0
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"pattern {i}: invalid regex: {exc}") from exc
        patterns.append(CustomPattern(category, compiled))
    return patterns


def _custom_spans(text: str, extra_patterns) -> list[Span]:
    spans: list[Span] = []
    for cp in extra_patterns or ():
        for m in cp.regex.finditer(text):
            if m.end() > m.start():  # ignore zero-width matches
                spans.append(Span(m.start(), m.end(), cp.category))
    return spans


def regex_pii_spans(text: str, extra_patterns=()) -> list[Span]:
    """High-precision spans for emails, cards (Luhn-checked), secrets, and any
    customer-supplied ``extra_patterns``."""
    spans: list[Span] = []
    for m in _EMAIL_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "private_email"))
    for rx in _SECRET_RES:
        for m in rx.finditer(text):
            spans.append(Span(m.start(), m.end(), "secret"))
    for start, end in _card_char_spans(text):
        spans.append(Span(start, end, "account_number"))
    spans.extend(_custom_spans(text, extra_patterns))
    return merge_spans(spans, [])


# A generated replacement token: [PERSON], [EMAIL_1], [EMPLOYEE_ID_2], ...
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")


def find_leaks(text: str, extra_patterns=()) -> list[tuple[str, str]]:
    """Return ``(kind, matched_text)`` for residual emails, cards, secrets, or
    customer-supplied ``extra_patterns``.

    A custom pattern match that falls entirely inside a generated placeholder (for
    example ``\\d+`` matching the ``1`` in ``[EMPLOYEE_ID_1]``) is not a leak: the
    original value was removed. Such matches are ignored so a broad recognizer
    cannot make every run fail its own leak check. The built-in email/card/secret
    patterns never match a placeholder, so they scan the whole text unchanged.
    """
    hits: list[tuple[str, str]] = []
    for m in _EMAIL_RE.finditer(text):
        hits.append(("email", m.group()))
    for rx in _SECRET_RES:
        for m in rx.finditer(text):
            hits.append(("secret", m.group()))
    for start, end in _card_char_spans(text):
        hits.append(("card", text[start:end]))
    if extra_patterns:
        holes = [(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]
        for sp in _custom_spans(text, extra_patterns):
            if not any(hs <= sp.start and sp.end <= he for hs, he in holes):
                hits.append((sp.category, text[sp.start:sp.end]))
    return hits
