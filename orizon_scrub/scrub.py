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

    def __init__(self, device: str | None = None, biases: dict | None = None):
        self._device = device
        self._biases = dict(biases or HIGH_RECALL_BIASES)
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
        return base, merge_spans(regex_pii_spans(base), model_spans)


class Scrubber:
    """Replace detected PII with consistent, per-conversation numbered placeholders.

    Only content is modified: message ``content`` (string or multimodal parts),
    tool-call ``arguments`` (JSON parsed, string values scrubbed, re-serialized),
    tool-result content, and legacy ``function_call`` arguments. Roles, tool
    names, ids, ordering, token counts, and every other field are left untouched.
    """

    def __init__(self, detector):
        self.detector = detector

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
        # Normalize to alphanumerics so one entity written in different formats
        # (e.g. a card as "4242 4242..." / "4242-4242..." / "4242...") maps to a
        # single token within the conversation. Fall back to the raw form if the
        # value has no alphanumerics.
        norm = re.sub(r"[^a-z0-9]+", "", value.lower()) or value.strip().lower()
        key = (category, norm)
        token = alias.get(key)
        if token is None:
            per_cat[category] = per_cat.get(category, 0) + 1
            prefix = CATEGORY_PREFIX.get(category)
            if prefix is None:
                prefix = re.sub(r"[^A-Za-z0-9]+", "_", category.upper()).strip("_") or "PII"
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


def _card_char_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of Luhn-valid 13-19 digit card numbers within ``text``.

    Scans each maximal digit run for the longest, earliest Luhn-valid window so a
    card adjacent to other digits is still located precisely.
    """
    spans: list[tuple[int, int]] = []
    for m in _DIGIT_RUN_RE.finditer(text):
        run = m.group()
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


def regex_pii_spans(text: str) -> list[Span]:
    """High-precision spans for emails, credit cards (Luhn-checked), and secrets."""
    spans: list[Span] = []
    for m in _EMAIL_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "private_email"))
    for rx in _SECRET_RES:
        for m in rx.finditer(text):
            spans.append(Span(m.start(), m.end(), "secret"))
    for start, end in _card_char_spans(text):
        spans.append(Span(start, end, "account_number"))
    return merge_spans(spans, [])


def find_leaks(text: str) -> list[tuple[str, str]]:
    """Return ``(kind, matched_text)`` for residual emails, cards, or secrets."""
    hits: list[tuple[str, str]] = []
    for m in _EMAIL_RE.finditer(text):
        hits.append(("email", m.group()))
    for rx in _SECRET_RES:
        for m in rx.finditer(text):
            hits.append(("secret", m.group()))
    for start, end in _card_char_spans(text):
        hits.append(("card", text[start:end]))
    return hits
