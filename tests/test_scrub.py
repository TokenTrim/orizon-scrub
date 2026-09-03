"""Scrubber tests: structure preservation, placeholder consistency, leak-freedom.

These use a deterministic regex stub in place of the model so they are fast and
require no weights download. The real model is exercised separately (see
test_real_model_smoke, gated behind ORIZON_SCRUB_E2E).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest

from orizon_scrub.scrub import (
    CustomPattern,
    PrivacyFilterDetector,
    Scrubber,
    Span,
    find_leaks,
    load_custom_patterns,
    merge_spans,
    regex_pii_spans,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"

# Ordered (category, pattern) list mimicking openai/privacy-filter's v2 taxonomy.
_STUB_PATTERNS = [
    ("private_email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("private_url", re.compile(r"https?://[^\s\"'>]+")),
    ("private_address", re.compile(r"\d+\s+[A-Za-z ]+?,\s+[A-Za-z]+,\s+[A-Z]{2}\s+\d{5}")),
    ("account_number", re.compile(r"\bAC-\d+\b")),
    ("account_number", re.compile(r"\b(?:\d[ \-]?){13,19}(?<=\d)")),  # cards
    ("private_phone", re.compile(r"\+?\d?[\s\-()]*\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}")),
    ("private_date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("secret", re.compile(r"\bsk-[A-Za-z0-9\-]{8,}\b")),
    ("private_person", re.compile(r"\b(?:Sarah Johnson|Michael Chen|Priya Patel)\b")),
]


class StubDetector:
    """Deterministic regex detector returning non-overlapping spans, like opf."""

    def detect(self, text):
        found = []
        for category, rx in _STUB_PATTERNS:
            for m in rx.finditer(text):
                if m.end() > m.start():
                    found.append((m.start(), m.end(), category))
        # Greedy longest-first non-overlap resolution (opf guarantees non-overlap).
        found.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
        kept, occupied = [], []
        for start, end, category in found:
            if any(start < oe and end > os_ for os_, oe in occupied):
                continue
            occupied.append((start, end))
            kept.append(Span(start, end, category))
        return text, kept


@pytest.fixture(scope="module")
def scrubbed():
    scrubber = Scrubber(StubDetector())
    counts: Counter = Counter()
    originals = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    results = [scrubber.scrub_conversation(o, counts) for o in originals]
    return originals, results, counts


def _conv(results, cid):
    return next(c for c in results if c.get("id") == cid or c.get("trace_id") == cid)


def test_all_seeded_pii_is_gone(scrubbed):
    _, results, _ = scrubbed
    blob = "\n".join(json.dumps(c, ensure_ascii=False) for c in results)
    for secret in [
        "Sarah Johnson", "sarah.johnson@acmemail.com", "AC-99823145",
        "4242 4242 4242 4242", "Michael Chen", "742 Evergreen Terrace",
        "2025-03-15", "sk-live-abc123XYZ456def789ghi012jkl",
        "priya.patel@example.org", "Priya Patel",
        "https://app.example.com/users/mchen",
    ]:
        assert secret not in blob, f"leaked: {secret!r}"


def test_leak_check_finds_nothing(scrubbed):
    _, results, _ = scrubbed
    for c in results:
        assert find_leaks(json.dumps(c, ensure_ascii=False)) == []


def test_placeholder_consistency(scrubbed):
    _, results, _ = scrubbed
    blob = json.dumps(_conv(results, "conv-001"), ensure_ascii=False)
    # One person across the whole conversation -> one token, reused.
    assert blob.count("[PERSON_1]") >= 4
    assert "[PERSON_2]" not in blob
    assert "[EMAIL_1]" in blob and "[PHONE_1]" in blob
    # Two distinct account numbers (AC-... and the card) -> two distinct tokens.
    assert "[ACCOUNT_1]" in blob and "[ACCOUNT_2]" in blob


def test_case_insensitive_entity_maps_to_same_token():
    scrubber = Scrubber(StubDetector())
    conv = {"messages": [
        {"role": "user", "content": "Sarah Johnson called."},
        {"role": "assistant", "content": "Noted, Sarah Johnson."},
    ]}
    out = scrubber.scrub_conversation(conv, Counter())
    t0 = out["messages"][0]["content"]
    t1 = out["messages"][1]["content"]
    assert "[PERSON_1]" in t0 and "[PERSON_1]" in t1


def test_structure_is_untouched(scrubbed):
    originals, results, _ = scrubbed
    for orig, out in zip(originals, results):
        assert set(orig.keys()) == set(out.keys())
        assert orig.get("usage") == out.get("usage")
        assert orig.get("id") == out.get("id")
        assert len(orig["messages"]) == len(out["messages"])
        for om, sm in zip(orig["messages"], out["messages"]):
            assert om["role"] == sm["role"]
            assert om.get("tool_call_id") == sm.get("tool_call_id")
            assert om.get("name") == sm.get("name")
            o_tc, s_tc = om.get("tool_calls"), sm.get("tool_calls")
            assert (o_tc is None) == (s_tc is None)
            if o_tc:
                assert len(o_tc) == len(s_tc)
                for otc, stc in zip(o_tc, s_tc):
                    assert otc["id"] == stc["id"]
                    assert otc["type"] == stc["type"]
                    assert otc["function"]["name"] == stc["function"]["name"]


def test_tool_call_arguments_still_parse(scrubbed):
    _, results, _ = scrubbed
    seen = 0
    for c in results:
        for m in c["messages"]:
            for tc in m.get("tool_calls", []):
                args = json.loads(tc["function"]["arguments"])  # must not raise
                assert isinstance(args, dict)
                seen += 1
                if tc["function"]["name"] == "lookup_account":
                    assert set(args.keys()) == {"name", "email"}  # keys preserved
    assert seen >= 3


def test_tool_result_keys_preserved_values_scrubbed(scrubbed):
    _, results, _ = scrubbed
    conv = _conv(results, "conv-001")
    tool_msgs = [m for m in conv["messages"] if m["role"] == "tool"]
    lookup = json.loads(tool_msgs[0]["content"])
    assert set(lookup.keys()) == {"account_number", "status", "phone"}
    assert lookup["status"] == "locked"          # non-PII value untouched
    assert lookup["account_number"] == "[ACCOUNT_1]"
    assert lookup["phone"] == "[PHONE_1]"


def test_attachment_removed(scrubbed):
    _, results, counts = scrubbed
    conv = _conv(results, "conv-003")
    blob = json.dumps(conv, ensure_ascii=False)
    assert "iVBORw0KGgo" not in blob and "data:image" not in blob
    assert "[ATTACHMENT REMOVED]" in blob
    assert counts["__attachments__"] >= 1
    # The text part alongside the image is still scrubbed.
    assert "Priya Patel" not in blob


def test_data_uri_string_is_treated_as_attachment():
    scrubber = Scrubber(StubDetector())
    conv = {"messages": [{"role": "user", "content": "data:image/png;base64,AAAABBBB"}]}
    counts: Counter = Counter()
    out = scrubber.scrub_conversation(conv, counts)
    assert out["messages"][0]["content"] == "[ATTACHMENT REMOVED]"
    assert counts["__attachments__"] == 1


def test_regex_pii_spans_catches_structured_pii():
    # The model is unreliable on formatted cards; regex must catch them.
    for card in ["4242 4242 4242 4242", "4111 1111 1111 1111", "4242424242424242"]:
        cats = {s.category for s in regex_pii_spans(f"pay with {card} today")}
        assert "account_number" in cats, card
    assert any(s.category == "private_email" for s in regex_pii_spans("mail a@b.co"))
    assert any(s.category == "secret" for s in regex_pii_spans("key sk-live-abc123XYZ456def789"))


def test_regex_pii_spans_rejects_non_cards():
    # A phone number and a random non-Luhn 16-digit id must not be flagged as cards.
    assert not [s for s in regex_pii_spans("call 415-555-0142") if s.category == "account_number"]
    assert not [s for s in regex_pii_spans("id 1234567812345678") if s.category == "account_number"]


def test_detection_is_a_superset_of_leak_check():
    # Anything the leak check would flag must be a detectable (thus redactable) span.
    text = "a@b.com card 4242 4242 4242 4242 key sk-live-abc123XYZ456def789ghi"
    assert find_leaks(text)  # sanity: this text does leak
    spans = regex_pii_spans(text)
    remaining = text
    for sp in sorted(spans, key=lambda s: s.start, reverse=True):
        remaining = remaining[: sp.start] + " " + remaining[sp.end :]
    assert find_leaks(remaining) == []


def test_merge_spans_primary_wins_on_overlap():
    primary = [Span(0, 19, "account_number")]      # regex: whole card
    secondary = [Span(0, 19, "private_phone")]      # model: mislabeled same span
    merged = merge_spans(primary, secondary)
    assert len(merged) == 1
    assert merged[0].category == "account_number"
    # Non-overlapping model spans survive.
    merged2 = merge_spans([Span(0, 5, "private_email")], [Span(10, 15, "private_person")])
    assert {s.category for s in merged2} == {"private_email", "private_person"}


def test_dict_shaped_content_is_scrubbed():
    scrubber = Scrubber(StubDetector())
    conv = {"messages": [{"role": "user", "content": {"note": "customer Sarah Johnson at a@b.com"}}]}
    out = scrubber.scrub_conversation(conv, Counter())
    note = out["messages"][0]["content"]["note"]
    assert "Sarah Johnson" not in note and "a@b.com" not in note
    assert "note" in out["messages"][0]["content"]  # key preserved
    assert find_leaks(json.dumps(out)) == []


def test_fused_card_is_detected():
    # A Luhn-valid card fused to adjacent digits must still be caught (no over-consume).
    text = "amount 991234567890123452 usd"  # 1234567890123452 is Luhn-valid
    spans = [s for s in regex_pii_spans(text) if s.category == "account_number"]
    assert spans, "fused card not detected"
    assert "1234567890123452" in text[spans[0].start:spans[0].end]
    assert find_leaks(text)  # and the leak net also catches it


def test_digit_run_inside_identifier_is_not_a_card():
    # A hex trace id holds a Luhn-valid substring but is a structural field, never
    # a card. A run flanked by an ASCII letter must not be flagged (this is the
    # ORI-141 false positive that aborted every PostHog pull).
    trace_id = "039626639469462ca37adcb9810f3724"  # contains Luhn-valid 9626639469462
    assert find_leaks(trace_id) == []
    assert [s for s in regex_pii_spans(trace_id) if s.category == "account_number"] == []
    call_id = "call_039626639469462ca37adcb9810f3724_0_0"
    assert find_leaks(call_id) == []
    # And embedded in a serialized empty conversation (the exact failing payload).
    payload = json.dumps({"trace_id": trace_id, "messages": []}, ensure_ascii=False)
    assert find_leaks(payload) == []


class _RegexDetector:
    """Detector whose spans come purely from the regex layer (incl. extra patterns)."""

    def __init__(self, extra=()):
        self.extra = extra

    def detect(self, text):
        return text, regex_pii_spans(text, self.extra)


def test_redact_mode_is_unlinkable():
    # Redact mode collapses every value of a category to one unnumbered token, so
    # two distinct people are indistinguishable and linkage is removed.
    scrubber = Scrubber(StubDetector(), mode="redact")
    conv = {"messages": [{"role": "user", "content": "Sarah Johnson met Michael Chen"}]}
    out = json.dumps(scrubber.scrub_conversation(conv, Counter()), ensure_ascii=False)
    assert "Sarah Johnson" not in out and "Michael Chen" not in out
    assert out.count("[PERSON]") == 2  # both, same token
    assert "[PERSON_1]" not in out and "[PERSON_2]" not in out


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        Scrubber(StubDetector(), mode="anonymize")


def test_custom_patterns_detected_scrubbed_and_leak_checked():
    cp = [CustomPattern("employee_id", re.compile(r"EMP-\d{4}"))]
    text = "ticket from EMP-1234 and EMP-1234 again"
    # Detected as a span with the custom category...
    cats = {s.category for s in regex_pii_spans(text, cp)}
    assert "employee_id" in cats
    # ...caught by the leak check...
    assert ("employee_id", "EMP-1234") in find_leaks(text, cp)
    # ...and replaced with a category-derived placeholder, consistently.
    scrubber = Scrubber(_RegexDetector(cp))
    conv = {"messages": [{"role": "user", "content": text}]}
    out = json.dumps(scrubber.scrub_conversation(conv, Counter()), ensure_ascii=False)
    assert "EMP-1234" not in out
    assert out.count("[EMPLOYEE_ID_1]") == 2  # same value -> same token
    # And a clean payload no longer leaks the custom pattern.
    assert find_leaks(out, cp) == []


def test_load_custom_patterns(tmp_path):
    p = tmp_path / "patterns.json"
    p.write_text(json.dumps([
        {"category": "employee_id", "pattern": r"EMP-\d{4}"},
        {"category": "internal_url", "pattern": r"intranet/\w+", "ignore_case": True},
    ]))
    pats = load_custom_patterns(str(p))
    assert [c.category for c in pats] == ["employee_id", "internal_url"]
    assert pats[0].regex.search("EMP-9999")
    assert pats[1].regex.search("INTRANET/Foo")  # ignore_case honored
    # Malformed files are rejected clearly.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"pattern": "x"}]))  # missing category
    with pytest.raises(ValueError):
        load_custom_patterns(str(bad))


def test_real_card_next_to_punctuation_still_caught():
    # The identifier guard must not swallow real cards bounded by quotes/punctuation.
    for text in ['"card":"4242424242424242"', "card=4242 4242 4242 4242.", "(4111111111111111)"]:
        assert find_leaks(text), text
        assert [s for s in regex_pii_spans(text) if s.category == "account_number"], text


def test_attachment_preserves_metadata_enums():
    scrubber = Scrubber(StubDetector())
    conv = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD", "detail": "high"}},
    ]}]}
    counts: Counter = Counter()
    out = scrubber.scrub_conversation(conv, counts)
    part = out["messages"][0]["content"][0]
    assert part["type"] == "image_url"                       # structure preserved
    assert part["image_url"]["url"] == "[ATTACHMENT REMOVED]"  # payload blanked
    assert part["image_url"]["detail"] == "high"             # metadata enum kept
    assert counts["__attachments__"] == 1


def test_intrastring_numbering_is_document_order():
    scrubber = Scrubber(StubDetector())
    conv = {"messages": [{"role": "user", "content": "Contact Sarah Johnson or Michael Chen"}]}
    out = scrubber.scrub_conversation(conv, Counter())
    text = out["messages"][0]["content"]
    # First-appearing entity gets _1, not the last.
    assert text.index("[PERSON_1]") < text.index("[PERSON_2]")


def test_email_regex_does_not_backtrack_catastrophically():
    import time

    payload = "x@" + "a" * 100000  # '@' then a long run that never forms a valid TLD
    t0 = time.perf_counter()
    assert find_leaks(payload) == []       # no valid email -> nothing flagged
    assert regex_pii_spans(payload) == []
    assert time.perf_counter() - t0 < 2.0  # linear, not O(n^2)


def test_non_media_content_part_is_scrubbed():
    scrubber = Scrubber(StubDetector())
    conv = {"messages": [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "The customer Sarah Johnson at priya.patel@example.org"},
        {"type": "text", "text": "How can I help?"},
    ]}]}
    out = scrubber.scrub_conversation(conv, Counter())
    blob = json.dumps(out)
    assert "Sarah Johnson" not in blob and "priya.patel@example.org" not in blob
    assert out["messages"][0]["content"][0]["type"] == "thinking"  # structure kept
    assert find_leaks(blob) == []


def test_card_formats_map_to_same_token():
    scrubber = Scrubber(StubDetector())
    conv = {"messages": [
        {"role": "user", "content": "pay 4242424242424242"},
        {"role": "user", "content": "confirm 4242 4242 4242 4242"},
        {"role": "user", "content": "again 4242-4242-4242-4242"},
    ]}
    out = scrubber.scrub_conversation(conv, Counter())
    tokens = [m["content"].split()[-1] for m in out["messages"]]
    assert tokens == ["[ACCOUNT_1]"] * 3  # one card in three formats -> one token


def test_wizard_local_file(monkeypatch, tmp_path):
    import orizon_scrub.__main__ as m

    monkeypatch.setattr(m, "PrivacyFilterDetector", lambda *a, **k: StubDetector())
    out = tmp_path / "wiz.jsonl"
    # local; path; output; device; mode(pseudonymize); patterns(skip)
    answers = iter(["1", str(FIXTURE), str(out), "cpu", "1", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    rc = m.main(["--wizard"])
    assert rc == 0
    assert out.exists()
    assert len([x for x in out.read_text().splitlines() if x.strip()]) == 3


def _opf_ready():
    try:
        import opf  # noqa: F401
    except ImportError:
        return False
    return os.environ.get("ORIZON_SCRUB_E2E") == "1"


@pytest.mark.skipif(not _opf_ready(), reason="set ORIZON_SCRUB_E2E=1 with opf installed")
def test_real_model_smoke():
    scrubber = Scrubber(PrivacyFilterDetector(device="cpu"))
    conv = {"messages": [{"role": "user", "content": "My name is Alice Smith, email alice@example.com."}]}
    out = scrubber.scrub_conversation(conv, Counter())
    text = out["messages"][0]["content"]
    assert "Alice Smith" not in text and "alice@example.com" not in text
    assert find_leaks(json.dumps(out)) == []
