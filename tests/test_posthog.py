"""PostHog tests: window parsing, HogQL keyset pagination, cursor durability, stitching.

All PostHog HTTP is mocked (no live key). The query function is injected via the
``query_fn`` parameter so tests control exactly what each page returns. The
cursor-durability tests drive ``main()`` with a stubbed model + a fake ``pull``.
"""

from __future__ import annotations

import json

import pytest

from orizon_scrub.posthog import (
    Rec,
    build_hogql,
    convert_messages,
    fetch_rows,
    load_cursor,
    parse_window,
    pull,
    rows_to_conversations,
    save_cursor,
)

COLUMNS = [
    "uuid", "timestamp", "distinct_id", "trace_id", "session_id", "span_id",
    "parent_id", "model", "provider", "input", "output_choices", "tools",
    "input_tokens", "output_tokens", "is_error",
]


def _page(rows):
    return {"columns": COLUMNS, "results": [[r.get(c) for c in COLUMNS] for r in rows]}


class FakeQuery:
    """Returns queued pages in order; records each request body."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.bodies = []

    def __call__(self, url, headers, body):
        self.bodies.append(body)
        return self.pages.pop(0) if self.pages else {"columns": COLUMNS, "results": []}

    def hogql(self, i):
        return self.bodies[i]["query"]["query"]


# -- window + query building ------------------------------------------------

def test_parse_window():
    assert parse_window("30d") == (30, "DAY")
    assert parse_window("12h") == (12, "HOUR")
    assert parse_window("4w") == (4, "WEEK")
    with pytest.raises(ValueError):
        parse_window("30x")


def test_build_hogql_has_keyset_and_window():
    sql = build_hogql(30, "DAY", "2026-08-01 00:00:00.000", "abc", 5000)
    assert "INTERVAL 30 DAY" in sql
    assert "event = '$ai_generation'" in sql
    assert "toDateTime64('2026-08-01 00:00:00.000', 6)" in sql
    assert "toString(uuid) > 'abc'" in sql
    assert "ORDER BY timestamp ASC, toString(uuid) ASC" in sql
    assert "LIMIT 5000" in sql
    assert "OFFSET" not in sql  # OFFSET paging is rejected by PostHog for personal keys


def test_normalize_ts_handles_posthog_iso_and_epoch():
    from orizon_scrub.posthog import normalize_ts

    # PostHog's own ISO form -> toDateTime64-parseable form, full precision kept.
    assert normalize_ts("2026-09-02T08:22:22.135000Z") == "2026-09-02 08:22:22.135000"
    # Sub-millisecond precision is preserved (not truncated to ms), so a keyset
    # cursor on such a row is compared exactly and not re-selected.
    assert normalize_ts("2026-09-02T08:22:22.135999Z") == "2026-09-02 08:22:22.135999"
    # Epoch sentinel and already-normalized values are unchanged (idempotent).
    assert normalize_ts("1970-01-01 00:00:00.000") == "1970-01-01 00:00:00.000"
    assert normalize_ts("2026-09-02 08:22:22.135") == "2026-09-02 08:22:22.135"
    # No fractional part -> milliseconds added.
    assert normalize_ts("2026-09-02T08:22:22Z") == "2026-09-02 08:22:22.000"


def test_build_hogql_normalizes_iso_cursor():
    # A resumed pull carries PostHog's ISO timestamp; the query must embed the
    # normalized form, never the raw 'T'/'Z' form that 500s ClickHouse.
    sql = build_hogql(30, "DAY", "2026-09-02T08:22:22.135000Z", "u1", 5000)
    assert "toDateTime64('2026-09-02 08:22:22.135000', 6)" in sql
    assert "2026-09-02T08:22:22.135000Z" not in sql


def test_build_hogql_first_pull_has_no_uuid_string_compare():
    # Epoch sentinel with an empty uuid: comparing the UUID column to '' is a
    # HogQL type error (HTTP 400), so the first pull must be timestamp-only.
    sql = build_hogql(30, "DAY", "1970-01-01 00:00:00.000", "", 5000)
    assert "uuid > ''" not in sql
    assert "toString(uuid) > ''" not in sql
    assert "timestamp > toDateTime64('1970-01-01 00:00:00.000', 6)" in sql


# -- cursor round-trip ------------------------------------------------------

def test_cursor_roundtrip_and_bad_files(tmp_path):
    p = tmp_path / "cursor.json"
    assert load_cursor(str(p)) == ("1970-01-01 00:00:00.000", "")  # missing -> epoch
    save_cursor(str(p), "2026-09-01T00:00:00Z", "u9")
    assert load_cursor(str(p)) == ("2026-09-01T00:00:00Z", "u9")
    for bad in ("null", "[]", "123", '"resume"', "{ not json"):
        p.write_text(bad)
        assert load_cursor(str(p)) == ("1970-01-01 00:00:00.000", "")  # tolerated


# -- pagination (no disk I/O in fetch_rows) ---------------------------------

def test_fetch_rows_pagination_and_final_cursor():
    r = lambda u, t: {"uuid": u, "timestamp": t, "trace_id": "t", "input": "[]"}
    fake = FakeQuery([
        _page([r("u1", "2026-09-01T00:00:00Z"), r("u2", "2026-09-01T00:00:01Z")]),
        _page([r("u3", "2026-09-01T00:00:02Z")]),  # short page -> stop
    ])
    rows, final = fetch_rows(
        "https://us.posthog.com", "1", "phx_key", "30d", ("1970-01-01 00:00:00.000", ""),
        limit=2, query_fn=fake,
    )
    assert [x["uuid"] for x in rows] == ["u1", "u2", "u3"]
    assert final == ("2026-09-01T00:00:02Z", "u3")  # last row seen
    # Second page query resumed from the first page's last row, with the ISO
    # timestamp normalized to a toDateTime64-parseable form (ORI-141: the raw
    # 'T'/'Z' form 500s in ClickHouse, which broke every page after the first).
    assert "u2" in fake.hogql(1) and "2026-09-01 00:00:01.000" in fake.hogql(1)
    assert "2026-09-01T00:00:01Z" not in fake.hogql(1)


def test_fetch_rows_resumes_from_start_cursor():
    fake = FakeQuery([])  # empty -> stop; nothing fetched
    rows, final = fetch_rows(
        "https://us.posthog.com", "1", "phx_key", "30d",
        ("2026-08-15 09:00:00.000", "prev"), limit=100, query_fn=fake,
    )
    assert rows == [] and final is None  # no rows -> no cursor advance
    assert "2026-08-15 09:00:00.000" in fake.hogql(0)
    assert "toString(uuid) > 'prev'" in fake.hogql(0)


# -- stitching + conversion -------------------------------------------------

def test_convert_messages_parts():
    msgs = convert_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image", "image": "data:image/png;base64,AAA"},
            {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}},
        ]},
    ], "tid")
    assert len(msgs) == 1
    m = msgs[0]
    assert m["role"] == "user"
    assert isinstance(m["content"], list)
    assert {"type": "text", "text": "look at this"} in m["content"]
    assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}} in m["content"]
    assert m["tool_calls"][0]["function"]["name"] == "f"
    assert m["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'  # object -> JSON string


def test_stitch_groups_by_trace_and_flags_incomplete():
    rows = [
        {  # t1: complete
            "uuid": "a", "timestamp": "2026-09-01T00:00:00Z", "trace_id": "t1",
            "model": "gpt-5-mini", "is_error": None,
            "input": json.dumps([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "I'm Michael Chen"},
            ]),
            "output_choices": json.dumps([
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "function", "function": {"name": "lookup", "arguments": {"q": "Michael Chen"}}},
                ]},
            ]),
        },
        {  # t2: errored -> incomplete
            "uuid": "b", "timestamp": "2026-09-01T00:00:01Z", "trace_id": "t2",
            "is_error": True,
            "input": json.dumps([{"role": "user", "content": "hi"}]),
            "output_choices": json.dumps([{"role": "assistant", "content": "ok"}]),
        },
    ]
    convs = rows_to_conversations(rows)
    assert [c["trace_id"] for c, _ in convs] == ["t1", "t2"]

    c1, incomplete1 = convs[0]
    assert incomplete1 is False
    assert [m["role"] for m in c1["messages"]] == ["system", "user", "assistant"]
    assert c1["model"] == "gpt-5-mini"
    tc = c1["messages"][2]["tool_calls"][0]
    assert tc["function"]["name"] == "lookup"
    assert json.loads(tc["function"]["arguments"]) == {"q": "Michael Chen"}

    _, incomplete2 = convs[1]
    assert incomplete2 is True


def test_stitch_multiple_generations_uses_last_input():
    """The last generation carries the fullest history; earlier ones are subsumed."""
    rows = [
        {"uuid": "a", "timestamp": "2026-09-01T00:00:00Z", "trace_id": "t",
         "input": json.dumps([{"role": "user", "content": "one"}]),
         "output_choices": json.dumps([{"role": "assistant", "content": "first"}])},
        {"uuid": "b", "timestamp": "2026-09-01T00:00:05Z", "trace_id": "t",
         "input": json.dumps([
             {"role": "user", "content": "one"},
             {"role": "assistant", "content": "first"},
             {"role": "user", "content": "two"},
         ]),
         "output_choices": json.dumps([{"role": "assistant", "content": "second"}])},
    ]
    (conv, incomplete), = rows_to_conversations(rows)
    assert incomplete is False
    assert [m["content"] for m in conv["messages"]] == ["one", "first", "two", "second"]


def test_missing_output_is_incomplete():
    rows = [{"uuid": "a", "timestamp": "2026-09-01T00:00:00Z", "trace_id": "t",
             "input": json.dumps([{"role": "user", "content": "hi"}]),
             "output_choices": None}]
    (_, incomplete), = rows_to_conversations(rows)
    assert incomplete is True


def test_pull_end_to_end():
    fake = FakeQuery([_page([
        {"uuid": "a", "timestamp": "2026-09-01T00:00:00Z", "trace_id": "t1",
         "input": json.dumps([{"role": "user", "content": "hi"}]),
         "output_choices": json.dumps([{"role": "assistant", "content": "hello"}])},
    ])])
    recs, final = pull(
        "https://us.posthog.com", "1", "phx_key", "30d",
        ("1970-01-01 00:00:00.000", ""), limit=100, query_fn=fake,
    )
    assert [r.id for r in recs] == ["t1"]
    assert recs[0].conv["messages"][0]["content"] == "hi"
    assert final == ("2026-09-01T00:00:00Z", "a")


# -- cursor durability at the main() level ----------------------------------

class _Blind:
    """A detector that finds nothing (isolates the pipeline from the model)."""

    def __init__(self, *a, **k):
        pass

    def detect(self, text):
        return text, []


def _run_posthog_main(monkeypatch, tmp_path, recs, final_cursor, out_name="out.jsonl",
                      extra_args=()):
    import orizon_scrub.__main__ as m

    monkeypatch.setattr(m, "PrivacyFilterDetector", _Blind)
    monkeypatch.setattr(m.posthog, "pull", lambda *a, **k: (recs, final_cursor))
    cursor = tmp_path / "cur.json"
    out = tmp_path / out_name
    rc = m.main([
        "--posthog", "--host", "https://us.posthog.com", "--api-key", "phx_k",
        "--project-id", "1", "--cursor-file", str(cursor), "-o", str(out),
        *extra_args,
    ])
    return rc, cursor, out


def test_main_writes_report(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    recs = [Rec(id="t1", conv={"trace_id": "t1", "messages": [{"role": "user", "content": "hi"}]})]
    rc, _, _ = _run_posthog_main(
        monkeypatch, tmp_path, recs, ("2026-09-01T00:00:00Z", "a"),
        extra_args=["--report", str(report), "--mode", "redact"],
    )
    assert rc == 0
    data = json.loads(report.read_text())
    assert data["tool"] == "orizon-scrub"
    assert data["mode"] == "redact"
    assert data["conversations_processed"] == 1
    assert data["spans_redacted_total"] == 0  # blind detector, no spans
    assert "spans_by_category" in data and "generated_at" in data


def test_report_path_collision_rejected(monkeypatch, tmp_path):
    # --report must not clobber the output or the PostHog cursor (ORI-141 review).
    recs = [Rec(id="t1", conv={"trace_id": "t1", "messages": [{"role": "user", "content": "hi"}]})]
    out_path = tmp_path / "out.jsonl"
    with pytest.raises(SystemExit):
        _run_posthog_main(monkeypatch, tmp_path, recs, ("2026-09-01T00:00:00Z", "a"),
                          extra_args=["--report", str(out_path)])  # == -o
    with pytest.raises(SystemExit):
        _run_posthog_main(monkeypatch, tmp_path, recs, ("2026-09-01T00:00:00Z", "a"),
                          extra_args=["--report", str(tmp_path / "cur.json")])  # == cursor


def test_local_empty_conversation_is_not_dropped(monkeypatch, tmp_path):
    # The empty-trace skip is PostHog-only; a local file is passed through as-is.
    import orizon_scrub.__main__ as m

    monkeypatch.setattr(m, "PrivacyFilterDetector", _Blind)
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        json.dumps({"id": "a", "messages": []}) + "\n"
        + json.dumps({"id": "b", "messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    out = tmp_path / "in.scrubbed.jsonl"
    rc = m.main([str(inp), "-o", str(out)])
    assert rc == 0
    ids = [json.loads(x)["id"] for x in out.read_text().splitlines() if x.strip()]
    assert ids == ["a", "b"]  # empty local conversation preserved


def test_fetch_rows_stops_when_cursor_does_not_advance():
    # Safety net against an infinite loop if a full page never advances the keyset.
    row = {"uuid": "u0", "timestamp": "2026-01-01 00:00:00.000", "trace_id": "t", "input": "[]"}
    page = {"columns": COLUMNS, "results": [[row.get(c) for c in COLUMNS]] * 2}
    calls = {"n": 0}

    def always_same(url, headers, body):
        calls["n"] += 1
        assert calls["n"] < 5, "fetch_rows looped instead of stopping"
        return page  # full page, last row == start cursor -> no advance

    rows, final = fetch_rows("h", "1", "k", "30d",
                             ("2026-01-01 00:00:00.000", "u0"), limit=2, query_fn=always_same)
    assert calls["n"] == 1


def test_main_bad_patterns_file_errors(monkeypatch, tmp_path):
    recs = [Rec(id="t1", conv={"trace_id": "t1", "messages": [{"role": "user", "content": "hi"}]})]
    with pytest.raises(SystemExit):
        _run_posthog_main(
            monkeypatch, tmp_path, recs, ("2026-09-01T00:00:00Z", "a"),
            extra_args=["--patterns", str(tmp_path / "does-not-exist.json")],
        )


def test_main_commits_cursor_only_after_success(monkeypatch, tmp_path):
    recs = [Rec(id="t1", conv={"trace_id": "t1", "messages": [{"role": "user", "content": "hi"}]})]
    rc, cursor, out = _run_posthog_main(monkeypatch, tmp_path, recs, ("2026-09-01T00:00:00Z", "a"))
    assert rc == 0
    assert out.exists()
    assert json.loads(cursor.read_text()) == {"ts": "2026-09-01T00:00:00Z", "uuid": "a"}


def test_main_skips_traces_with_no_messages(monkeypatch, tmp_path):
    # Metadata-only projects (no $ai_input) stitch to empty conversations. They must
    # not be written and must not trip the leak check via their trace id (ORI-141).
    recs = [
        Rec(id="039626639469462ca37adcb9810f3724",
            conv={"trace_id": "039626639469462ca37adcb9810f3724", "messages": []},
            incomplete=True),
        Rec(id="t2", conv={"trace_id": "t2", "messages": [{"role": "user", "content": "hi"}]}),
    ]
    rc, cursor, out = _run_posthog_main(monkeypatch, tmp_path, recs, ("2026-09-01T00:00:00Z", "a"))
    assert rc == 0
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert [c["trace_id"] for c in lines] == ["t2"]  # empty trace dropped, content kept


def test_main_all_empty_exits_zero_and_commits_cursor(monkeypatch, tmp_path):
    # An all-metadata pull is a clean success: no file written, but the cursor
    # advances so the same empty traces are not re-pulled forever.
    recs = [Rec(id="e1", conv={"trace_id": "e1", "messages": []}, incomplete=True)]
    rc, cursor, out = _run_posthog_main(monkeypatch, tmp_path, recs, ("2026-09-05T00:00:00Z", "z"))
    assert rc == 0
    assert not out.exists()
    assert json.loads(cursor.read_text()) == {"ts": "2026-09-05T00:00:00Z", "uuid": "z"}


def test_main_does_not_commit_cursor_on_leak(monkeypatch, tmp_path):
    # Blind detector leaves an email -> leak check fails -> no output, no cursor.
    recs = [Rec(id="t1", conv={"trace_id": "t1", "messages": [{"role": "user", "content": "mail a@b.com"}]})]
    rc, cursor, out = _run_posthog_main(monkeypatch, tmp_path, recs, ("2026-09-01T00:00:00Z", "a"))
    assert rc == 1
    assert not out.exists()
    assert not cursor.exists()  # cursor NOT advanced -> data can be re-fetched


def test_main_resume_appends_to_existing_export(monkeypatch, tmp_path):
    import orizon_scrub.__main__ as m

    monkeypatch.setattr(m, "PrivacyFilterDetector", _Blind)
    cursor = tmp_path / "cur.json"
    cursor.write_text(json.dumps({"ts": "2026-09-01T00:00:00Z", "uuid": "a"}))  # resuming
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"trace_id": "t0", "messages": [{"role": "user", "content": "old"}]}) + "\n")
    recs = [Rec(id="t1", conv={"trace_id": "t1", "messages": [{"role": "user", "content": "new"}]})]
    monkeypatch.setattr(m.posthog, "pull", lambda *a, **k: (recs, ("2026-09-02T00:00:00Z", "b")))
    rc = m.main([
        "--posthog", "--host", "h", "--api-key", "phx_k", "--project-id", "1",
        "--cursor-file", str(cursor), "-o", str(out),
    ])
    assert rc == 0
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert [c["trace_id"] for c in lines] == ["t0", "t1"]  # appended, prior export kept
    assert json.loads(cursor.read_text()) == {"ts": "2026-09-02T00:00:00Z", "uuid": "b"}


def test_main_rolls_back_partial_append_on_failure(monkeypatch, tmp_path):
    import orizon_scrub.__main__ as m

    monkeypatch.setattr(m, "PrivacyFilterDetector", _Blind)
    cursor = tmp_path / "cur.json"
    cursor.write_text(json.dumps({"ts": "t1", "uuid": "a"}))  # append mode
    out = tmp_path / "out.jsonl"
    out.write_text("PRIOR EXPORT\n")
    recs = [Rec(id="t2", conv={"trace_id": "t2", "messages": [{"role": "user", "content": "new"}]})]
    monkeypatch.setattr(m.posthog, "pull", lambda *a, **k: (recs, ("t2ts", "b")))

    def boom(_fd):
        raise OSError("disk full")

    monkeypatch.setattr(m.os, "fsync", boom)  # fail the durable append
    with pytest.raises(OSError):
        m.main([
            "--posthog", "--host", "h", "--api-key", "phx_k", "--project-id", "1",
            "--cursor-file", str(cursor), "-o", str(out),
        ])
    assert out.read_text() == "PRIOR EXPORT\n"  # partial append rolled back
    assert json.loads(cursor.read_text()) == {"ts": "t1", "uuid": "a"}  # cursor not advanced


def test_wizard_posthog(monkeypatch, tmp_path):
    import orizon_scrub.__main__ as m

    monkeypatch.setattr(m, "PrivacyFilterDetector", _Blind)
    recs = [Rec(id="t1", conv={"trace_id": "t1", "messages": [{"role": "user", "content": "hi"}]})]
    monkeypatch.setattr(m.posthog, "pull", lambda *a, **k: (recs, ("ts", "u")))
    monkeypatch.setattr(m, "_prompt_secret", lambda label: "phx_secret")
    out = tmp_path / "w.jsonl"
    cur = tmp_path / "c.json"
    # choice=PostHog, host, project id, window, output, device, mode, patterns(skip)
    answers = iter(["2", "https://us.posthog.com", "999", "30d", str(out), "cpu", "1", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    rc = m.main(["--wizard", "--cursor-file", str(cur)])
    assert rc == 0
    assert out.exists()
    assert json.loads(out.read_text().splitlines()[0])["trace_id"] == "t1"


def test_main_empty_pull_does_not_overwrite_existing_output(monkeypatch, tmp_path):
    import orizon_scrub.__main__ as m

    monkeypatch.setattr(m, "PrivacyFilterDetector", _Blind)
    monkeypatch.setattr(m.posthog, "pull", lambda *a, **k: ([], None))
    cursor = tmp_path / "cur.json"
    out = tmp_path / "out.jsonl"
    out.write_text("PREVIOUS GOOD OUTPUT\n")
    rc = m.main([
        "--posthog", "--host", "h", "--api-key", "phx_k", "--project-id", "1",
        "--cursor-file", str(cursor), "-o", str(out),
    ])
    assert rc == 0
    assert out.read_text() == "PREVIOUS GOOD OUTPUT\n"  # not clobbered by empty result
    assert not cursor.exists()


# -- transient-error retry (ORI-141) ----------------------------------------

class _Resp:
    def __init__(self, code, payload=None, text="err"):
        self.status_code = code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_http_query_retries_transient_5xx(monkeypatch):
    # A cold ClickHouse 500 that clears on retry must not abort the pull.
    import requests

    from orizon_scrub import posthog as ph

    ok = {"columns": ["x"], "results": [[1]]}
    calls = {"n": 0}

    def fake_post(url, headers=None, data=None, timeout=None):
        calls["n"] += 1
        return _Resp(500) if calls["n"] < 3 else _Resp(200, ok)

    monkeypatch.setattr(requests, "post", fake_post)
    assert ph._http_query("u", {}, {"query": {}}, sleep=lambda _s: None) == ok
    assert calls["n"] == 3


def test_http_query_does_not_retry_4xx(monkeypatch):
    # A deterministic 400 (bad query / auth / scope) must surface immediately.
    import requests

    from orizon_scrub import posthog as ph

    calls = {"n": 0}

    def fake_post(url, headers=None, data=None, timeout=None):
        calls["n"] += 1
        return _Resp(400, text="bad query")

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(requests.exceptions.HTTPError):
        ph._http_query("u", {}, {"query": {}}, sleep=lambda _s: None)
    assert calls["n"] == 1
