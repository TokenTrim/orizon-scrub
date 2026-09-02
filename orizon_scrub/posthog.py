"""Pull LLM traces from PostHog and stitch them into OpenAI-chat conversations.

PostHog LLM analytics emits flat ``$ai_generation`` events; each carries the full
message history (``$ai_input``) and the reply (``$ai_output_choices``) plus a
``$ai_trace_id`` that groups a conversation. We read them via the HogQL query API
using keyset pagination on ``(timestamp, uuid)`` (OFFSET paging is rejected with
HTTP 400 for personal API keys), persist a resume cursor, then group by trace id,
convert to the same chat format as a local JSONL file, and hand off to the scrubber.

Reads require a PostHog **personal** API key (``phx_...``) with the ``query:read``
scope. The public project key (``phc_...``) is write-only and cannot read events.
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

_WINDOW_UNITS = {"d": "DAY", "h": "HOUR", "w": "WEEK"}

# Columns selected from each $ai_generation event (AI fields live under properties.*).
_SELECT_COLUMNS = (
    "uuid",
    "timestamp",
    "distinct_id",
    "properties.$ai_trace_id AS trace_id",
    "properties.$ai_session_id AS session_id",
    "properties.$ai_span_id AS span_id",
    "properties.$ai_parent_id AS parent_id",
    "properties.$ai_model AS model",
    "properties.$ai_provider AS provider",
    "properties.$ai_input AS input",
    "properties.$ai_output_choices AS output_choices",
    "properties.$ai_tools AS tools",
    "properties.$ai_input_tokens AS input_tokens",
    "properties.$ai_output_tokens AS output_tokens",
    "properties.$ai_is_error AS is_error",
)

_EPOCH_TS = "1970-01-01 00:00:00.000"


@dataclass
class Rec:
    """A conversation ready to scrub, plus stitching metadata for the summary."""

    id: str
    conv: dict
    incomplete: bool = False
    skipped: bool = False
    reason: str | None = None


def parse_window(window: str) -> tuple[int, str]:
    """Parse a window like ``30d`` / ``12h`` / ``4w`` into ``(amount, HogQL unit)``."""
    m = re.fullmatch(r"\s*(\d+)\s*([dhw])\s*", window.lower())
    if not m:
        raise ValueError(
            f"Invalid --window {window!r}; use <number><d|h|w>, e.g. 30d, 12h, 4w."
        )
    return int(m.group(1)), _WINDOW_UNITS[m.group(2)]


def build_query_url(host: str, project_id: str) -> str:
    return f"{host.rstrip('/')}/api/projects/{project_id}/query/"


def build_hogql(amount: int, unit: str, cur_ts: str, cur_uuid: str, limit: int) -> str:
    """Build a keyset-paginated HogQL query for one page of $ai_generation events.

    ``cur_ts``/``cur_uuid`` come from PostHog's own response (or the epoch sentinel),
    so they are trusted; single quotes are still escaped defensively.
    """
    ts = cur_ts.replace("'", "''")
    uid = cur_uuid.replace("'", "''")
    columns = ",\n       ".join(_SELECT_COLUMNS)
    # Keyset predicate. On the first pull the cursor is the epoch sentinel with an
    # empty uuid; comparing the UUID-typed `uuid` column against '' is a type error
    # in HogQL/ClickHouse (HTTP 400), so fall back to a timestamp-only bound then.
    # `uuid` is cast to string for a well-defined tie-break comparison, and the
    # timestamp sentinel is parsed with toDateTime64 to accept the millisecond form.
    lo = f"toDateTime64('{ts}', 3)"
    if uid:
        keyset = (
            f"  AND ((timestamp > {lo}) "
            f"OR (timestamp = {lo} AND toString(uuid) > '{uid}'))\n"
        )
    else:
        keyset = f"  AND timestamp > {lo}\n"
    return (
        f"SELECT {columns}\n"
        "FROM events\n"
        "WHERE event = '$ai_generation'\n"
        f"  AND timestamp >= now() - INTERVAL {int(amount)} {unit}\n"
        f"{keyset}"
        "ORDER BY timestamp ASC, toString(uuid) ASC\n"
        f"LIMIT {int(limit)}"
    )


def load_cursor(path: str) -> tuple[str, str]:
    """Load the resume cursor; a missing/malformed/non-object file resets to epoch."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, ValueError, TypeError, OSError):
        return _EPOCH_TS, ""
    if not isinstance(data, dict):
        return _EPOCH_TS, ""
    return str(data.get("ts", _EPOCH_TS)), str(data.get("uuid", ""))


def save_cursor(path: str, ts: str, uuid: str) -> None:
    """Atomically persist the resume cursor."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"ts": ts, "uuid": uuid}, fh)
    os.replace(tmp, path)


def _http_query(url, headers, body, timeout=120):  # pragma: no cover - needs network
    import requests

    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
    if resp.status_code >= 400:
        # Surface PostHog's error body — it names the offending query/field,
        # which a bare raise_for_status() throws away.
        detail = resp.text[:1000]
        raise requests.exceptions.HTTPError(
            f"{resp.status_code} from PostHog query API: {detail}", response=resp
        )
    return resp.json()


def fetch_rows(
    host,
    project_id,
    api_key,
    window,
    start,
    *,
    limit=5000,
    query_fn=_http_query,
):
    """Fetch every ``$ai_generation`` row from ``start`` to now.

    ``start`` is a ``(ts, uuid)`` cursor tuple. Returns ``(rows, final_cursor)``
    where ``final_cursor`` is the ``(ts, uuid)`` of the last row fetched, or None
    if nothing was fetched. Pages with keyset pagination on ``(timestamp, uuid)``;
    because ``uuid`` is unique the composite keyset returns each row exactly once.

    This performs NO disk I/O. The caller advances the persisted cursor only after
    the corresponding output is durably written, so an interrupted or failed run
    never advances the cursor past data that was not committed to the output.
    """
    amount, unit = parse_window(window)
    url = build_query_url(host, project_id)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    cur_ts, cur_uuid = start
    rows_out: list[dict] = []

    while True:
        hogql = build_hogql(amount, unit, cur_ts, cur_uuid, limit)
        body = {
            "query": {"kind": "HogQLQuery", "query": hogql},
            "name": "orizon_scrub_export",
        }
        data = query_fn(url, headers, body)
        columns = data.get("columns") or []
        rows = data.get("results") or []
        if not rows:
            break
        for row in rows:
            rows_out.append(dict(zip(columns, row)))
        cur_ts = str(rows_out[-1].get("timestamp", cur_ts))
        cur_uuid = str(rows_out[-1].get("uuid", cur_uuid))
        if len(rows) < limit:
            break

    final = (cur_ts, cur_uuid) if rows_out else None
    return rows_out, final


def _parse_json_field(value):
    """PostHog JSON props often come back as JSON strings; return the parsed value."""
    if value is None or value == "":
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return None


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def convert_messages(messages, trace_id, base_index=0):
    """Convert PostHog messages to OpenAI-chat messages.

    PostHog content is a string or a list of typed parts (``text``/``image``/
    ``function``). ``function`` parts are lifted into an OpenAI ``tool_calls``
    array with ``arguments`` serialized to a JSON string (PostHog stores an
    object); image parts become ``image_url`` parts the scrubber will strip.
    """
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content")
        msg = {"role": role}
        tool_calls = []
        if isinstance(content, list):
            text_parts, media_parts = [], []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif ptype == "function" and isinstance(part.get("function"), dict):
                    fn = part["function"]
                    tool_calls.append(
                        {
                            "id": f"call_{trace_id}_{base_index + i}_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": json.dumps(
                                    fn.get("arguments", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
                elif ptype in ("image", "image_url"):
                    media_parts.append({"type": "image_url", "image_url": {"url": _image_url(part)}})
                else:
                    media_parts.append(part)
            if media_parts:
                parts = []
                joined = "\n".join(text_parts).strip()
                if joined:
                    parts.append({"type": "text", "text": joined})
                parts.extend(media_parts)
                msg["content"] = parts
            else:
                msg["content"] = "\n".join(text_parts)
        elif content is None:
            msg["content"] = ""
        else:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out.append(msg)
    return out


def _image_url(part):
    img = part.get("image")
    if isinstance(img, str):
        return img
    img_url = part.get("image_url")
    if isinstance(img_url, dict):
        return img_url.get("url", "")
    if isinstance(img_url, str):
        return img_url
    return ""


def stitch_trace(trace_id, rows):
    """Stitch one trace's $ai_generation rows into a single conversation.

    Rows are ordered by ``(timestamp, uuid)``; the last generation carries the
    fullest input history, so its ``$ai_input`` becomes the transcript and its
    ``$ai_output_choices`` the final assistant turn. A trace is marked incomplete
    when input/output is missing/unparseable or any generation errored (which
    includes AI-privacy-mode traces whose messages are stripped at source).
    """
    rows = sorted(rows, key=lambda r: (str(r.get("timestamp")), str(r.get("uuid"))))
    incomplete = any(_truthy(r.get("is_error")) for r in rows)
    last = rows[-1]

    input_msgs = _parse_json_field(last.get("input"))
    output_msgs = _parse_json_field(last.get("output_choices"))

    messages = []
    if isinstance(input_msgs, list):
        messages.extend(convert_messages(input_msgs, trace_id))
    else:
        incomplete = True
    if isinstance(output_msgs, list) and output_msgs:
        messages.extend(convert_messages(output_msgs, trace_id, base_index=len(messages)))
    else:
        incomplete = True
    if not messages:
        incomplete = True

    conv = {"trace_id": trace_id, "messages": messages}
    if last.get("model"):
        conv["model"] = last["model"]
    return conv, incomplete


def rows_to_conversations(rows):
    """Group flat rows by trace id (first-seen order) and stitch each into a conversation."""
    groups: OrderedDict[str, list] = OrderedDict()
    for row in rows:
        tid = row.get("trace_id") or row.get("uuid") or "unknown"
        groups.setdefault(str(tid), []).append(row)
    convs = []
    for tid, grp in groups.items():
        conv, incomplete = stitch_trace(tid, grp)
        convs.append((conv, incomplete))
    return convs


def pull(
    host,
    project_id,
    api_key,
    window,
    start,
    *,
    limit=5000,
    query_fn=_http_query,
):
    """Fetch + stitch. Returns ``(list[Rec], final_cursor)``.

    ``final_cursor`` is the position the caller should persist *after* the scrubbed
    output is durably committed.
    """
    rows, final = fetch_rows(
        host, project_id, api_key, window, start, limit=limit, query_fn=query_fn
    )
    recs = [
        Rec(id=conv["trace_id"], conv=conv, incomplete=incomplete)
        for conv, incomplete in rows_to_conversations(rows)
    ]
    return recs, final
