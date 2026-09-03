"""Generate realistic PII-laden banking agent conversations (OpenAI-chat format).

Used to exercise the scrubber against high-PII, real-shaped data. Content is
synthetic but modeled on Meridian Trust Bank support/underwriting transcripts,
with names, emails, phones, addresses, SSNs, Luhn-valid cards, account numbers,
DOBs, and the occasional API key and URL, spread across message text AND
tool-call arguments/results.

Importable (``generate(n, seed)`` returns a deterministic list) and runnable:

    python tests/gen_pii.py 1000 out.jsonl        # writes 1000 conversations

The committed sample ``tests/fixtures/pii_sample.jsonl`` is ``generate(30, 42)``.
"""
from __future__ import annotations

import json
import random
import sys

FIRST = ["Sarah", "Michael", "Priya", "David", "Emma", "Carlos", "Aisha", "Liam",
         "Sofia", "Noah", "Yuki", "Mateo", "Fatima", "Oliver", "Chen", "Ingrid",
         "Omar", "Elena", "Jamal", "Nina", "Tomas", "Grace", "Ravi", "Lena"]
LAST = ["Johnson", "Chen", "Patel", "Okafor", "Garcia", "Nguyen", "Smith", "Rossi",
        "Andersson", "Kim", "Haddad", "Williams", "Silva", "Novak", "Ivanova",
        "Reed", "Mbeki", "Larsson", "Costa", "Khan", "Meyer", "Dubois"]
STREETS = ["Maple Ave", "Birch Ln", "Kungsgatan", "Elm St", "Oak Drive", "Vasagatan",
           "Harbor Rd", "Cedar Ct", "Sveavagen", "Park Blvd"]
CITIES = [("Stockholm", "AB", "100 04"), ("Boston", "MA", "02108"),
          ("Austin", "TX", "78701"), ("Berlin", "BE", "10115"),
          ("Toronto", "ON", "M5H 2N2"), ("Denver", "CO", "80202")]
DOMAINS = ["gmail.com", "outlook.com", "proton.me", "acmemail.com", "corp.example",
           "fastmail.com"]
SCENARIOS = ["fraud_review", "card_dispute", "wire_transfer", "wealth_portfolio",
             "loan_underwriting", "account_reset"]
CHANNELS = ["ivr_bot", "banker_console", "support_chat"]


def luhn_ok(num: str) -> bool:
    s, par = 0, len(num) % 2
    for i, ch in enumerate(num):
        d = int(ch)
        if i % 2 == par:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return s % 10 == 0


def make_card() -> str:
    """A Luhn-valid 16-digit test card, grouped in fours."""
    body = "4" + "".join(random.choice("0123456789") for _ in range(14))
    for d in "0123456789":
        if luhn_ok(body + d):
            n = body + d
            return f"{n[0:4]} {n[4:8]} {n[8:12]} {n[12:16]}"
    return "4242 4242 4242 4242"


def _person() -> dict:
    f, l = random.choice(FIRST), random.choice(LAST)
    return {
        "name": f"{f} {l}",
        "email": f"{f.lower()}.{l.lower()}@{random.choice(DOMAINS)}",
        "phone": f"+1 ({random.randint(200,989)}) {random.randint(200,989)}-{random.randint(1000,9999)}",
        "ssn": f"{random.randint(100,899)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
        "card": make_card(),
        "acct": f"MTB-ACCT-{random.randint(10**7, 10**8-1)}",
        "dob": f"19{random.randint(50,99)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "addr": (lambda c: f"{random.randint(1,9999)} {random.choice(STREETS)}, {c[0]}, {c[1]} {c[2]}")(random.choice(CITIES)),
    }


def make_conversation(i: int) -> dict:
    """Build one conversation. Deterministic given the module RNG state."""
    p = _person()
    scen = random.choice(SCENARIOS)
    tid = f"mtb-{i:05d}-{random.randint(10**7,10**8-1):x}"
    msgs = [{"role": "system", "content": (
        "You are Athena, a support and underwriting assistant for Meridian Trust Bank. "
        "Verify identity before discussing account details and follow bank policy.")}]

    if scen == "card_dispute":
        msgs.append({"role": "user", "content":
            f"Hi, I'm {p['name']}. I want to dispute a charge on my card {p['card']}. "
            f"You can reach me at {p['email']} or {p['phone']}."})
        msgs.append({"role": "assistant", "content": f"Thanks {p['name']}. Let me pull up that card.",
            "tool_calls": [{"id": f"call_{i}_0", "type": "function", "function": {
                "name": "lookup_card", "arguments": json.dumps({"card": p["card"], "email": p["email"]})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}_0", "name": "lookup_card",
            "content": json.dumps({"account_number": p["acct"], "holder": p["name"], "status": "active"})})
        msgs.append({"role": "assistant", "content":
            f"I found account {p['acct']} for {p['name']}. I've opened a dispute and will email {p['email']}."})
    elif scen == "wire_transfer":
        msgs.append({"role": "user", "content":
            f"This is {p['name']} (DOB {p['dob']}), SSN {p['ssn']}. Please wire from {p['acct']} "
            f"to my landlord at 55 {random.choice(STREETS)}. My number is {p['phone']}."})
        msgs.append({"role": "assistant", "content":
            f"Confirming identity for {p['name']}, DOB {p['dob']}. I'll set up the transfer from {p['acct']}."})
    elif scen == "loan_underwriting":
        msgs.append({"role": "user", "content":
            f"Applicant {p['name']}, email {p['email']}, lives at {p['addr']}. "
            f"SSN {p['ssn']}, DOB {p['dob']}. Requesting a mortgage pre-approval."})
        msgs.append({"role": "assistant", "content": "Running the risk model now.",
            "tool_calls": [{"id": f"call_{i}_0", "type": "function", "function": {
                "name": "run_risk_model", "arguments": json.dumps(
                    {"name": p["name"], "ssn": p["ssn"], "address": p["addr"]})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}_0", "name": "run_risk_model",
            "content": json.dumps({"score": random.randint(580, 820), "account": p["acct"]})})
        msgs.append({"role": "assistant", "content":
            f"{p['name']}, based on the model I can pre-approve. Confirmation goes to {p['email']}."})
    elif scen == "account_reset":
        msgs.append({"role": "user", "content":
            f"I'm locked out. Name {p['name']}, email {p['email']}, phone {p['phone']}. "
            f"Reset link please. Internal note: temp key sk-live-{''.join(random.choice('abcdef0123456789') for _ in range(24))}."})
        msgs.append({"role": "assistant", "content":
            f"Sent a reset link to {p['email']} and an SMS to {p['phone']}, {p['name']}."})
    elif scen == "fraud_review":
        msgs.append({"role": "user", "content":
            f"Reviewing possible fraud for {p['name']}, card {p['card']}, account {p['acct']}. "
            f"Customer address on file: {p['addr']}."})
        msgs.append({"role": "assistant", "content":
            f"Flagged card {p['card']} on {p['acct']}. I'll notify {p['name']}."})
    else:  # wealth_portfolio
        msgs.append({"role": "user", "content":
            f"Portfolio review for {p['name']} ({p['email']}). Reference doc "
            f"https://portal.meridiantrust.example/clients/{p['acct']}. Contact {p['phone']}."})
        msgs.append({"role": "assistant", "content":
            f"Reviewed {p['name']}'s portfolio. Summary sent to {p['email']}."})

    return {"id": f"conv-{i:05d}", "trace_id": tid, "model": "gpt-4o", "scenario": scen,
            "channel": random.choice(CHANNELS), "institution": "Meridian Trust Bank",
            "messages": msgs}


def generate(n: int, seed: int = 42) -> list[dict]:
    """Return ``n`` conversations, reproducibly for a given ``seed``."""
    random.seed(seed)
    return [make_conversation(i) for i in range(n)]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    n = int(argv[0]) if argv else 1000
    out = argv[1] if len(argv) > 1 else "pii_traces.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for conv in generate(n):
            fh.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"wrote {n} conversations to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
