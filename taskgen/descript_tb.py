"""De-script the converted tau-bench retail tasks into tau3's style.

The converted tasks hand the customer a script: every one states the exact order id, 79% the
exact payment-method id, 57% the exact option dicts, as one flat string. tau3's 114 tasks
state a goal instead (11% give an order id, 0% a payment id) in structured instructions, and
with a competent customer the base agent solves the converted set ~0.90 against tau3's ~0.63.
This rewrites WHAT THE CUSTOMER IS TOLD, never the gold actions, so
every task's correct end state -- and so its reward -- is unchanged:

  * order ids      -> "your <status> order with the <product names>", only when that description
                      matches exactly one of the user's orders; otherwise the id is kept
  * payment ids    -> "your gift card" / "your PayPal account" / "your <brand> credit card ending
                      in <last four>", only when unique among the user's methods of that kind
  * option dicts   -> plain words ("color gold, band material metal"); same information
  * "(same as #W…)" on address changes is dropped; the full address is already stated
  * format         -> tau3's StructuredUserInstructions (reason_for_call / known_info /
                      unknown_info / task_instructions)
  * identity       -> customers who were given an email forget it in half the tasks (chosen by
                      a hash of the task id) and are given their zip instead, so the agent must
                      authenticate by name + zip; tau3 has customers forget or withhold in 75%

Deterministic, no LLM. Usage (repo root):
    python3 taskgen/descript_tb.py [path/to/retail/db.json]
writes tb500_descripted.json (same task ids) and descript_report.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

TASKS = Path(__file__).resolve().parent.parent / "tasks"
DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else TASKS.parent / "data" / "retail" / "db.json"  # tau2 b7ea907

ORDER = re.compile(r"#W\d{7}")
PAY = re.compile(r"\b(gift_card|credit_card|paypal)_\d+\b")
OPTS = re.compile(r"\{('[^']+': '[^']*'(?:, )?)+\}")
HEAD = re.compile(r"^Your name is (?P<name>.+?) and your (?:email is (?P<email>\S+?)|zip code is "
                  r"(?P<zip>\d{5}))\. You are (?P<traits>[^.]*)\.\s*(?P<body>.*)$", re.S)


def words(opts: str) -> str:
    d = eval(opts, {"__builtins__": {}})  # noqa: S307 -- literal dicts from our own task file
    return ", ".join(f"{k} {v}" for k, v in d.items() if k != "order_id")


def order_desc(db: dict, oid: str) -> str:
    o = db["orders"][oid]
    names = sorted({i["name"] for i in o["items"]})
    what = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"your {o['status']} order with the {what}"


def unique_order(db: dict, oid: str) -> bool:
    uid = db["orders"][oid]["user_id"]
    mine = order_desc(db, oid)
    return sum(order_desc(db, o) == mine for o in db["users"][uid]["orders"]) == 1


def pay_desc(db: dict, uid: str, pid: str) -> str | None:
    pms = db["users"][uid]["payment_methods"]
    pm = pms.get(pid)
    if pm is None:
        return None
    same = [p for p in pms.values() if p["source"] == pm["source"]]
    if pm["source"] == "gift_card":
        return "your gift card" if len(same) == 1 else None
    if pm["source"] == "paypal":
        return "your PayPal account" if len(same) == 1 else None
    if sum(p.get("last_four") == pm.get("last_four") for p in same) == 1:
        return f"your {pm.get('brand', 'credit')} card ending in {pm['last_four']}"
    return None


def descript(task: dict, db: dict) -> tuple[dict, dict]:
    s = task["user_scenario"]["instructions"]
    m = HEAD.match(s)
    if not m:
        raise ValueError(f"{task['id']}: unrecognised instruction head")
    body = m["body"].strip()
    oids = ORDER.findall(body)
    uid = db["orders"][oids[0]]["user_id"]
    rep = {"orders_described": 0, "orders_kept": 0, "payments_described": 0, "payments_kept": 0}

    body = re.sub(r"\s*\(same as #W\d{7}\)", "", body)
    # address dicts carry an order_id key; words() drops it
    body = OPTS.sub(lambda x: "(" + words(x.group(0)) + ")", body)

    def sub_order(x: re.Match) -> str:
        oid = x.group(0)
        if unique_order(db, oid):
            rep["orders_described"] += 1
            return order_desc(db, oid)
        rep["orders_kept"] += 1
        return oid
    body = ORDER.sub(sub_order, body)
    body = re.sub(r"\bCancel order your\b", "Cancel your", body)
    body = re.sub(r"\bReturn your\b", "Return items from your", body)

    def sub_pay(x: re.Match) -> str:
        d = pay_desc(db, uid, x.group(0))
        if d:
            rep["payments_described"] += 1
            return d
        rep["payments_kept"] += 1
        return x.group(0)
    body = PAY.sub(sub_pay, body)
    body = re.sub(r"\bvia your\b", "using your", body)

    zip_ = db["users"][uid]["address"]["zip"]
    forget = m["email"] and int(hashlib.sha256(task["id"].encode()).hexdigest(), 16) % 2 == 0
    if m["email"] and not forget:
        known, unknown = f"You are {m['name']}. Your email is {m['email']}.", ""
    else:
        known = f"You are {m['name']} and your zip code is {m['zip'] or zip_}."
        unknown = "You do not remember your email address."
    rep["forgets_email"] = bool(forget)

    out = json.loads(json.dumps(task))
    out["user_scenario"] = {"persona": None, "instructions": {
        "domain": "retail",
        "reason_for_call": body,
        "known_info": known,
        "unknown_info": unknown,
        "task_instructions": f"You are {m['traits']}.",
    }}
    # the correct outcome must not have moved
    assert out["evaluation_criteria"] == task["evaluation_criteria"]
    return out, rep


def main() -> int:
    db = json.loads(DB_PATH.read_text())
    tasks = json.loads((TASKS / "tb500_retail.json").read_text())
    out, report = [], {}
    for t in tasks:
        nt, rep = descript(t, db)
        out.append(nt)
        report[t["id"]] = rep
    (TASKS / "tb500_descripted.json").write_text(json.dumps(out, indent=1))
    (TASKS / "descript_report.json").write_text(json.dumps(report, indent=1))
    n = len(report)
    tot = {k: sum(r[k] for r in report.values()) for k in next(iter(report.values()))}
    print(f"{n} tasks | orders described {tot['orders_described']}, kept {tot['orders_kept']} | "
          f"payments described {tot['payments_described']}, kept {tot['payments_kept']} | "
          f"forget email {tot['forgets_email']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
