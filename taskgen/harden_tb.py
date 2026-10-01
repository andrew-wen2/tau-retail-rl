"""Harden the de-scripted training tasks toward tau3's mix of difficulty.

De-scripting (descript_tb.py) closed ~30% of the gap between the converted tasks and tau3's 114.
The rest is task DESIGN the converted set lacks. Four transforms, each keeping the correct end
state exactly defined and checked against the store DB; nothing is taken from tau3's own tasks:

  1. vague targets    exchange/modify targets become "the cheapest available one with display
                      LCD"-style descriptions, only when the DB shows the gold variant is the
                      UNIQUE item that fits (strict price order, available, not the current item)
  2. multi-step       two tasks of the same customer in the same split, on disjoint orders, become
                      one call whose gold is both action lists (new id "tbc_<a>_<b>")
  3. stated amounts   the customer asks what they will be refunded / pay; the exact amount goes in
                      communicate_info, which tau2 checks by substring, no LLM judge (single
                      tasks only; combined tasks never include a change of mind or an amount,
                      whose wording is ambiguous across two parts)
  5. change of mind   "the first time the agent asks you to confirm, keep the <item>": that item
                      leaves the gold action. The policy requires confirmation before every write,
                      so the condition always fires

Which eligible tasks get which transform is a hash of the task id, so the output is deterministic
and every transform keeps a share of tasks untouched (tau3 is a mix, not all-hard). The DB is
only read here; the VM validates every task as a tau2 Task and replays every gold action list.

Usage (repo root): python3 taskgen/harden_tb.py [db.json]
  -> tb500_hard.json, split_hard.json (train/val/all incl. combined), harden_report.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

TASKS = Path(__file__).resolve().parent.parent / "tasks"
DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else TASKS.parent / "data" / "retail" / "db.json"  # tau2 b7ea907
WRITES = {"cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
          "modify_pending_order_payment", "return_delivered_order_items",
          "exchange_delivered_order_items", "modify_user_address"}
SWAPS = {"exchange_delivered_order_items", "modify_pending_order_items"}
# share of ELIGIBLE tasks each transform is applied to
P_VAGUE, P_MIND, P_AMOUNT = 0.7, 0.5, 0.5
P_FORBID, P_DECOY = 0.5, 0.8
P_CONFLICT, P_ADDCONF, P_INFO, P_DEST, P_DEFADDR = 0.6, 0.7, 0.5, 0.6, 0.35


def roll(tid: str, salt: str, p: float) -> bool:
    return int(hashlib.sha256(f"{salt}:{tid}".encode()).hexdigest(), 16) % 1000 < p * 1000


def words(opts: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in opts.items())


class Store:
    def __init__(self, db: dict):
        self.db = db
        self.item = {}      # item_id -> (product_id, variant)
        for pid, p in db["products"].items():
            for iid, v in p["variants"].items():
                self.item[iid] = (pid, v)

    def affordable(self, a: dict, drop: int) -> bool:
        """After dropping item `drop` from a swap, can the payment method still cover the price
        difference? The policy rejects a gift card with too little balance (found by the VM
        gold replay: tb_0086)."""
        args = a["arguments"]
        if a["name"] not in SWAPS:
            return True
        keep = [i for i in range(len(args["item_ids"])) if i != drop]
        diff = sum(self.item[args["new_item_ids"][i]][1]["price"]
                   - self.order_item(args["order_id"], args["item_ids"][i])["price"] for i in keep)
        uid = self.db["orders"][args["order_id"]]["user_id"]
        pm = self.db["users"][uid]["payment_methods"].get(args.get("payment_method_id", ""), {})
        return not (pm.get("source") == "gift_card" and diff > pm.get("balance", 0))

    def order_item(self, oid: str, iid: str) -> dict:
        return next(i for i in self.db["orders"][oid]["items"] if i["item_id"] == iid)

    def vague(self, old_iid: str, new_iid: str) -> str | None:
        """A description that picks out exactly new_iid among the product's available variants
        other than the current item, or None."""
        pid, gold = self.item[new_iid]
        cands = [v for v in self.db["products"][pid]["variants"].values()
                 if v["available"] and v["item_id"] != old_iid]
        if gold not in cands:
            return None
        def only(pool, key, pick):
            best = pick(v["price"] for v in pool)
            return [v for v in pool if v["price"] == best] == [gold] if pool else False
        tries = [("the cheapest available option", cands, min),
                 ("the most expensive available option", cands, max)]
        for k, val in gold["options"].items():
            pool = [v for v in cands if v["options"].get(k) == val]
            tries += [(f"the cheapest available one with {k} {val}", pool, min),
                      (f"the most expensive available one with {k} {val}", pool, max)]
        for phrase, pool, pick in tries:
            if only(pool, None, pick):
                return phrase
        return None


def gold_writes(t: dict) -> list[dict]:
    return [a for a in t["evaluation_criteria"]["actions"] if a["name"] in WRITES]


def apply_vague(t: dict, st: Store, rep: dict) -> None:
    ins = t["user_scenario"]["instructions"]
    text = ins["reason_for_call"]
    for a in gold_writes(t):
        if a["name"] not in SWAPS:
            continue
        oid = a["arguments"]["order_id"]
        for old, new in zip(a["arguments"]["item_ids"], a["arguments"]["new_item_ids"]):
            phrase = st.vague(old, new)
            if not phrase or not roll(t["id"] + old, "vague", P_VAGUE):
                continue
            name = st.order_item(oid, old)["name"]
            cur = words(st.order_item(oid, old)["options"])
            pat = re.compile(re.escape(f"{name} ({cur}) to (") + r"[^)]*\)")
            if len(pat.findall(text)) != 1:
                continue            # cannot locate this target unambiguously in the text
            text = pat.sub(f"{name} ({cur}) to {phrase}", text)
            rep["vague_targets"] += 1
    ins["reason_for_call"] = text


NEEDS = {"return_delivered_order_items": "delivered", "exchange_delivered_order_items": "delivered",
         "cancel_pending_order": "pending", "modify_pending_order_items": "pending",
         "modify_pending_order_address": "pending", "modify_pending_order_payment": "pending"}


def apply_decoy(t: dict, st: Store, rep: dict) -> None:
    """Fix 3 (research round, TASTE's near-miss records): describe the order by ONE product only,
    when another of the user's orders has that product in a status the request's policy rules out.
    The policy then leaves exactly one eligible order, so the gold is still unique -- but the agent
    has to check status, not just match a name."""
    if not roll(t["id"], "decoy", P_DECOY):
        return
    ins = t["user_scenario"]["instructions"]
    for a in gold_writes(t):
        oid = a["arguments"]["order_id"]
        need = NEEDS.get(a["name"])
        if need is None:
            continue
        uid = st.db["orders"][oid]["user_id"]
        ids = a["arguments"].get("item_ids") or [i["item_id"] for i in st.db["orders"][oid]["items"]]
        name = st.order_item(oid, ids[0])["name"]
        having = [o for o in st.db["users"][uid]["orders"]
                  if any(i["name"] == name for i in st.db["orders"][o]["items"])]
        eligible = [o for o in having if st.db["orders"][o]["status"] == need]
        if len(having) < 2 or eligible != [oid]:
            continue
        full = order_desc_of(st, oid)
        if full not in ins["reason_for_call"]:
            continue
        ins["reason_for_call"] = ins["reason_for_call"].replace(full, f"your order with the {name}")
        rep["decoy"] = True
        return


def order_desc_of(st: Store, oid: str) -> str:
    o = st.db["orders"][oid]
    names = sorted({i["name"] for i in o["items"]})
    what = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"your {o['status']} order with the {what}"


def apply_mind(t: dict, st: Store, rep: dict) -> None:
    """Fix 5, revised (research round): changes of mind that SWITCH the correct action, the
    Trajectory2Task 'changing intent' type and tau3's dominant failure (the wrong set of writes).
    exchange -> return (refund to the original payment method), modify -> cancel. Only on tasks
    with one write, so 'the first time the agent asks you to confirm' is unambiguous. Falls back
    to dropping one item from a multi-item write (the first version, kept for variety)."""
    if rep.get("conflict") or rep.get("add_at_confirm") or not roll(t["id"], "mind", P_MIND):
        return
    ws = gold_writes(t)
    ins = t["user_scenario"]["instructions"]
    if len(ws) == 1 and ws[0]["name"] in SWAPS:
        a = ws[0]
        oid = a["arguments"]["order_id"]
        names = sorted({st.order_item(oid, i)["name"] for i in a["arguments"]["item_ids"]})
        what = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        if a["name"] == "exchange_delivered_order_items":
            orig = st.db["orders"][oid]["payment_history"][0]["payment_method_id"]
            a["name"] = "return_delivered_order_items"
            a["arguments"] = {"order_id": oid, "item_ids": a["arguments"]["item_ids"],
                              "payment_method_id": orig}
            ins["reason_for_call"] += (
                f" The first time the agent asks you to confirm the exchange, change your mind: you "
                f"no longer want replacements. Return the {what} instead, with the refund going to "
                f"the original payment method.")
        else:
            a["name"] = "cancel_pending_order"
            a["arguments"] = {"order_id": oid, "reason": "no longer needed"}
            ins["reason_for_call"] += (
                " The first time the agent asks you to confirm the change, change your mind: cancel "
                "the whole order instead, because you no longer need it.")
        rep["change_of_mind"] = "switch"
        return
    for a in ws:
        ids = a["arguments"].get("item_ids", [])
        if a["name"] not in SWAPS | {"return_delivered_order_items"} or len(ids) < 2:
            continue
        oid = a["arguments"]["order_id"]
        names = [st.order_item(oid, i)["name"] for i in ids]
        drop = len(ids) - 1
        if names.count(names[drop]) != 1:
            continue                # two items of the same product: "keep the X" is ambiguous
        if not st.affordable(a, drop):
            continue                # dropping it would leave a price difference the gift card can't cover
        a["arguments"]["item_ids"] = ids[:drop] + ids[drop + 1:]
        if "new_item_ids" in a["arguments"]:
            n = a["arguments"]["new_item_ids"]
            a["arguments"]["new_item_ids"] = n[:drop] + n[drop + 1:]
        ins["reason_for_call"] += (
            f" The first time the agent asks you to confirm, change your mind about the "
            f"{names[drop]}: you now want to keep it as it is. Go ahead with everything else.")
        rep["change_of_mind"] = "drop"
        return


def apply_forbidden(t: dict, st: Store, rep: dict) -> None:
    """Fix 2 (research round, Trajectory2Task 'infeasible intent', TASTE's forbidden demands):
    the customer first asks for something the policy does not allow, and only after the agent says
    no, asks for the task's real request. The gold is unchanged and always contains a write, so
    doing nothing never scores -- the free-reward trap of a pure refusal task."""
    if (rep.get("change_of_mind") or rep.get("conflict") or rep.get("add_at_confirm")
            or not roll(t["id"], "forbid", P_FORBID)):
        return
    ws = gold_writes(t)
    if len(ws) != 1:
        return
    a, ins = ws[0], t["user_scenario"]["instructions"]
    oid = a["arguments"]["order_id"]
    pre = None
    if a["name"] == "return_delivered_order_items":
        pre = ("Start by asking the agent to cancel this order entirely. Only if the agent tells you "
               "it cannot be cancelled, ask for the return below instead.")
    elif a["name"] == "cancel_pending_order":
        pre = ("At first, give your reason for cancelling as having found a better price "
               "elsewhere. Only if the agent says that reason is not accepted, give the reason "
               "below instead.")
    elif a["name"] == "exchange_delivered_order_items":
        item = st.order_item(oid, a["arguments"]["item_ids"][0])["name"]
        other = next(p["name"] for p in st.db["products"].values() if p["name"] != item)
        pre = (f"Start by asking to exchange the {item} for a {other} instead. Only if the agent "
               f"says an exchange has to stay within the same product, ask for the exchange below.")
    elif a["name"] == "modify_pending_order_items":
        item = st.order_item(oid, a["arguments"]["item_ids"][0])["name"]
        pre = (f"Start by asking to return the {item}. Only if the agent says the order has not "
               f"been delivered yet, ask for the change below instead.")
    if pre:
        ins["reason_for_call"] = pre + " " + ins["reason_for_call"]
        rep["forbidden_first"] = a["name"]


# --------------------------------------------------------------------------------------------
# Round 3 (Sept 26): devices read directly off tau3's inconsistently-solved tasks (A, B, C, D, G).


def names_str(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def apply_conflict(t: dict, st: Store, rep: dict) -> None:
    """A: a second request on the SAME delivered order, which policy makes impossible once the
    first is done ("Exchange or modify order tools can only be called once per order"; a return or
    exchange moves the order out of 'delivered'). The customer prefers the task's own request, so
    the gold is unchanged; an agent that acts on the other one first cannot recover. The other
    request is recorded so the VM can prove it blocks the gold (tau3 tasks 27, 98)."""
    ws = gold_writes(t)
    if len(ws) != 1 or ws[0]["name"] not in ("return_delivered_order_items",
                                             "exchange_delivered_order_items"):
        return
    if not roll(t["id"], "conflict", P_CONFLICT):
        return
    a = ws[0]
    oid = a["arguments"]["order_id"]
    items = st.db["orders"][oid]["items"]
    used = set(a["arguments"]["item_ids"])
    gold_names = {st.order_item(oid, i)["name"] for i in used}
    all_names = [i["name"] for i in items]
    cand = [i for i in items if i["item_id"] not in used and all_names.count(i["name"]) == 1
            and i["name"] not in gold_names]
    if not cand:
        return
    x = cand[0]
    ins = t["user_scenario"]["instructions"]
    gold_kind = "return" if a["name"].startswith("return") else "exchange"
    if gold_kind == "return":
        pid = st.item[x["item_id"]][0]
        alts = sorted((v for v in st.db["products"][pid]["variants"].values()
                       if v["available"] and v["item_id"] != x["item_id"]), key=lambda v: v["price"])
        if not alts or (len(alts) > 1 and alts[0]["price"] == alts[1]["price"]):
            return
        other = {"name": "exchange_delivered_order_items",
                 "arguments": {"order_id": oid, "item_ids": [x["item_id"]],
                               "new_item_ids": [alts[0]["item_id"]],
                               "payment_method_id": st.db["orders"][oid]["payment_history"][0]["payment_method_id"]}}
        ask = f"exchange the {x['name']} for the cheapest available option"
    else:
        other = {"name": "return_delivered_order_items",
                 "arguments": {"order_id": oid, "item_ids": [x["item_id"]],
                               "payment_method_id": st.db["orders"][oid]["payment_history"][0]["payment_method_id"]}}
        ask = f"return the {x['name']}, refunded to the original payment method"
    gn = names_str(sorted(gold_names))
    ins["reason_for_call"] += (
        f" From the same order, you also want to {ask}. Mention both requests at the same time. "
        f"If the agent says only one of them can be done, you prefer the {gold_kind} of the {gn}.")
    rep["conflict"] = other


def apply_add_at_confirm(t: dict, st: Store, rep: dict) -> None:
    """B: the customer holds back one item and adds it at confirmation ("do not say yes yet, you
    also want to return the X"); gold is the full return. With one return per order, an agent that
    executes before the addition cannot fix it (tau3 task 92)."""
    ws = gold_writes(t)
    if len(ws) != 1 or ws[0]["name"] != "return_delivered_order_items" or rep.get("conflict"):
        return
    ids = ws[0]["arguments"]["item_ids"]
    if len(ids) < 2 or not roll(t["id"], "addconf", P_ADDCONF):
        return
    oid = ws[0]["arguments"]["order_id"]
    names = [st.order_item(oid, i)["name"] for i in ids]
    last = names[-1]
    ins = t["user_scenario"]["instructions"]
    tail = f" {last};"
    if names.count(last) != 1 or ins["reason_for_call"].count(tail) != 1:
        return
    ins["reason_for_call"] = ins["reason_for_call"].replace(tail, "") + (
        f" When the agent asks you to confirm, do not say yes yet: tell them you also want to "
        f"return the {last} from the same order, and only then confirm.")
    rep["add_at_confirm"] = last


def apply_info_question(t: dict, st: Store, rep: dict) -> None:
    """C: an information request across the customer's orders -- a tracking number, or the total
    paid for every item of one product -- which the agent must look up and state. Graded by
    tau2's substring check on communicate_info, no judge (tau3 tasks 26, 76)."""
    if not roll(t["id"], "info", P_INFO):
        return
    uid = st.db["orders"][gold_writes(t)[0]["arguments"]["order_id"]]["user_id"]
    # only orders and products the task's own writes leave alone: otherwise the right answer
    # depends on whether the agent looks before or after its own change
    touched = {a["arguments"].get("order_id") for a in gold_writes(t)}
    touched_names = {i["name"] for o in touched if o for i in st.db["orders"][o]["items"]}
    orders = [st.db["orders"][o] for o in st.db["users"][uid]["orders"] if o not in touched]
    ins, crit = t["user_scenario"]["instructions"], t["evaluation_criteria"]
    if roll(t["id"], "info-kind", 0.5):
        for o in orders:
            if o["status"] not in ("delivered", "processed"):
                continue
            tr = [x for f in o["fulfillments"] for x in f["tracking_id"]]
            if len(tr) == 1:
                ins["reason_for_call"] += (f" You also want to know the tracking number of "
                                           f"{order_desc_of(st, o['order_id'])}.")
                crit["communicate_info"] = list(crit.get("communicate_info") or []) + [tr[0]]
                rep["info_question"] = "tracking"
                return
    per = {}
    for o in orders:
        for i in o["items"]:
            per.setdefault(i["name"], []).append((o["status"], i["price"]))
    for name, rows in sorted(per.items()):
        if name in touched_names:
            continue
        if len(rows) >= 2 and all(s != "cancelled" for s, _ in rows):
            total = f"{sum(p for _, p in rows):.2f}"
            ins["reason_for_call"] += (f" You also want to know the total price of all the "
                                       f"{name} items you have bought across your orders.")
            crit["communicate_info"] = list(crit.get("communicate_info") or []) + [total]
            rep["info_question"] = "total"
            return


def city_of(st: Store, oid: str) -> tuple[str, str]:
    a = st.db["orders"][oid]["address"]
    return a["city"], a["state"]


def apply_by_destination(t: dict, st: Store, rep: dict) -> None:
    """D (1): name the order by where it was shipped ("the order you sent to Texas"), when that
    city is unique among the customer's orders; the agent must search by attribute (tau3 26, 97)."""
    if rep.get("decoy") or not roll(t["id"], "dest", P_DEST):
        return
    ins = t["user_scenario"]["instructions"]
    for a in gold_writes(t):
        oid = a["arguments"]["order_id"]
        uid = st.db["orders"][oid]["user_id"]
        cities = [city_of(st, o) for o in st.db["users"][uid]["orders"]]
        c = city_of(st, oid)
        full = order_desc_of(st, oid)
        if cities.count(c) == 1 and len(set(cities)) > 1 and full in ins["reason_for_call"]:
            ins["reason_for_call"] = ins["reason_for_call"].replace(
                full, f"your {st.db['orders'][oid]['status']} order that was shipped to {c[0]}, {c[1]}")
            rep["by_destination"] = True
            return


def apply_default_address(t: dict, st: Store, rep: dict) -> None:
    """D (2): "change your default address to your <city> address -- you would rather not say it,
    it is on one of your orders". Adds modify_user_address, a tool the converted set never used
    (a tool-coverage gap), with the address read from that order (tau3 tasks 86, 97)."""
    if not roll(t["id"], "defaddr", P_DEFADDR):
        return
    uid = st.db["orders"][gold_writes(t)[0]["arguments"]["order_id"]]["user_id"]
    u = st.db["users"][uid]
    cities = [city_of(st, o) for o in u["orders"]]
    for o in u["orders"]:
        addr = st.db["orders"][o]["address"]
        c = city_of(st, o)
        if cities.count(c) == 1 and (c[0], c[1]) != (u["address"]["city"], u["address"]["state"]):
            t["evaluation_criteria"]["actions"].append({
                "action_id": f"{t['id']}_addr", "requestor": "assistant", "name": "modify_user_address",
                "arguments": {"user_id": uid, **{k: addr[k] for k in
                              ("address1", "address2", "city", "state", "country", "zip")}},
                "info": None, "compare_args": None})
            t["user_scenario"]["instructions"]["reason_for_call"] += (
                f" You also want to change your default address to your {c[0]} address. You would "
                f"rather not say the street address out loud; it is on one of your orders.")
            rep["default_address"] = c[0]
            return


def apply_twin_items(t: dict, st: Store, rep: dict) -> None:
    """G: the order holds two variants of one product and the customer names only the product,
    giving the distinguishing option only if asked ("if the agent asks which laptop, it is the
    15-inch") -- the agent must ask before acting (tau3 task 93: solved 1 in 8)."""
    ins = t["user_scenario"]["instructions"]
    for a in gold_writes(t):
        if a["name"] not in SWAPS:
            continue
        oid = a["arguments"]["order_id"]
        items = st.db["orders"][oid]["items"]
        for iid in a["arguments"]["item_ids"]:
            me = st.order_item(oid, iid)
            twins = [i for i in items if i["name"] == me["name"] and i["item_id"] != iid]
            if len(twins) != 1:
                continue
            diff = [k for k, v in me["options"].items() if twins[0]["options"].get(k) != v]
            if not diff:
                continue
            head = f"{me['name']} ({words(me['options'])}) to "
            if ins["reason_for_call"].count(head) != 1:
                continue
            ins["reason_for_call"] = ins["reason_for_call"].replace(head, f"{me['name']} to ")
            ins["reason_for_call"] += (f" If the agent asks which {me['name']} you mean, it is the "
                                       f"one with {diff[0]} {me['options'][diff[0]]}.")
            rep["twin_items"] = True
            return


def amount(a: dict, st: Store) -> tuple[str, float] | None:
    args = a["arguments"]
    if a["name"] == "cancel_pending_order":
        return "refunded", sum(i["price"] for i in st.db["orders"][args["order_id"]]["items"])
    if a["name"] == "return_delivered_order_items":
        return "refunded", sum(st.order_item(args["order_id"], i)["price"] for i in args["item_ids"])
    if a["name"] in SWAPS:
        d = sum(st.item[n][1]["price"] - st.order_item(args["order_id"], o)["price"]
                for o, n in zip(args["item_ids"], args["new_item_ids"]))
        return ("charged" if d > 0 else "refunded"), abs(d)
    return None


def apply_amount(t: dict, st: Store, rep: dict) -> None:
    if rep.get("conflict") or not roll(t["id"], "amount", P_AMOUNT):
        return
    for a in gold_writes(t):
        r = amount(a, st)
        if r is None or r[1] < 0.01:
            continue
        what, amt = r
        s = f"{amt:.2f}"
        t["evaluation_criteria"]["communicate_info"] = [s]
        verb = "be refunded" if what == "refunded" else "have to pay"
        t["user_scenario"]["instructions"]["reason_for_call"] += (
            f" Before you confirm, ask the agent exactly how much you will {verb} for this, and "
            f"make sure they tell you the amount.")
        rep["stated_amount"] = s
        return


def combine(a: dict, b: dict) -> dict:
    t = json.loads(json.dumps(a))
    t["id"] = f"tbc_{a['id'][3:]}_{b['id'][3:]}"
    ia, ib = a["user_scenario"]["instructions"], b["user_scenario"]["instructions"]
    t["user_scenario"]["instructions"]["reason_for_call"] = (
        ia["reason_for_call"] + " Also: " + ib["reason_for_call"])
    t["evaluation_criteria"]["actions"] = (a["evaluation_criteria"]["actions"]
                                           + b["evaluation_criteria"]["actions"])
    t["evaluation_criteria"]["communicate_info"] = []
    return t


def main() -> int:
    st = Store(json.loads(DB_PATH.read_text()))
    split = json.loads((TASKS / "split_tb500.json").read_text())
    val = set(split["val"])
    base = {t["id"]: t for t in json.loads((TASKS / "tb500_descripted.json").read_text())
            if t["id"] in set(split["all"])}
    report, out, vague_only = {}, {}, {}
    for tid, t0 in base.items():
        t = json.loads(json.dumps(t0))
        rep = {"vague_targets": 0, "change_of_mind": False, "stated_amount": None,
               "forbidden_first": None, "decoy": False, "conflict": None, "add_at_confirm": None,
               "info_question": None, "by_destination": False, "default_address": None,
               "twin_items": False}
        v = json.loads(json.dumps(t0))
        apply_vague(v, st, {"vague_targets": 0})
        vague_only[tid] = v          # combined tasks are built from these: no mind, no amount
        apply_decoy(t, st, rep)      # text only: needs the de-scripted order description intact
        apply_by_destination(t, st, rep)   # D1, where no decoy
        apply_vague(t, st, rep)      # before the switch: the opening request stays vague
        apply_twin_items(t, st, rep)       # G, after vague (needs "Name (opts) to")
        apply_conflict(t, st, rep)         # A
        apply_add_at_confirm(t, st, rep)   # B
        apply_mind(t, st, rep)       # changes the gold; before amounts, which follow what happens
        apply_forbidden(t, st, rep)  # only where no change of mind
        apply_amount(t, st, rep)
        apply_info_question(t, st, rep)    # C
        apply_default_address(t, st, rep)  # D2: appends a write last
        out[tid], report[tid] = t, rep
    # 2. combine same-customer tasks within a split, disjoint orders, each task used at most once
    def uid(t):
        return st.db["orders"][gold_writes(t)[0]["arguments"]["order_id"]]["user_id"]
    groups: dict[tuple, list[str]] = {}
    for tid in sorted(out):
        groups.setdefault((uid(out[tid]), tid in val), []).append(tid)
    combos = {}
    for (u, is_val), ids in groups.items():
        used = set()
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if a in used or b in used:
                    continue
                oa = {x["arguments"].get("order_id") for x in gold_writes(vague_only[a])}
                ob = {x["arguments"].get("order_id") for x in gold_writes(vague_only[b])}
                if oa & ob:
                    continue
                c = combine(vague_only[a], vague_only[b])
                rep = {"combined_from": [a, b], "vague_targets": 0, "change_of_mind": False,
                       "stated_amount": None, "forbidden_first": None, "decoy": False,
                       "conflict": None, "add_at_confirm": None, "info_question": None,
                       "by_destination": False, "default_address": None, "twin_items": False}
                combos[c["id"]] = (c, is_val)
                report[c["id"]] = rep
                used |= {a, b}
    for cid, (c, _) in combos.items():
        out[cid] = c
    new_split = {"train": sorted([t for t in split["train"] if t in out]
                                 + [c for c, (_, v) in combos.items() if not v]),
                 "val": sorted([t for t in split["val"] if t in out]
                               + [c for c, (_, v) in combos.items() if v])}
    new_split["all"] = sorted(new_split["train"] + new_split["val"])
    assert not set(new_split["train"]) & set(new_split["val"])
    (TASKS / "tb500_hard.json").write_text(json.dumps(list(out.values()), indent=1))
    (TASKS / "split_hard.json").write_text(json.dumps(new_split, indent=1))
    (TASKS / "harden_report.json").write_text(json.dumps(report, indent=1))
    singles = [r for k, r in report.items() if "combined_from" not in r]
    print(f"tasks {len(out)} ({len(base)} single + {len(combos)} combined) | train "
          f"{len(new_split['train'])}, val {len(new_split['val'])}")
    print(f"vague targets {sum(r['vague_targets'] for r in singles)} in "
          f"{sum(r['vague_targets'] > 0 for r in singles)} tasks | change of mind "
          f"{sum(bool(r['change_of_mind']) for r in singles)} | stated amount "
          f"{sum(r['stated_amount'] is not None for r in report.values())}")
    print(f"switching change of mind {sum(r['change_of_mind'] == 'switch' for r in singles)}, "
          f"dropping {sum(r['change_of_mind'] == 'drop' for r in singles)} | forbidden-first "
          f"{sum(bool(r['forbidden_first']) for r in singles)} | decoy orders "
          f"{sum(r['decoy'] for r in singles)}")
    print(f"A conflict {sum(bool(r['conflict']) for r in singles)} | B add-at-confirm "
          f"{sum(bool(r['add_at_confirm']) for r in singles)} | C info question "
          f"{sum(bool(r['info_question']) for r in singles)} | D by-destination "
          f"{sum(r['by_destination'] for r in singles)}, default address "
          f"{sum(bool(r['default_address']) for r in singles)} | G twin items "
          f"{sum(r['twin_items'] for r in singles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
