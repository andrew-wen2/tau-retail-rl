"""Targeted training tasks for the failure patterns p4-b's rollouts showed (Sept 26).

Four families, one per learnable failure class in the mixed-group failures of p4-b steps 12-21
(transcript read). Every task is built answer-first from the store DB
(SENTINEL / tau-bench-synthetic style): pick the entities, construct the exact gold writes,
then write the customer request around them, so the correct end state is exact and unique.

  items_addr   one pending order needs an item change AND a shipping-address change. The agent
               kept doing the item change and dropping the address, claiming nothing can change
               after an item modify -- false in tau2 (address/payment accept 'pending (item
               modified)'). Both requests are stated in the first message.
  decoy_first  two orders in the SAME status hold the named product and the decoy comes first
               in the user's order list; only an attribute tells them apart. The agent kept
               stopping at the first order with a matching product name.
  keep_options change only some options of an item; the rest must stay as they are. The agent
               kept picking a variant that also changed options nobody mentioned.
  pairing      several items changed in one call, requested in a different order from the order
               record, or one of two variants of the same product in one order. The agent kept
               mis-pairing item_ids with new_item_ids, and changing the wrong twin.

Each family has three levels; the trainer promotes a family to the next level when its recent
success passes a threshold (RLVE-style adaptive difficulty). Ids: tt_<family>_L<level>_<n>.

Usage (repo root): python3 taskgen/gen_targeted.py [db.json] -> tb_targeted.json, targeted_report.json
Orders and users that any val task touches are excluded here; the VM drops overlaps with the
held-out 114 and checks gold replay + order-invariance of multi-item swaps (build_v5.py).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import harden_tb as H

PER_FAMILY = 180
PERSONAS = ["You are logical, busy, outgoing, independent, pessimistic.",
            "You are polite, patient, detail-oriented.", "You are impatient and direct.",
            "You are friendly, chatty, a little disorganized.", "You are cautious and organized.",
            "You are shy, flexible, curious.", "You are rigid, confident, messy."]
HONEST = (" Never state an order id or item id you were not given; refer to your orders by the "
          "items in them.")


def pm_desc(pm: dict) -> str:
    if pm["source"] == "gift_card":
        return "gift card"
    if pm["source"] == "paypal":
        return "PayPal account"
    return f"{pm['brand']} card ending in {pm['last_four']}"


class Gen:
    def __init__(self, db: dict, exclude_orders: set[str], exclude_users: set[str]):
        self.st = H.Store(db)
        self.db = db
        self.ex_o, self.ex_u = exclude_orders, exclude_users
        self.out: list[dict] = []
        self.report: dict[str, dict] = {}
        self.seen: set[str] = set()

    # ---------------------------------------------------------------- helpers
    def orders(self, status: str | None = None):
        for oid, o in self.db["orders"].items():
            if oid in self.ex_o or o["user_id"] in self.ex_u:
                continue
            if status is None or o["status"] == status:
                yield oid, o

    def pay(self, o: dict) -> dict | None:
        """The order's original payment method, if the user still has it."""
        pid = o["payment_history"][0]["payment_method_id"]
        return self.db["users"][o["user_id"]]["payment_methods"].get(pid)

    def alts(self, iid: str) -> list[dict]:
        pid, _ = self.st.item[iid]
        return [v for v in self.db["products"][pid]["variants"].values()
                if v["available"] and v["item_id"] != iid]

    def unique_by_options(self, pid: str, opts: dict) -> dict | None:
        m = [v for v in self.db["products"][pid]["variants"].values()
             if v["available"] and v["options"] == opts]
        return m[0] if len(m) == 1 else None

    def only_change(self, iid: str, k: str, v: str) -> dict | None:
        """The one available variant equal to iid's options except k=v."""
        pid, cur = self.st.item[iid]
        want = dict(cur["options"], **{k: v})
        return self.unique_by_options(pid, want)

    def affordable(self, pm: dict, diff: float) -> bool:
        return not (pm["source"] == "gift_card" and diff > pm.get("balance", 0))

    def swap_action(self, tid: str, o: dict, olds: list[str], news: list[str], pm: dict) -> dict:
        name = ("exchange_delivered_order_items" if o["status"] == "delivered"
                else "modify_pending_order_items")
        return {"action_id": f"{tid}_0", "requestor": "assistant", "name": name,
                "arguments": {"order_id": o["order_id"], "item_ids": olds, "new_item_ids": news,
                              "payment_method_id": pm["id"]},
                "info": None, "compare_args": None}

    def diff(self, o: dict, olds: list[str], news: list[str]) -> float:
        return sum(self.st.item[n][1]["price"] - self.st.order_item(o["order_id"], a)["price"]
                   for a, n in zip(olds, news))

    def emit(self, family: str, level: int, uid: str, text: str, actions: list[dict],
             rng: random.Random, **rep) -> None:
        key = json.dumps([text, actions], sort_keys=True)
        if key in self.seen:
            return
        self.seen.add(key)
        n = sum(1 for t in self.out if t["id"].startswith(f"tt_{family}_L{level}_"))
        tid = f"tt_{family}_L{level}_{n:03d}"
        for i, a in enumerate(actions):
            a["action_id"] = f"{tid}_{i}"
        u = self.db["users"][uid]
        self.out.append({
            "id": tid,
            "description": {"purpose": f"targeted:{family}:L{level}", "relevant_policies": None,
                            "notes": "gen_targeted.py (Sept 26), answer-first from the store DB."},
            "user_scenario": {"persona": None, "instructions": {
                "domain": "retail", "reason_for_call": text,
                "known_info": f"You are {u['name']['first_name']} {u['name']['last_name']} and "
                              f"your zip code is {u['address']['zip']}.",
                "unknown_info": "You do not remember your email address.",
                "task_instructions": rng.choice(PERSONAS) + HONEST}},
            "initial_state": None,
            "evaluation_criteria": {"actions": actions, "communicate_info": [],
                                    "nl_assertions": None, "reward_basis": ["DB", "COMMUNICATE"]}})
        self.report[tid] = {"family": family, "level": level, **rep}

    def count(self, family: str) -> int:
        return sum(1 for t in self.out if t["id"].startswith(f"tt_{family}_"))

    # ---------------------------------------------------------------- families
    def items_addr(self, rng: random.Random) -> None:
        cands = list(self.orders("pending"))
        rng.shuffle(cands)
        for oid, o in cands:
            if self.count("items_addr") >= PER_FAMILY:
                return
            pm = self.pay(o)
            if pm is None:
                continue
            level = 1 + len(self.out) % 3
            uid = o["user_id"]
            # the new shipping address: L1 spelled out (another user's), L2+ "the address of
            # your order shipped to <city>" -- unique among this user's orders, not this order's
            cities = [H.city_of(self.st, x) for x in self.db["users"][uid]["orders"]]
            mine = H.city_of(self.st, oid)
            by_city = [x for x in self.db["users"][uid]["orders"]
                       if cities.count(H.city_of(self.st, x)) == 1 and H.city_of(self.st, x) != mine]
            if level >= 2 and not by_city:
                level = 1
            if level == 1:
                other = rng.choice([u for u in self.db["users"].values() if u["user_id"] != uid])
                addr = other["address"]
                if (addr["city"], addr["state"]) == mine:
                    continue
                a_txt = (f"{addr['address1']}, {addr['address2']}, {addr['city']}, "
                         f"{addr['state']} {addr['zip']}, {addr['country']}")
                addr_req = f"change its shipping address to {a_txt}"
            else:
                src = self.db["orders"][by_city[0]]["address"]
                addr = src
                c = H.city_of(self.st, by_city[0])
                addr_req = (f"change its shipping address to the address of your order that was "
                            f"shipped to {c[0]}, {c[1]} (you would rather not say the street "
                            f"address; it is on that order)")
            # the item change(s)
            items = [i for i in o["items"] if self.alts(i["item_id"])]
            names = [i["name"] for i in o["items"]]
            items = [i for i in items if names.count(i["name"]) == 1]
            if not items:
                continue
            rng.shuffle(items)
            olds, news, parts = [], [], []
            for it in items[: (2 if level == 3 else 1)]:
                cur = it["options"]
                pick = None
                if level >= 2:
                    for k in cur:
                        for v in {x["options"][k] for x in self.alts(it["item_id"])}:
                            if v != cur[k] and (g := self.only_change(it["item_id"], k, v)):
                                pick = (g, f"change only the {k} of the {it['name']} to {v} "
                                           f"and keep everything else about it the same")
                                break
                        if pick:
                            break
                if pick is None:
                    g = rng.choice(self.alts(it["item_id"]))
                    if self.unique_by_options(self.st.item[g["item_id"]][0], g["options"]) is None:
                        continue
                    pick = (g, f"change the {it['name']} ({H.words(cur)}) to the one with "
                               f"{H.words(g['options'])}")
                olds.append(it["item_id"])
                news.append(pick[0]["item_id"])
                parts.append(pick[1])
            if not olds or not self.affordable(pm, self.diff(o, olds, news)):
                continue
            what = H.names_str(sorted({i["name"] for i in o["items"]}))
            text = (f"For your pending order with the {what}, you want two things done: "
                    f"{'; and '.join(parts)}, paying any price difference with your "
                    f"{pm_desc(pm)}; and also {addr_req}. Mention both requests in your first "
                    f"message. If the agent says the address can no longer be changed after the "
                    f"item change, insist politely that you still want it changed.")
            tid = "x"
            acts = [self.swap_action(tid, o, olds, news, pm),
                    {"action_id": "", "requestor": "assistant", "name": "modify_pending_order_address",
                     "arguments": {"order_id": oid, **{k: addr[k] for k in (
                         "address1", "address2", "city", "state", "country", "zip")}},
                     "info": None, "compare_args": None}]
            self.emit("items_addr", level, uid, text, acts, rng, n_items=len(olds))

    def decoy_first(self, rng: random.Random) -> None:
        users = [u for u in self.db["users"].values() if u["user_id"] not in self.ex_u]
        rng.shuffle(users)
        for u in users:
            if self.count("decoy_first") >= PER_FAMILY:
                return
            ords = [x for x in u["orders"] if x not in self.ex_o]
            for gi, gid in enumerate(ords):
                g = self.db["orders"][gid]
                if g["status"] not in ("pending", "delivered"):
                    continue
                for it in g["items"]:
                    decoys = [self.db["orders"][d] for d in ords[:gi]
                              if self.db["orders"][d]["status"] == g["status"]
                              and any(i["name"] == it["name"] for i in self.db["orders"][d]["items"])]
                    if not decoys or [i["name"] for i in g["items"]].count(it["name"]) != 1:
                        continue
                    level = 1 + len(self.out) % 3
                    dnames = {i["name"] for d in decoys for i in d["items"]}
                    dopts = [i["options"] for d in decoys for i in d["items"] if i["name"] == it["name"]]
                    key = next(((k, v) for k, v in it["options"].items()
                                if all(o.get(k) != v for o in dopts)), None)
                    other = next((i["name"] for i in g["items"] if i["name"] not in dnames), None)
                    if level == 3 and other is None:
                        level = 1
                    if level < 3 and key is None:
                        if other is None:
                            continue
                        level = 3
                    if level == 1:
                        which = f"the one with the {it['name']} that has {key[0]} {key[1]}"
                        extra = ""
                    elif level == 2:
                        which = f"the one with the {it['name']}"
                        extra = (f" You have more than one order with a {it['name']}; do not say "
                                 f"which unless the agent asks. If asked, it is the {it['name']} "
                                 f"with {key[0]} {key[1]}.")
                    else:
                        which = f"the one with the {it['name']} that also had the {other} in it"
                        extra = ""
                    if g["status"] == "delivered":
                        pm = self.pay(g)
                        if pm is None:
                            continue
                        act = {"action_id": "", "requestor": "assistant",
                               "name": "return_delivered_order_items",
                               "arguments": {"order_id": gid, "item_ids": [it["item_id"]],
                                             "payment_method_id": pm["id"]},
                               "info": None, "compare_args": None}
                        text = (f"You want to return the {it['name']} from your delivered order -- "
                                f"{which} -- with the refund going to the original payment "
                                f"method.{extra}")
                    else:
                        act = {"action_id": "", "requestor": "assistant", "name": "cancel_pending_order",
                               "arguments": {"order_id": gid, "reason": "no longer needed"},
                               "info": None, "compare_args": None}
                        text = (f"You want to cancel your pending order -- {which} -- because you no "
                                f"longer need it.{extra}")
                    self.emit("decoy_first", level, u["user_id"], text, [act], rng,
                              decoys=len(decoys))
                    break
                if self.count("decoy_first") >= PER_FAMILY:
                    return

    def keep_options(self, rng: random.Random) -> None:
        cands = [x for s in ("delivered", "pending") for x in self.orders(s)]
        rng.shuffle(cands)
        for oid, o in cands:
            if self.count("keep_options") >= PER_FAMILY:
                return
            pm = self.pay(o)
            if pm is None:
                continue
            level = 1 + len(self.out) % 3
            names = [i["name"] for i in o["items"]]
            olds, news, parts = [], [], []
            for it in o["items"]:
                if names.count(it["name"]) != 1 or len(olds) == (2 if level == 3 else 1):
                    continue
                cur = it["options"]
                found = None
                for k in cur:
                    for v in sorted({x["options"][k] for x in self.alts(it["item_id"])}):
                        g = self.only_change(it["item_id"], k, v) if v != cur[k] else None
                        if not g:
                            continue
                        # the test only bites if another available variant also has k=v but
                        # differs elsewhere (otherwise "one with k v" is already unique)
                        rivals = [x for x in self.alts(it["item_id"])
                                  if x["options"].get(k) == v and x["item_id"] != g["item_id"]]
                        if level >= 2 and not rivals:
                            continue
                        found = (g, k, v)
                        break
                    if found:
                        break
                if not found:
                    continue
                g, k, v = found
                if level == 1:
                    parts.append(f"change only the {k} of the {it['name']} ({H.words(cur)}) to {v}, "
                                 f"keeping every other option the same")
                elif len(olds) == 0:
                    parts.append(f"get the {it['name']} ({H.words(cur)}) with {k} {v} instead")
                else:
                    ph = self.st.vague(it["item_id"], g["item_id"])
                    if ph is None:
                        continue
                    parts.append(f"change the {it['name']} ({H.words(cur)}) to {ph}")
                olds.append(it["item_id"])
                news.append(g["item_id"])
            if not olds or (level == 3 and len(olds) < 2):
                continue
            if not self.affordable(pm, self.diff(o, olds, news)):
                continue
            verb = "exchange" if o["status"] == "delivered" else "modify"
            what = H.names_str(sorted(set(names)))
            text = (f"For your {o['status']} order with the {what}, you want to {verb} items: "
                    f"{'; and '.join(parts)}. Use your {pm_desc(pm)} for any price difference.")
            if level >= 2:
                text += (" If the agent asks about any option you did not mention, say it should "
                         "stay exactly as it is now.")
            self.emit("keep_options", level, o["user_id"], text,
                      [self.swap_action("x", o, olds, news, pm)], rng, n_items=len(olds))

    def pairing(self, rng: random.Random) -> None:
        cands = [x for s in ("delivered", "pending") for x in self.orders(s)]
        rng.shuffle(cands)
        for oid, o in cands:
            if self.count("pairing") >= PER_FAMILY:
                return
            pm = self.pay(o)
            if pm is None:
                continue
            level = 1 + len(self.out) % 3
            names = [i["name"] for i in o["items"]]
            twins = [(a, b) for i, a in enumerate(o["items"]) for b in o["items"][i + 1:]
                     if a["name"] == b["name"]]
            verb = "exchange" if o["status"] == "delivered" else "modify"
            if level == 3 and twins:
                a, b = twins[0]
                k = next((k for k, v in a["options"].items() if b["options"].get(k) != v), None)
                alts = [x for x in self.alts(a["item_id"]) if x["item_id"] != b["item_id"]]
                g = next((x for x in alts if self.unique_by_options(self.st.item[x["item_id"]][0],
                                                                    x["options"])), None)
                if k is None or g is None or not self.affordable(pm, self.diff(o, [a["item_id"]], [g["item_id"]])):
                    continue
                text = (f"You want to {verb} one {a['name']} in your {o['status']} order that has two "
                        f"of them: the one with {k} {a['options'][k]}. You want it replaced by the "
                        f"one with {H.words(g['options'])}. Use your {pm_desc(pm)} for any price "
                        f"difference. Name only the product at first; say which one (the {k}) "
                        f"only if the agent asks.")
                self.emit("pairing", 3, o["user_id"], text,
                          [self.swap_action("x", o, [a["item_id"]], [g["item_id"]], pm)], rng, kind="twin")
                continue
            if level == 3:
                level = 2
            uniq = [i for i in o["items"] if names.count(i["name"]) == 1]
            want = 2 if level == 1 else 3
            picks = []
            for it in uniq:
                opts = [x for x in self.alts(it["item_id"])
                        if self.unique_by_options(self.st.item[x["item_id"]][0], x["options"])]
                if opts:
                    picks.append((it, rng.choice(opts)))
                if len(picks) == want:
                    break
            if len(picks) < want:
                continue
            olds = [p[0]["item_id"] for p in picks]
            news = [p[1]["item_id"] for p in picks]
            if not self.affordable(pm, self.diff(o, olds, news)):
                continue
            # requested in the REVERSE of the order record's item order
            parts = [f"the {it['name']} ({H.words(it['options'])}) to the one with {H.words(g['options'])}"
                     for it, g in reversed(picks)]
            text = (f"For your {o['status']} order with the {H.names_str(sorted(set(names)))}, you want "
                    f"to {verb} several items at once: {'; '.join(parts)}. Use your {pm_desc(pm)} for "
                    f"any price difference.")
            self.emit("pairing", level, o["user_id"], text,
                      [self.swap_action("x", o, olds, news, pm)], rng, kind="reversed", n_items=want)


def main() -> int:
    db = json.loads(H.DB_PATH.read_text())
    val_tasks = [t for t in json.loads((H.TASKS / "tb500_hard.json").read_text())
                 if t["id"] in set(json.loads((H.TASKS / "split_hard.json").read_text())["val"])]
    val_tasks += json.loads((H.TASKS / "tb_vary_val.json").read_text())
    ex_o, ex_u = set(), set()
    for t in val_tasks:
        for a in t["evaluation_criteria"]["actions"]:
            if "order_id" in a["arguments"]:
                ex_o.add(a["arguments"]["order_id"])
            if "user_id" in a["arguments"]:
                ex_u.add(a["arguments"]["user_id"])
    g = Gen(db, ex_o, ex_u)
    for fam in ("items_addr", "decoy_first", "keep_options", "pairing"):
        getattr(g, fam)(random.Random(f"targeted:{fam}"))
    (H.TASKS / "tb_targeted.json").write_text(json.dumps(g.out, indent=1))
    (H.TASKS / "targeted_report.json").write_text(json.dumps(g.report, indent=1))
    c = Counter((r["family"], r["level"]) for r in g.report.values())
    print(f"{len(g.out)} targeted tasks | excluded {len(ex_o)} val orders, {len(ex_u)} val users")
    for k in sorted(c):
        print(f"  {k[0]:>13} L{k[1]}: {c[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
