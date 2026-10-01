"""v6 training tasks: tau3-structured, tau3-styled, answer-first (Sept 27).

Why. The base agent's tau3 failures concentrate in task STRUCTURES the training set under-
represents (base success with / without the feature, gpt-5.2 customer, n=8): 3+ writes in one
call 0.49 / 0.72 (tau3 18% of tasks, training 9%), writes on 2+ orders 0.61 / 0.72 (32% vs 19%),
order-address changes 0.43, relative/preference targets 0.46, multi-item swaps 0.36. tau3's
scenarios are also written differently from ours: the customer is rarely handed an exact option
list (11% vs 60%), states a motive, describes targets relative to what they have or by another
order, withholds facts until asked, and gets pacing and behaviour rules. The customer PROMPT is
identical -- training and eval both run tau2's UserSimulator over str(user_scenario) -- so the
scenario text is the whole difference.

What. Every task is built answer-first from the store DB: pick a customer, pick requests on their
orders, compute the exact gold writes, and only then render the scenario. Each target
description is checked to pick out exactly one available variant. Families (ids t6_<fam>_<n>):

  bundle     3-5 writes across 2-3 of one customer's orders: returns, exchanges, single-item
             modifies, order-address and default-address changes looked up rather than spoken,
             sometimes a tracking-number question graded by communicate_info
  single     one or two writes on one order with a hard target: relative ("the next size up,
             everything else the same"), by another order ("the same one I have in my other
             order"), preference ("the cheapest in black"), or a twin item named only if asked
  choose     "list every option that fits and its price, then I'll take the most expensive /
             cheapest": the candidate prices go in communicate_info
  refuse     a request the policy forbids (cancel a delivered order, return a pending one,
             change the address of a delivered order) plus a question whose answer the agent must
             state -- no write in the gold, but not a free reward

Nothing is taken from tau3's tasks: they were read for structure and style only (disclosed in the
write-up), orders and users any val task touches are excluded here, and the VM drops any task that
shares a write signature with the held-out 114 (build_v6.py).

Usage (repo root): python3 taskgen/gen_v6.py [db.json] -> tb_v6.json, v6_report.json
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

import harden_tb as H
from gen_targeted import Gen, pm_desc

N = {"bundle": 320, "single": 260, "choose": 90, "refuse": 70}
ADJ = ["patient", "busy", "friendly", "rigid", "cautious", "impatient", "organized", "messy",
       "polite", "direct", "anxious", "optimistic", "pessimistic", "logical", "chatty", "shy"]
ORD_WORDS = [["small", "medium", "large"], ["low", "medium", "high"], ["basic", "professional"],
             ["beginner", "intermediate", "expert"], ["standard", "high-back"], ["half", "full"],
             ["S", "M", "L", "XL", "XXL"], ["1080p", "2K", "4K", "5K"]]
UNIT = {"gb": 1, "tb": 1024, "ml": 1, "l": 1000, "liter": 1000, "liters": 1000, "cup": 1, "cups": 1,
        "mah": 1, "mp": 1, "inch": 1, "inches": 1, "ft": 1, "hours": 1, "mm": 1, "bar": 1, "x": 1,
        "degrees": 1, "w": 1, "lbs": 1, "": 1}
DIR_WORD = {"size": ("size", "bigger", "smaller"), "capacity": ("capacity", "larger", "smaller"),
            "storage": ("storage", "more", "less"), "resolution": ("resolution", "higher", "lower"),
            "screen size": ("screen", "bigger", "smaller"), "pieces": ("piece count", "higher", "lower"),
            "frame size": ("frame", "larger", "smaller"), "ram": ("RAM", "more", "less"),
            "RAM": ("RAM", "more", "less"), "battery life": ("battery life", "longer", "shorter"),
            "zoom": ("zoom", "stronger", "weaker"), "diameter": ("diameter", "larger", "smaller"),
            "height": ("height", "taller", "shorter"), "thickness": ("thickness", "thicker", "thinner"),
            "length": ("length", "longer", "shorter"), "processor": ("processor", "faster", "slower"),
            "pressure": ("pressure", "higher", "lower"), "difficulty level": ("difficulty", "harder", "easier"),
            "room size": ("room size", "larger", "smaller"), "ventilation": ("ventilation", "more", "less"),
            "weight range": ("weight range", "heavier", "lighter"), "field of view": ("field of view", "wider", "narrower")}


def rank(key: str, values: set[str]) -> list[str] | None:
    """The values of one option key in increasing order, or None if they are not orderable."""
    for ws in ORD_WORDS:
        if values <= set(ws):
            return [w for w in ws if w in values]
    if key == "processor" and values <= {"i3", "i5", "i7", "i9"}:
        return sorted(values)
    out = []
    for v in values:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-?\s*([a-zA-Z]*)(?: SSD)?(?:-\d+ lbs)?", v.strip())
        if not m or m.group(2).lower() not in UNIT:
            m2 = re.fullmatch(r"(\d+)-\d+ lbs", v)
            if not m2:
                return None
            out.append((float(m2.group(1)), v))
            continue
        out.append((float(m.group(1)) * UNIT[m.group(2).lower()], v))
    if len({u for u, _ in out}) != len(out):
        return None
    return [v for _, v in sorted(out)]


class V6(Gen):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fam_n = Counter()
        # target styles to draw from; p4-i (V6_STYLES=relative,partial) keeps only the two the
        # adapter fails most (0.60 and 0.58 success in p4-h's rollouts vs 0.82 for the rest)
        self.styles = ["relative", "relative", "cross_order", "partial", "preference", "exact"]
        self.suite_only = False

    # ------------------------------------------------------------ targets
    def product_values(self, pid: str, key: str) -> set[str]:
        return {v["options"][key] for v in self.db["products"][pid]["variants"].values() if key in v["options"]}

    def target(self, o: dict, it: dict, style: str, rng: random.Random) -> tuple[dict, str, str] | None:
        """(gold variant, phrase for the customer, motive) for changing item `it`, or None."""
        iid, name, cur = it["item_id"], it["name"], it["options"]
        pid = self.st.item[iid][0]
        keys = list(cur)
        rng.shuffle(keys)
        if style == "relative":
            for k in keys:
                order = rank(k, self.product_values(pid, k))
                if not order or k not in DIR_WORD or cur[k] not in order:
                    continue
                i = order.index(cur[k])
                for step, idx in ((1, i + 1), (-1, i - 1)):
                    if 0 <= idx < len(order) and (g := self.only_change(iid, k, order[idx])):
                        noun, up, down = DIR_WORD[k]
                        word = up if step > 0 else down
                        phrase = (f"the same {name} but with the next {noun} "
                                  f"{'up' if step > 0 else 'down'} ({word} than yours), every other "
                                  f"option unchanged")
                        motive = ("it turned out to be too small for what you need" if step > 0
                                  else "it is more than you actually need")
                        return g, phrase, motive
            return None
        if style == "cross_order":
            uid = o["user_id"]
            others = [self.db["orders"][x] for x in self.db["users"][uid]["orders"] if x != o["order_id"]]
            hold = [(x, i) for x in others for i in x["items"] if i["name"] == name and i["item_id"] != iid]
            if len(hold) != 1:
                return None
            x, i2 = hold[0]
            g = self.st.item[i2["item_id"]][1]
            if not g["available"]:
                return None
            where = f"your {x['status']} order" if sum(y["status"] == x["status"] for y in others) == 1 \
                else f"your order that was shipped to {x['address']['city']}"
            if where.startswith("your order that") and sum(y["address"]["city"] == x["address"]["city"] for y in others) != 1:
                return None
            return g, f"exactly the same kind of {name} as the one in {where}", "you liked that one better"
        if style == "partial":
            for k in keys:
                for v in sorted(self.product_values(pid, k) - {cur[k]}):
                    if (g := self.only_change(iid, k, v)):
                        return g, f"the same {name} but with {k} {v}, keeping everything else as it is", \
                            f"you would prefer {k} {v}"
            return None
        if style == "preference":
            for g in sorted(self.alts(iid), key=lambda v: rng.random()):
                ph = self.st.vague(iid, g["item_id"])
                if ph:
                    return g, ph.replace("available option", f"available {name}").replace(
                        "available one", f"available {name}"), \
                        "you are watching your budget" if "cheapest" in ph else "you want the best one"
            return None
        if style == "exact":
            for g in sorted(self.alts(iid), key=lambda v: rng.random()):
                if self.unique_by_options(pid, g["options"]):
                    return g, f"the {name} with {H.words(g['options'])}", "you picked the wrong one"
        return None

    def twin_note(self, o: dict, it: dict) -> str | None:
        twins = [i for i in o["items"] if i["name"] == it["name"] and i["item_id"] != it["item_id"]]
        if not twins:
            return ""
        if len(twins) > 1:
            return None
        k = next((k for k, v in it["options"].items() if twins[0]["options"].get(k) != v), None)
        if k is None:
            return None       # identical twins: which one changes is not observable in the DB
        return (f" You have two {it['name']}s in that order; say which one only if the agent asks: "
                f"the one with {k} {it['options'][k]}.")

    # ------------------------------------------------------------ requests on one order
    def order_desc(self, o: dict, rng: random.Random, others: list[dict]) -> str:
        """How the customer names the order: by product, by destination, or by status."""
        names = sorted({i["name"] for i in o["items"]})
        c = o["address"]["city"]
        if rng.random() < 0.35 and sum(x["address"]["city"] == c for x in others) == 0:
            return f"your {o['status']} order that was shipped to {c}"
        n = rng.choice(names)
        if sum(any(i["name"] == n for i in x["items"]) for x in others) == 0:
            return f"your {o['status']} order with the {n}"
        return f"your {o['status']} order with the {H.names_str(names)}"

    def request(self, o: dict, rng: random.Random, others: list[dict], kinds: list[str]):
        """One or two gold writes on order o, with the sentences asking for them."""
        pm = self.pay(o)
        if pm is None:
            return None
        desc = self.order_desc(o, rng, others)
        acts, says, motives = [], [], []
        names = [i["name"] for i in o["items"]]
        if o["status"] == "delivered":
            kind = rng.choice([k for k in kinds if k in ("return", "exchange")] or ["return"])
            if kind == "return":
                its = [i for i in o["items"] if names.count(i["name"]) == 1]
                if not its:
                    return None
                pick = rng.sample(its, min(len(its), rng.choice([1, 1, 2])))
                acts.append({"name": "return_delivered_order_items",
                             "arguments": {"order_id": o["order_id"], "item_ids": [i["item_id"] for i in pick],
                                           "payment_method_id": pm["id"]}})
                says.append(f"return the {H.names_str([i['name'] for i in pick])} from {desc}, "
                            f"refunded to the original payment method")
                motives.append("you no longer need " + ("it" if len(pick) == 1 else "them"))
                return acts, says, motives, ""
            its = [i for i in o["items"] if self.alts(i["item_id"])]
            rng.shuffle(its)
            for it in its:
                note = self.twin_note(o, it)
                if note is None:
                    continue
                style = rng.choice(self.styles)
                t = self.target(o, it, style, rng) or self.target(o, it, "partial", rng)
                if not t or not self.affordable(pm, self.diff(o, [it["item_id"]], [t[0]["item_id"]])):
                    continue
                acts.append(self.swap_action("x", o, [it["item_id"]], [t[0]["item_id"]], pm))
                says.append(f"exchange the {it['name']} from {desc} for {t[1]}, using your "
                            f"{pm_desc(pm)} for any price difference")
                motives.append(t[2])
                return acts, says, motives, note
            return None
        if o["status"] == "pending":
            kind = rng.choice([k for k in kinds if k in ("modify", "address", "modify+address", "cancel")] or ["modify"])
            if kind == "cancel":
                acts.append({"name": "cancel_pending_order",
                             "arguments": {"order_id": o["order_id"], "reason": "no longer needed"}})
                says.append(f"cancel {desc} because you no longer need it")
                motives.append("")
                return acts, says, motives, ""
            note = ""
            if kind in ("modify", "modify+address"):
                its = [i for i in o["items"] if self.alts(i["item_id"])]
                rng.shuffle(its)
                done = False
                for it in its:
                    note = self.twin_note(o, it)
                    if note is None:
                        continue
                    style = rng.choice(self.styles)
                    t = self.target(o, it, style, rng) or self.target(o, it, "partial", rng)
                    if not t or not self.affordable(pm, self.diff(o, [it["item_id"]], [t[0]["item_id"]])):
                        continue
                    acts.append(self.swap_action("x", o, [it["item_id"]], [t[0]["item_id"]], pm))
                    says.append(f"change the {it['name']} in {desc} to {t[1]}, paying any difference "
                                f"with your {pm_desc(pm)}")
                    motives.append(t[2])
                    done = True
                    break
                if not done:
                    return None
            if kind in ("address", "modify+address"):
                a = self.lookup_address(o, rng, others)
                if a is None:
                    return None if kind == "address" else (acts, says, motives, note)
                addr, how = a
                acts.append({"name": "modify_pending_order_address",
                             "arguments": {"order_id": o["order_id"], **{k: addr[k] for k in (
                                 "address1", "address2", "city", "state", "country", "zip")}}})
                says.append(("also " if kind == "modify+address" else "") +
                            f"change the shipping address of {desc if kind == 'address' else 'that same order'} to {how}")
                motives.append("you will not be at the old address")
            return acts, says, motives, note
        return None

    def lookup_address(self, o: dict, rng: random.Random, others: list[dict]):
        """An address the customer will not spell out: their default, or the one on another order
        (named by city, which must be unique among their orders). None if neither differs."""
        u = self.db["users"][o["user_id"]]
        cur = (o["address"]["address1"], o["address"]["zip"])
        opts = []
        if (u["address"]["address1"], u["address"]["zip"]) != cur:
            opts.append((u["address"], "your default address on your account (you would rather not "
                                       "say it out loud; the agent can look it up)"))
        cities = [x["address"]["city"] for x in others]
        for x in others:
            a = x["address"]
            if cities.count(a["city"]) == 1 and (a["address1"], a["zip"]) != cur and a["city"] != o["address"]["city"]:
                opts.append((a, f"the {a['city']} address you used for another order (you would rather "
                                f"not say the street; it is on that order)"))
        if self.suite_only:
            # p4-i: prefer addresses whose suite differs from the current one -- the agent kept
            # copying the old address2 into a looked-up address (55 wrong address2 in p4-h)
            diff = [x for x in opts if x[0]["address2"] != o["address"]["address2"]]
            opts = diff or opts
        return rng.choice(opts) if opts else None

    # ------------------------------------------------------------ rendering
    def render(self, fam: str, uid: str, says: list[str], motives: list[str], notes: str,
               acts: list[dict], rng: random.Random, comm: list[str] | None = None, extra: str = "",
               rep: dict | None = None) -> None:
        u = self.db["users"][uid]
        mot = next((m for m in motives if m), "")
        lead = rng.choice(["You are contacting the store because", "You are calling since", "You reach out because"])
        body = "; ".join(says)
        text = (f"{lead} {mot}. " if mot else "") + f"You want to {body}." + extra + notes
        pace = rng.random()
        ti = f"You are {', '.join(rng.sample(ADJ, 3))}."
        if len(says) > 1 and pace < 0.5:
            ti += " You like to say one thing at a time: bring up the next request only after the agent has finished the current one."
        elif len(says) > 1 and pace < 0.8:
            ti += " Mention all of your requests in your first message."
        if any(a["name"] in ("exchange_delivered_order_items", "modify_pending_order_items") for a in acts):
            ti += " You do not want to cancel any orders."
        if acts and rng.random() < 0.3:
            ti += " Do not end the conversation until every change you asked for has been made."
        ti += " Never state an order id or item id; refer to your orders the way the instructions do."
        if rng.random() < 0.5:
            known = f"You are {u['name']['first_name']} {u['name']['last_name']} and your zip code is {u['address']['zip']}."
            unknown = "You do not remember your email address."
        else:
            known = f"You are {u['name']['first_name']} {u['name']['last_name']}. Your email is {u['email']}."
            unknown = None
        key = json.dumps([body, acts], sort_keys=True)
        if key in self.seen:
            return
        self.seen.add(key)
        tid = f"t6_{fam}_{self.fam_n[fam]:03d}"
        self.fam_n[fam] += 1
        for i, a in enumerate(acts):
            a.update({"action_id": f"{tid}_{i}", "requestor": "assistant", "info": None, "compare_args": None})
        self.out.append({
            "id": tid,
            "description": {"purpose": f"v6:{fam}", "relevant_policies": None,
                            "notes": "gen_v6.py (Sept 27), answer-first from the store DB."},
            "user_scenario": {"persona": None, "instructions": {
                "domain": "retail", "reason_for_call": text, "known_info": known,
                "unknown_info": unknown, "task_instructions": ti}},
            "initial_state": None,
            "evaluation_criteria": {"actions": acts, "communicate_info": comm or [],
                                    "nl_assertions": None, "reward_basis": ["DB", "COMMUNICATE"]}})
        self.report[tid] = {"family": fam, "writes": len(acts), **(rep or {})}

    # ------------------------------------------------------------ families
    def users(self, rng):
        us = [u for u in self.db["users"].values() if u["user_id"] not in self.ex_u]
        rng.shuffle(us)
        return us

    def open_orders(self, u):
        return [self.db["orders"][x] for x in u["orders"] if x not in self.ex_o
                and self.db["orders"][x]["status"] in ("pending", "delivered")]

    def bundle(self, rng):
        for u in self.users(rng) * getattr(self, "reps", 3):
            if self.fam_n["bundle"] >= N["bundle"]:
                return
            ords = self.open_orders(u)
            if len(ords) < 2:
                continue
            pick = rng.sample(ords, min(len(ords), rng.choice([2, 3, 3])))
            acts, says, motives, notes = [], [], [], ""
            for o in pick:
                others = [x for x in (self.db["orders"][y] for y in u["orders"]) if x["order_id"] != o["order_id"]]
                kinds = (["return", "exchange"] if o["status"] == "delivered"
                         else ["modify+address", "modify", "address", "modify+address", "cancel"])
                r = self.request(o, rng, others, kinds)
                if r:
                    acts += r[0]; says += r[1]; motives += r[2]; notes += r[3]
            if rng.random() < 0.3:          # default-address change to an address on an order
                a = self.lookup_address({"user_id": u["user_id"], "address": u["address"], "order_id": ""},
                                        rng, [self.db["orders"][y] for y in u["orders"]])
                if a and not a[1].startswith("your default"):
                    acts.append({"name": "modify_user_address", "arguments": {"user_id": u["user_id"], **{
                        k: a[0][k] for k in ("address1", "address2", "city", "state", "country", "zip")}}})
                    says.append(f"change your default address on your account to {a[1]}")
            comm, extra = [], ""
            if rng.random() < 0.3:
                touched = {a["arguments"].get("order_id") for a in acts}
                for y in u["orders"]:
                    x = self.db["orders"][y]
                    tr = [t for f in x["fulfillments"] for t in f["tracking_id"]]
                    if y not in touched and x["status"] in ("delivered", "processed") and len(tr) == 1 \
                            and sum(self.db["orders"][z]["status"] == x["status"] for z in u["orders"]) == 1:
                        comm, extra = [tr[0]], f" You also want the tracking number of your {x['status']} order."
                        break
            if len(acts) >= 3 and len({a["arguments"].get("order_id") for a in acts if "order_id" in a["arguments"]}) >= 2:
                self.render("bundle", u["user_id"], says, motives, notes, acts, rng, comm, extra)

    def single(self, rng):
        for u in self.users(rng) * 3:
            if self.fam_n["single"] >= N["single"]:
                return
            for o in rng.sample(self.open_orders(u), len(self.open_orders(u))):
                others = [x for x in (self.db["orders"][y] for y in u["orders"]) if x["order_id"] != o["order_id"]]
                kinds = ["exchange"] if o["status"] == "delivered" else ["modify", "modify+address"]
                r = self.request(o, rng, others, kinds)
                if r and any(a["name"] in ("exchange_delivered_order_items", "modify_pending_order_items") for a in r[0]):
                    self.render("single", u["user_id"], *r[1:3], r[3], r[0], rng)
                    break

    def choose(self, rng):
        for u in self.users(rng) * 2:
            if self.fam_n["choose"] >= N["choose"]:
                return
            for o in self.open_orders(u):
                pm = self.pay(o)
                if pm is None:
                    continue
                names = [i["name"] for i in o["items"]]
                for it in o["items"]:
                    if names.count(it["name"]) != 1:
                        continue
                    alts = self.alts(it["item_id"])
                    ks = [k for k in it["options"] if 2 <= len({a["options"][k] for a in alts}) ]
                    if not ks:
                        continue
                    k = rng.choice(ks)
                    v = rng.choice(sorted({a["options"][k] for a in alts}))
                    cands = [a for a in alts if a["options"][k] == v]
                    prices = sorted(a["price"] for a in cands)
                    if not 2 <= len(cands) <= 4 or len(set(prices)) != len(prices):
                        continue
                    most = rng.random() < 0.5
                    g = max(cands, key=lambda a: a["price"]) if most else min(cands, key=lambda a: a["price"])
                    if not self.affordable(pm, self.diff(o, [it["item_id"]], [g["item_id"]])):
                        continue
                    verb = "exchange" if o["status"] == "delivered" else "change"
                    desc = self.order_desc(o, rng, [x for x in (self.db["orders"][y] for y in u["orders"]) if x["order_id"] != o["order_id"]])
                    say = (f"{verb} the {it['name']} from {desc} for one with {k} {v}. Before choosing, ask "
                           f"the agent to list every available option with {k} {v} and its price, then "
                           f"pick the {'most expensive' if most else 'cheapest'} one, using your {pm_desc(pm)} "
                           f"for any difference")
                    comm = [f"{p:.2f}".rstrip("0").rstrip(".") for p in prices]
                    self.render("choose", u["user_id"], [say], ["you want to compare prices first"], "",
                                [self.swap_action("x", o, [it["item_id"]], [g["item_id"]], pm)], rng, comm)
                    break
                else:
                    continue
                break

    def refuse(self, rng):
        for u in self.users(rng) * 2:
            if self.fam_n["refuse"] >= N["refuse"]:
                return
            ords = [self.db["orders"][x] for x in u["orders"] if x not in self.ex_o]
            for o in ords:
                ask = None
                if o["status"] == "delivered":
                    ask = "cancel it entirely and get a refund"
                elif o["status"] == "processed":
                    ask = "change its shipping address"
                elif o["status"] == "pending":
                    ask = "return the items in it now"
                if not ask:
                    continue
                tr = [t for f in o["fulfillments"] for t in f["tracking_id"]]
                total = f"{sum(i['price'] for i in o['items']):.2f}"
                q, comm = ((f" Also ask for its tracking number.", [tr[0]]) if len(tr) == 1
                           else (" Also ask what the total price of the items in it is.", [total]))
                desc = self.order_desc(o, rng, [x for x in ords if x is not o])
                say = f"{ask} for {desc}"
                extra = (q + " If the agent says your request is not possible, accept that and do not "
                         "ask for anything else instead.")
                self.render("refuse", u["user_id"], [say], ["you want to sort out that order"], "", [],
                            rng, comm, extra)
                break


def main() -> int:
    """Env overrides (Sept 27, p4-g): V6_ONLY=bundle V6_N=500 V6_REPS=10 V6_PREFIX=t6b V6_OUT=tb_v6b.json
    generates only more bundles, under new ids, skipping any request already in tb_v6.json."""
    import os
    only = os.environ.get("V6_ONLY")
    prefix = os.environ.get("V6_PREFIX", "t6")
    out_name = os.environ.get("V6_OUT", "tb_v6.json")
    if only:
        n = int(os.environ.get("V6_N", "300"))
        N.clear()
        for f in only.split(","):
            N[f] = n
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
    g = V6(db, ex_o, ex_u)
    g.reps = int(os.environ.get("V6_REPS", "3"))
    if os.environ.get("V6_STYLES"):
        g.styles = os.environ["V6_STYLES"].split(",")
    g.suite_only = os.environ.get("V6_SUITE") == "1"
    for fam in N:
        getattr(g, fam)(random.Random(f"v6:{fam}" if prefix == "t6" else f"v6:{fam}:{prefix}"))
    canon = lambda t: json.dumps(sorted(json.dumps([a["name"], a["arguments"]], sort_keys=True)  # noqa: E731
                                        for a in t["evaluation_criteria"]["actions"]))
    if out_name != "tb_v6.json" and (H.TASKS / "tb_v6.json").exists():
        old = {canon(t) for t in json.loads((H.TASKS / "tb_v6.json").read_text())}
        keep = [t for t in g.out if canon(t) not in old]
        g.report = {t["id"]: g.report[t["id"]] for t in keep}
        g.out = keep
    for t in g.out:
        t["id"] = t["id"].replace("t6_", prefix + "_", 1)
        for a in t["evaluation_criteria"]["actions"]:
            a["action_id"] = a["action_id"].replace("t6_", prefix + "_", 1)
    g.report = {k.replace("t6_", prefix + "_", 1): v for k, v in g.report.items()}
    (H.TASKS / out_name).write_text(json.dumps(g.out, indent=1))
    (H.TASKS / out_name.replace("tb_", "").replace(".json", "_report.json")).write_text(
        json.dumps(g.report, indent=1))
    w = Counter(r["writes"] for r in g.report.values())
    print(f"{len(g.out)} {prefix} tasks {dict(g.fam_n)} | writes per task {dict(sorted(w.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
