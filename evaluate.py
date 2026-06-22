"""Evaluate a trained policy by actually playing the game.

Two ways to play:
  - greedy rollout: take the best-scoring legal neighbor every step. Pure
    policy, but per-step mistakes pile up on long ladders.
  - beam search: keep the k best partial ladders by summed log-prob. A tiny bit
    of search, a big rescue on the long ones. (Greedy kept dying about 8 steps
    into 5-letter puzzles, which is what pushed me to add it.)

The legal-moves mask is the same courtesy any board-game AI gets; the net still
has to pick the right move out of the ~10-30 on offer.

Run it as a script for the full report: greedy vs beam solve rates, a breakdown
by ladder length, a few sample ladders next to the BFS optimum, and how often
the raw unmasked pick would've been a real word anyway.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ladder import load_words, encode, decode, build_adjacency, shortest_path, load_json
from model import make_model


def batched_rollout(model, device, pairs, adjs, budget=20):
    """Greedy: play many episodes at once, one forward per step per length.

    pairs: dicts with start/target/opt.  adjs: {word_length: adjacency}.
    Returns copies of the pairs with path / solved / len filled in.
    """
    eps = [dict(p, cur=p["start"], path=[p["start"]], visited={p["start"]}, dead=False)
           for p in pairs]
    model.eval()
    for _ in range(budget):
        active = [e for e in eps if not e["dead"] and e["cur"] != e["target"]]
        if not active:
            break
        by_len = defaultdict(list)
        for e in active:
            by_len[len(e["cur"])].append(e)
        for L, group in by_len.items():
            cur = torch.tensor([encode(e["cur"]) for e in group], device=device)
            tgt = torch.tensor([encode(e["target"]) for e in group], device=device)
            with torch.no_grad():
                logits = model(cur, tgt).cpu()
            adj = adjs[L]
            for j, e in enumerate(group):
                best, best_s = None, None
                for n in adj[e["cur"]]:
                    if n in e["visited"]:
                        continue
                    i = next(k for k in range(L) if n[k] != e["cur"][k])
                    s = logits[j, i, ord(n[i]) - 97].item()
                    if best is None or s > best_s:
                        best, best_s = n, s
                if best is None:
                    e["dead"] = True  # greedy walked itself into a corner
                else:
                    e["cur"] = best
                    e["visited"].add(best)
                    e["path"].append(best)
    for e in eps:
        e["solved"] = e["cur"] == e["target"]
        e["len"] = len(e["path"]) - 1
    return eps


def beam_rollout(model, device, pairs, adjs, budget=20, beam=8):
    """Beam search over ladders, scored by summed log-prob of the moves.

    Steps are synchronized across the beam, so the first time the target
    shows up is also the fewest-steps ladder this beam can find.
    """
    eps = [dict(p, items=[(0.0, p["start"], (p["start"],), {p["start"]})], done=False)
           for p in pairs]
    model.eval()
    for step in range(1, budget + 1):
        by_len = defaultdict(list)
        for e in eps:
            if not e["done"]:
                for it in e["items"]:
                    by_len[len(e["target"])].append((e, it))
        if not by_len:
            break
        for L, lst in by_len.items():
            cur = torch.tensor([encode(it[1]) for _, it in lst], device=device)
            tgt = torch.tensor([encode(e["target"]) for e, _ in lst], device=device)
            with torch.no_grad():
                logp = F.log_softmax(model(cur, tgt).view(len(lst), -1), dim=1).cpu()
            for (e, it), row in zip(lst, logp):
                e.setdefault("_expand", []).append((it, row))

        for e in eps:
            if e["done"] or "_expand" not in e:
                continue
            adj = adjs[len(e["target"])]
            cands = []
            for (score, cur, path, visited), row in e.pop("_expand"):
                for n in adj[cur]:
                    if n in visited:
                        continue
                    i = next(k for k in range(len(n)) if n[k] != cur[k])
                    s = score + row[i * 26 + ord(n[i]) - 97].item()
                    if n == e["target"]:
                        e["done"], e["len"], e["path"] = True, step, list(path) + [n]
                        break
                    cands.append((s, n, path + (n,), visited | {n}))
                if e["done"]:
                    break
            if e["done"]:
                continue
            best = {}  # dedup by current word, keep best score
            for c in cands:
                if c[1] not in best or c[0] > best[c[1]][0]:
                    best[c[1]] = c
            e["items"] = sorted(best.values(), key=lambda c: -c[0])[:beam]
            if not e["items"]:
                e["done"] = True  # exhausted every option

    for e in eps:
        e["solved"] = "len" in e
        if not e["solved"]:
            e["len"] = budget
            e["path"] = list(e["items"][0][2]) if e.get("items") else [e["start"]]
        e.pop("items", None)
    return eps


def summarize(eps):
    solved = [e for e in eps if e["solved"]]
    optimal = [e for e in solved if e["len"] == e["opt"]]
    extra = sum(e["len"] - e["opt"] for e in solved) / max(len(solved), 1)
    return {"n": len(eps), "solve": len(solved) / len(eps),
            "optimal": len(optimal) / len(eps), "extra": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state"])

    adjs = {L: build_adjacency(load_words(p)) for L, p in ckpt["words"].items()}

    show = []
    for d in ckpt["data"]:
        meta = load_json(Path(d) / "meta.json")
        pairs = load_json(Path(d) / "rollout_val.json")
        greedy = batched_rollout(model, device, pairs, adjs, args.budget)
        beamed = beam_rollout(model, device, pairs, adjs, args.budget, args.beam)
        for tag, eps in [("greedy", greedy), (f"beam{args.beam}", beamed)]:
            s = summarize(eps)
            print(f"[L={meta['L']}] {tag:7s} n={s['n']}  solve={s['solve']:.3f}  "
                  f"optimal={s['optimal']:.3f}  extra_steps={s['extra']:.2f}")
        by = defaultdict(list)
        for e in beamed:
            by[e["opt"]].append(e)
        for opt_d in sorted(by):
            grp = by[opt_d]
            sv = sum(e["solved"] for e in grp) / len(grp)
            op = sum(e["solved"] and e["len"] == e["opt"] for e in grp) / len(grp)
            print(f"    opt={opt_d:2d}: n={len(grp):3d}  solve={sv:.2f}  optimal={op:.2f}")
        show += [(g, b) for g, b in zip(greedy, beamed) if g["opt"] >= 4]
        print()

    # a few ladders next to what BFS would do
    rng = random.Random(args.seed)
    for g, b in rng.sample(show, min(args.examples, len(show))):
        bfs = shortest_path(adjs[len(g["start"])], g["start"], g["target"])
        print(f"{g['start']} => {g['target']}  (optimal: {g['opt']})")
        for tag, e in [("greedy", g), ("beam  ", b)]:
            if e["solved"]:
                print(f"  {tag} ({e['len']:2d}): {' -> '.join(e['path'])}")
            else:
                print(f"  {tag} FAIL : {' -> '.join(e['path'][:8])} ...")
        print(f"  bfs    ({len(bfs) - 1:2d}): {' -> '.join(bfs)}")
        print()

    # curiosity metric: without the dictionary mask, is the argmax legal?
    for d in ckpt["data"]:
        meta = load_json(Path(d) / "meta.json")
        vocab = set(load_words(meta["words"]))
        val = torch.load(Path(d) / "val.pt")
        n = min(len(val["cur"]), 5000)
        with torch.no_grad():
            am = model(val["cur"][:n].to(device),
                       val["tgt"][:n].to(device)).view(n, -1).argmax(1).cpu()
        legal = 0
        for k in range(n):
            w = decode(val["cur"][k])
            i, c = divmod(am[k].item(), 26)
            cand = w[:i] + chr(97 + c) + w[i + 1:]
            legal += cand != w and cand in vocab
        print(f"[L={meta['L']}] raw argmax (no mask) is a legal move "
              f"{100 * legal / n:.1f}% of the time")


if __name__ == "__main__":
    main()
