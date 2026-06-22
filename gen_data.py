"""Generate the training data with BFS.

The cheap trick here: one BFS from a target word hands you the ladder distance
from every other word in its component in a single pass. That turns every word
into a labeled example for free, since the best moves from any word are just the
neighbors sitting one step closer to the target. No path sampling, and you get
every optimal move for a state instead of one arbitrary shortest path (most
states have a few, and picking just one would be label noise).

Targets get split into train/val, so the val states all aim at words the net
was never trained to reach.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import torch

from ladder import load_words, encode, build_adjacency, bfs_dists, components, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", type=int, default=600)
    ap.add_argument("--val-targets", type=int, default=60)
    ap.add_argument("--per-target", type=int, default=300)
    ap.add_argument("--max-dist", type=int, default=10)
    ap.add_argument("--rollout-pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    words = load_words(args.words)
    L = len(words[0])
    adj = build_adjacency(words)
    comp = sorted(components(adj)[0])  # stay inside the biggest component
    print(f"{len(words)} words (L={L}), biggest component {len(comp)}")

    n_total = args.targets + args.val_targets
    picked = rng.sample(comp, n_total)
    splits = {"train": picked[:args.targets], "val": picked[args.targets:]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split, tgts in splits.items():
        cur_l, tgt_l, y_l = [], [], []
        dist_hist = Counter()
        for t in tgts:
            dist = bfs_dists(adj, t)
            states = [w for w, d in dist.items() if 1 <= d <= args.max_dist]
            rng.shuffle(states)
            for w in states[:args.per_target]:
                y = torch.zeros(L, 26, dtype=torch.bool)
                for n in adj[w]:
                    if dist[n] == dist[w] - 1:  # optimal move
                        i = next(k for k in range(L) if n[k] != w[k])
                        y[i][ord(n[i]) - 97] = True
                cur_l.append(encode(w))
                tgt_l.append(encode(t))
                y_l.append(y)
                dist_hist[dist[w]] += 1
        data = {
            "cur": torch.tensor(cur_l, dtype=torch.long),
            "tgt": torch.tensor(tgt_l, dtype=torch.long),
            "y": torch.stack(y_l),
        }
        torch.save(data, out / f"{split}.pt")
        print(f"{split}: {len(cur_l)} states from {len(tgts)} targets, "
              f"dist histogram {dict(sorted(dist_hist.items()))}")

    # held-out (start, target) pairs for rollout evaluation, val targets only
    per = args.rollout_pairs // len(splits["val"]) + 1
    pairs = []
    for t in splits["val"]:
        dist = bfs_dists(adj, t)
        cands = [w for w, d in dist.items() if 2 <= d <= args.max_dist]
        for w in rng.sample(cands, min(per, len(cands))):
            pairs.append({"start": w, "target": t, "opt": dist[w]})
    rng.shuffle(pairs)
    pairs = pairs[:args.rollout_pairs]
    save_json(pairs, out / "rollout_val.json", indent=1)
    save_json({"words": args.words, "L": L}, out / "meta.json")
    print(f"rollout_val: {len(pairs)} pairs")


if __name__ == "__main__":
    main()
