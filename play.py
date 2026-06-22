"""Solve one ladder with a trained net, side by side with BFS.

    python play.py east west --ckpt runs/l4_mlp/best.pt
"""

import argparse

import torch

from ladder import load_words, build_adjacency, shortest_path
from model import make_model
from evaluate import batched_rollout, beam_rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("target")
    ap.add_argument("--ckpt", default="runs/l4_mlp/best.pt")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--beam", type=int, default=8)
    args = ap.parse_args()

    s, t = args.start.lower(), args.target.lower()
    if len(s) != len(t):
        raise SystemExit("words must be the same length")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    L = len(s)
    if L not in ckpt["words"]:
        raise SystemExit(f"this checkpoint only knows lengths {sorted(ckpt['words'])}")
    adj = build_adjacency(load_words(ckpt["words"][L]))
    for w in (s, t):
        if w not in adj:
            raise SystemExit(f"'{w}' is not in the {L}-letter word list")

    model = make_model(ckpt["cfg"])
    model.load_state_dict(ckpt["state"])

    bfs = shortest_path(adj, s, t)
    if bfs is None:
        raise SystemExit("BFS says these words aren't connected, so there's no ladder")

    pair = [{"start": s, "target": t, "opt": len(bfs) - 1}]
    print(f"bfs    ({len(bfs) - 1} steps): {' -> '.join(bfs)}")
    for tag, ep in [("greedy", batched_rollout(model, "cpu", pair, {L: adj}, args.budget)[0]),
                    (f"beam{args.beam}", beam_rollout(model, "cpu", pair, {L: adj}, args.budget, args.beam)[0])]:
        if ep["solved"]:
            note = "optimal!" if ep["len"] == ep["opt"] else f"+{ep['len'] - ep['opt']} extra"
            print(f"{tag:6s} ({ep['len']} steps, {note}): {' -> '.join(ep['path'])}")
        else:
            print(f"{tag:6s} gave up: {' -> '.join(ep['path'][:8])} ...")


if __name__ == "__main__":
    main()
