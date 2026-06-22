"""Train a policy net to imitate BFS-optimal moves.

Loss is cross-entropy against the full set of optimal moves, spread evenly.
Since a state usually has several equally short next moves, training toward one
arbitrary shortest path would just be label noise.

Every few epochs (--eval-every) the net plays real held-out ladders with a
greedy rollout, so log.csv ends up being a learning curve of actual solve rate
against wall-clock time, which is really the whole question here.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ladder import load_words, build_adjacency, load_json
from model import make_model, n_params
from evaluate import batched_rollout, summarize


def soft_ce(logits, y):
    B = logits.shape[0]
    logp = F.log_softmax(logits.view(B, -1), dim=1)
    p = y.view(B, -1).float()
    p = p / p.sum(1, keepdim=True)
    return -(p * logp).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="one or more gen_data output dirs")
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="mlp")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)  # mlp
    ap.add_argument("--d", type=int, default=128)       # transformer
    ap.add_argument("--layers", type=int, default=3)    # transformer
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--rollout-n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metas = [load_json(Path(d) / "meta.json") for d in args.data]
    if args.arch == "mlp" and len({m["L"] for m in metas}) > 1:
        ap.error("mlp is fixed-length; train it on one word length at a time")

    cfg = {"arch": args.arch, "L": metas[0]["L"], "hidden": args.hidden,
           "d": args.d, "layers": args.layers}
    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_ds = [torch.load(Path(d) / "train.pt") for d in args.data]
    val_ds = [torch.load(Path(d) / "val.pt") for d in args.data]
    n_train = sum(len(d["cur"]) for d in train_ds)
    print(f"device={device} arch={args.arch} params={n_params(model):,}")
    print(f"train states: {n_train}  val states: {sum(len(d['cur']) for d in val_ds)}")

    # rollout machinery: the word graph per length, plus the held-out pairs
    adjs = {}
    pairs = []
    for m, d in zip(metas, args.data):
        adjs[m["L"]] = build_adjacency(load_words(m["words"]))
        pairs += load_json(Path(d) / "rollout_val.json")[:args.rollout_n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.csv", "w", newline="")
    log = csv.writer(logf)
    log.writerow(["epoch", "examples", "train_s", "total_s", "loss",
                  "val_acc", "solve", "optimal", "extra"])

    def save(name):
        torch.save({"cfg": cfg, "state": model.state_dict(),
                    "words": {m["L"]: m["words"] for m in metas},
                    "data": args.data}, out / name)

    t0 = time.time()
    train_s = 0.0
    examples = 0
    best_solve = -1.0
    hist = []  # (total_s, solve) for the time-to-X summary
    for epoch in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        # batch schedule across datasets; every batch is one word length
        sched = []
        for di, d in enumerate(train_ds):
            perm = torch.randperm(len(d["cur"]))
            for s in range(0, len(perm), args.bs):
                sched.append((di, perm[s:s + args.bs]))
        random.shuffle(sched)

        tot_loss = 0.0
        for di, idx in sched:
            d = train_ds[di]
            cur, tgt = d["cur"][idx].to(device), d["tgt"][idx].to(device)
            y = d["y"][idx].to(device)
            loss = soft_ce(model(cur, tgt), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            examples += len(idx)
        train_s += time.time() - te
        tot_loss /= n_train

        # val accuracy = argmax move is one of the optimal moves
        model.eval()
        accs = []
        with torch.no_grad():
            for d in val_ds:
                hits = 0
                for s in range(0, len(d["cur"]), 4096):
                    cur = d["cur"][s:s + 4096].to(device)
                    tgt = d["tgt"][s:s + 4096].to(device)
                    y = d["y"][s:s + 4096]
                    B = len(cur)
                    am = model(cur, tgt).view(B, -1).argmax(1).cpu()
                    hits += y.view(B, -1).gather(1, am[:, None]).sum().item()
                accs.append(hits / len(d["cur"]))
        val_acc = sum(accs) / len(accs)

        row = {"solve": "", "optimal": "", "extra": ""}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eps = batched_rollout(model, device, pairs, adjs, args.budget)
            row = summarize(eps)
            hist.append((time.time() - t0, row["solve"]))
            if row["solve"] >= best_solve:
                best_solve = row["solve"]
                save("best.pt")

        total_s = time.time() - t0
        log.writerow([epoch, examples, f"{train_s:.1f}", f"{total_s:.1f}",
                      f"{tot_loss:.4f}", f"{val_acc:.4f}",
                      row["solve"], row["optimal"], row["extra"]])
        logf.flush()
        msg = (f"epoch {epoch:3d} | {total_s:6.1f}s | loss {tot_loss:.4f} | "
               f"val_acc {val_acc:.3f}")
        if row["solve"] != "":
            msg += (f" | solve {row['solve']:.3f} optimal {row['optimal']:.3f}"
                    f" extra {row['extra']:.2f}")
        print(msg, flush=True)

    save("last.pt")
    logf.close()
    print(f"\ndone in {time.time() - t0:.1f}s, best solve {best_solve:.3f}")
    for thresh in (0.5, 0.8, 0.9, 0.95):
        hit = next((f"{t:.0f}s" for t, sv in hist if sv >= thresh), "never")
        print(f"  time to {int(thresh * 100)}% solve: {hit}")


if __name__ == "__main__":
    main()
