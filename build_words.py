"""Build the word database.

Crosses the big dwyl dictionary (tells us what's a real word) with norvig's
web frequency counts (tells us which words people actually use) and keeps the
top-N per length. Taking the whole dictionary makes the graph full of junk
like "yald"; taking only the top 1000 makes it too sparse for good ladders.
A few thousand per length is the sweet spot.

Downloads the raw lists on first run if they're missing.
"""

import argparse
from pathlib import Path

from ladder import build_adjacency, components

RAW = Path(__file__).parent / "words" / "raw"
OUT = Path(__file__).parent / "words"

SOURCES = {
    "words_alpha.txt": "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
    "count_1w.txt": "https://norvig.com/ngrams/count_1w.txt",
}


def fetch_raw():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = RAW / name
        if not path.exists():
            import requests
            print(f"downloading {name} ...")
            path.write_bytes(requests.get(url, timeout=60).content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--top", type=int, nargs="+", default=[3500, 6000],
                    help="words to keep per length, by frequency")
    args = ap.parse_args()
    assert len(args.lengths) == len(args.top)

    fetch_raw()
    alpha = set((RAW / "words_alpha.txt").read_text().split())

    ranked = []  # count_1w.txt is already sorted by count, descending
    with open(RAW / "count_1w.txt") as f:
        for line in f:
            ranked.append(line.split()[0])

    for length, top in zip(args.lengths, args.top):
        words = [w for w in ranked
                 if len(w) == length and w in alpha and w.isascii() and w.isalpha()]
        words = words[:top]
        adj = build_adjacency(words)
        comps = components(adj)
        isolated = sum(1 for ns in adj.values() if not ns)
        edges = sum(len(ns) for ns in adj.values()) // 2
        avg_deg = 2 * edges / len(words)

        out = OUT / f"words{length}.txt"
        out.write_text("\n".join(sorted(words)) + "\n")
        print(f"len={length}: kept {len(words)} -> {out}")
        print(f"  edges={edges} avg_degree={avg_deg:.1f} isolated={isolated} "
              f"components={len(comps)} biggest={len(comps[0])} "
              f"({100 * len(comps[0]) / len(words):.0f}%)")


if __name__ == "__main__":
    main()
