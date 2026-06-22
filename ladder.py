"""Core word ladder stuff: adjacency, BFS, components.

The neighbor trick: bucket every word by its one-blank patterns, so "_ant"
collects cant/pant/want and friends. One pass, no O(n^2) pair comparison.
"""

import itertools
import json
from collections import defaultdict, deque
from pathlib import Path


def load_words(path):
    with open(path) as f:
        return [w.strip().lower() for w in f if w.strip()]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path, indent=None):
    Path(path).write_text(json.dumps(obj, indent=indent))


def encode(word):
    return [ord(c) - 97 for c in word]


def decode(idxs):
    return "".join(chr(97 + int(i)) for i in idxs)


def build_adjacency(words):
    buckets = defaultdict(list)
    for w in words:
        for i in range(len(w)):
            buckets[w[:i] + "_" + w[i + 1:]].append(w)
    adj = {w: set() for w in words}
    for group in buckets.values():
        # everything in a bucket is one letter apart, so it's all mutual
        # neighbors; combinations does each pair once instead of twice
        for a, b in itertools.combinations(group, 2):
            adj[a].add(b)
            adj[b].add(a)
    # sorted lists: deterministic iteration, and cheaper to loop over than sets
    return {w: sorted(ns) for w, ns in adj.items()}


def bfs_dists(adj, src):
    """Distance from src to every reachable word."""
    dist = {src: 0}
    q = deque([src])
    while q:
        w = q.popleft()
        for n in adj[w]:
            if n not in dist:
                dist[n] = dist[w] + 1
                q.append(n)
    return dist


def shortest_path(adj, src, dst):
    """One shortest ladder src -> dst, or None if disconnected."""
    if src == dst:
        return [src]
    parent = {src: None}
    q = deque([src])
    while q:
        w = q.popleft()
        for n in adj[w]:
            if n in parent:
                continue
            parent[n] = w
            if n == dst:
                path = [n]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]
            q.append(n)
    return None


def components(adj):
    """Connected components, biggest first."""
    seen = set()
    comps = []
    for w in adj:
        if w not in seen:
            comp = set(bfs_dists(adj, w))
            seen |= comp
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps
