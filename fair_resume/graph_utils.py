"""Token dependency graph G=(V,E) and BFS/DFS context inspection.

Implements Algorithm 1 of the research paper (Sec. V-E/G). Proof-of-concept:
traversal is for human-auditable CONTEXT REVIEW only. Graph measures (degree)
are never used as a demographic or employability score, and traversal does not
decide which attributes are protected.
"""
from collections import deque, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


class DependencyGraph:
    """Nodes = token indices 0..n-1; edges = (head, dependent, label).

    Adjacency-list storage -> O(|V|+|E|) space; traversals are O(|V|+|E|) time.
    Neighbours are treated as undirected for context inspection (a token's
    head AND its dependents are both context)."""

    def __init__(self, tokens: Sequence[str], edges: Iterable[Tuple[int, int, str]]):
        self.tokens = list(tokens)
        self.edges = [(h, d, l) for h, d, l in edges if h != d]
        n = len(self.tokens)
        self._adj: Dict[int, List[int]] = defaultdict(list)
        for h, d, _ in self.edges:
            if not (0 <= h < n and 0 <= d < n):
                raise ValueError(f"edge ({h},{d}) outside token range 0..{n-1}")
            if d not in self._adj[h]:
                self._adj[h].append(d)
            if h not in self._adj[d]:
                self._adj[d].append(h)

    def __len__(self):
        return len(self.tokens)

    def neighbors(self, u: int) -> List[int]:
        return list(self._adj.get(u, []))

    # -- Algorithm 1 (BFS variant) ------------------------------------------
    def bfs(self, seeds: Iterable[int], max_depth: int = None) -> List[Tuple[int, int]]:
        """Return [(node, depth)] in visit order. Cycle-safe via visited set.
        max_depth=None -> whole connected component; else bounded neighbourhood."""
        visited, order = set(), []
        q = deque((s, 0) for s in seeds)
        while q:
            u, d = q.popleft()
            if u in visited:
                continue
            visited.add(u)
            order.append((u, d))
            if max_depth is not None and d >= max_depth:
                continue
            for v in self._adj.get(u, []):
                if v not in visited:
                    q.append((v, d + 1))
        return order

    # -- Algorithm 1 (DFS variant) ------------------------------------------
    def dfs(self, seeds: Iterable[int]) -> List[int]:
        """Iterative DFS (no recursion limit issues). Cycle-safe."""
        visited, order = set(), []
        stack = list(reversed(list(seeds)))
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            order.append(u)
            for v in reversed(self._adj.get(u, [])):
                if v not in visited:
                    stack.append(v)
        return order

    def bounded_neighborhood(self, seeds: Iterable[int], depth: int = 1) -> List[int]:
        """Context tokens within `depth` hops of the seeds, excluding the seeds.
        Uses BFS (a depth-limited DFS with a visited set can under-explore)."""
        seeds = list(seeds)
        s = set(seeds)
        return sorted(n for n, _ in self.bfs(seeds, max_depth=depth) if n not in s)

    def degree_summary(self) -> Dict[int, int]:
        """Per-token degree, for context inspection only (NOT a score)."""
        return {i: len(self._adj.get(i, [])) for i in range(len(self.tokens))}


# Synthetic sentence from IPR-I Fig. 3.2:
# "Alex built Python APIs in Delhi and she led testing."
SYNTHETIC_TOKENS = ["Alex", "built", "Python", "APIs", "in", "Delhi", "and",
                    "she", "led", "testing"]
SYNTHETIC_EDGES = [
    (1, 0, "nsubj"), (1, 3, "dobj"), (3, 2, "compound"), (1, 4, "prep"),
    (4, 5, "pobj"), (1, 6, "cc"), (1, 8, "conj"), (8, 7, "nsubj"),
    (8, 9, "dobj"),
]


def synthetic_graph() -> DependencyGraph:
    return DependencyGraph(SYNTHETIC_TOKENS, SYNTHETIC_EDGES)
