"""GraphContext — per-graph state: adjacency, neighbor pools, Laplacian PE cache.

Replaces the graph-state portion of SpatialRetriever and STALLM._node_pe_cache.
Not an nn.Module — pure data + algorithms owned by STALLM.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Int, jaxtyped
from numpy.typing import NDArray
from torch import Tensor

from model.node_pe import laplacian_eigvecs


class GraphContext:
    """Holds one graph's adjacency matrix, neighbour pools, and raw Laplacian eigenvectors.

    NodePE projection lives on SpatialReasoner (nn.Module on GPU); GraphContext
    only computes raw eigvecs on CPU and manages neighbour pools.
    """

    @beartype
    def __init__(
        self,
        adj_mx,
        n_aux: int,
        neighbor_order: str = "topological",
        neighbor_fill: str = "higher_order",
        pe_k: int = 16,
        node_pe_enabled: bool = True,
    ):
        if neighbor_order not in {"index", "topological"}:
            raise ValueError(f"unknown neighbor_order: {neighbor_order}")
        if neighbor_fill not in {"repeat_1hop", "higher_order"}:
            raise ValueError(f"unknown neighbor_fill: {neighbor_fill}")

        self.n_aux = n_aux
        self.neighbor_order = neighbor_order
        self.neighbor_fill = neighbor_fill

        self._adj_mx = torch.as_tensor(adj_mx, dtype=torch.float32)
        self._aux_pools: Optional[Dict[int, NDArray[np.int64]]] = None
        self._aux_ids_tensor: Optional[Tensor] = None
        self._topo_rank: Optional[NDArray[np.int64]] = None

        # PE — raw eigenvectors only; projection is done by SpatialReasoner
        self._pe_k = pe_k
        self._node_pe_enabled = node_pe_enabled and pe_k > 0
        self._raw_eigvecs: Optional[Tensor] = None

    @jaxtyped(typechecker=beartype)
    def sample_neighbors(self, node_ids: Int[Tensor, "C"], device: torch.device) -> Int[Tensor, "C K"]:
        """Return (len(node_ids), n_aux) auxiliary-node IDs for each target node."""
        all_aux = self.aux_ids_tensor(device)
        return all_aux[node_ids]

    def raw_eigvecs(self) -> Optional[Tensor]:
        """Return raw Laplacian eigenvectors (CPU tensor), or None if PE disabled.

        Projection to embedding space is done by SpatialReasoner's NodePE.
        """
        if not self._node_pe_enabled:
            return None
        if self._raw_eigvecs is None:
            self._raw_eigvecs = laplacian_eigvecs(self._adj_mx, self._pe_k)
        return self._raw_eigvecs

    @beartype
    def resample(self) -> Dict[int, NDArray[np.int64]]:
        pools = self._neighbor_id_pools()
        self._aux_pools = pools
        self._aux_ids_tensor = None
        return pools

    @beartype
    def set_pools(self, pools: Dict[int, NDArray[np.int64]]) -> None:
        self._aux_pools = {node_id: np.asarray(pool, dtype=np.int64) for node_id, pool in pools.items()}
        self._aux_ids_tensor = None

    @beartype
    def sample_sets(self, n_sets: int, seed: Optional[int] = None) -> List[Dict[int, NDArray[np.int64]]]:
        _ = n_sets, seed
        pools = self._neighbor_id_pools()
        return [{node_id: pool.copy() for node_id, pool in pools.items()} for _ in range(n_sets)]

    @jaxtyped(typechecker=beartype)
    def aux_ids_tensor(self, device: torch.device) -> Int[Tensor, "N K"]:
        pools = self._aux_pools
        if pools is None:
            self.resample()
            pools = self._aux_pools
            assert pools is not None
        cached = self._aux_ids_tensor
        if cached is None or cached.device != device:
            N = len(pools)
            ids = torch.zeros(N, self.n_aux, dtype=torch.long)
            for i in range(N):
                ids[i] = torch.as_tensor(pools[i], dtype=torch.long)
            cached = ids.to(device)
            self._aux_ids_tensor = cached
        return cached

    def invalidate_pe(self) -> None:
        """Drop cached eigenvectors (call when adjacency changes)."""
        self._raw_eigvecs = None

    def _adjacency(self) -> NDArray[np.bool_]:
        A = np.asarray(self._adj_mx.cpu().numpy())
        return np.maximum(A, A.T) > 0

    def _directed_adjacency(self) -> NDArray[np.bool_]:
        A = np.asarray(self._adj_mx.cpu().numpy())
        directed = A > 0
        np.fill_diagonal(directed, False)
        return directed

    def _topological_rank(self) -> NDArray[np.int64]:
        if self._topo_rank is not None:
            return self._topo_rank

        directed = self._directed_adjacency()
        N = directed.shape[0]
        graph = [np.flatnonzero(directed[i]).tolist() for i in range(N)]

        index = 0
        stack: List[int] = []
        on_stack = [False] * N
        indices = [-1] * N
        lowlink = [0] * N
        components: List[List[int]] = []

        def strongconnect(v: int) -> None:
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack[v] = True

            for w in graph[v]:
                if indices[w] == -1:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                components.append(sorted(component))

        for node in range(N):
            if indices[node] == -1:
                strongconnect(node)

        comp_id = np.zeros(N, dtype=np.int64)
        for cid, component in enumerate(components):
            for node in component:
                comp_id[node] = cid

        comp_edges = {cid: set() for cid in range(len(components))}
        indegree = {cid: 0 for cid in range(len(components))}
        for src in range(N):
            src_comp = int(comp_id[src])
            for dst in graph[src]:
                dst_comp = int(comp_id[dst])
                if src_comp != dst_comp and dst_comp not in comp_edges[src_comp]:
                    comp_edges[src_comp].add(dst_comp)
                    indegree[dst_comp] += 1

        comp_min_node = {cid: min(component) for cid, component in enumerate(components)}
        ready = sorted([cid for cid, deg in indegree.items() if deg == 0], key=lambda cid: comp_min_node[cid])
        comp_rank = {}
        rank_value = 0
        while ready:
            cid = ready.pop(0)
            comp_rank[cid] = rank_value
            rank_value += 1
            for nxt in sorted(comp_edges[cid], key=lambda item: comp_min_node[item]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
                    ready.sort(key=lambda item: comp_min_node[item])

        rank = np.asarray([comp_rank[int(comp_id[node])] for node in range(N)], dtype=np.int64)
        self._topo_rank = rank
        return rank

    def _order_neighbors(self, node_id: int, neighbors: NDArray[np.int64]) -> NDArray[np.int64]:
        if self.neighbor_order == "index":
            return np.sort(neighbors)
        topo_rank = self._topological_rank()
        return np.asarray(
            sorted(neighbors.tolist(), key=lambda x: (int(topo_rank[int(x)]), int(x))),
            dtype=np.int64,
        )

    def _k_hop_neighbors(self, start: int, hops: int, adj: NDArray[np.bool_]) -> NDArray[np.int64]:
        visited = {start}
        frontier = {start}
        current = set()
        for _ in range(hops):
            current = set()
            for node in frontier:
                current.update(np.flatnonzero(adj[node]).tolist())
            current -= visited
            visited.update(current)
            frontier = current
        return np.asarray(sorted(current), dtype=np.int64)

    def _fill_neighbors(self, node_id: int, one_hop: NDArray[np.int64], adj: NDArray[np.bool_]) -> NDArray[np.int64]:
        if one_hop.size >= self.n_aux:
            return one_hop[: self.n_aux]

        if self.neighbor_fill == "repeat_1hop":
            repeat_count = (self.n_aux + one_hop.size - 1) // one_hop.size
            return np.tile(one_hop, repeat_count)[: self.n_aux]

        pieces = [one_hop]
        seen = set(one_hop.tolist()) | {node_id}
        for hops in (2, 3):
            if sum(len(piece) for piece in pieces) >= self.n_aux:
                break
            hop_nodes = self._k_hop_neighbors(node_id, hops, adj)
            hop_nodes = np.asarray([x for x in hop_nodes.tolist() if x not in seen], dtype=np.int64)
            hop_nodes = self._order_neighbors(node_id, hop_nodes)
            if hop_nodes.size == 0:
                continue
            pieces.append(hop_nodes)
            seen.update(hop_nodes.tolist())

        combined = np.concatenate(pieces) if pieces else one_hop
        if combined.size >= self.n_aux:
            return combined[: self.n_aux]
        repeat_count = (self.n_aux + combined.size - 1) // combined.size
        return np.tile(combined, repeat_count)[: self.n_aux]

    def _neighbor_id_pools(self) -> Dict[int, NDArray[np.int64]]:
        adj = self._adjacency()
        N = adj.shape[0]
        pools: Dict[int, NDArray[np.int64]] = {}

        for i in range(N):
            neighbors = np.flatnonzero(adj[i]).astype(np.int64)
            neighbors = neighbors[neighbors != i]
            if neighbors.size == 0:
                raise ValueError(f"node {i} has no graph neighbors")
            ordered = self._order_neighbors(i, neighbors)
            pools[i] = self._fill_neighbors(i, ordered, adj)

        return pools
