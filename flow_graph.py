"""
P&ID flow-direction pipeline (page 61)
========================================
Mirrors the ISE blog's "Graph Construction" + "Graph Traversal / momentum
propagation" steps, adapted to the three JSON files actually available:

  - page_61.json                 -> off-page connectors (SEED direction: inflow/outflow)
  - line_hough_transform_61.json -> raw Hough line segments (NO direction, fragmented)
  - page_61_valve.json           -> valves (topology nodes only, normalized bbox)

Pipeline:
  1. Merge fragmented Hough segments into logical pipe runs (horizontal / vertical).
  2. Build a topology graph: pipe-run endpoints <-> connectors/valves <-> other
     pipe-run endpoints, via proximity matching (with a buffer, like ISE's
     "extend lines by small buffer" step).
  3. Seed direction at connector nodes from their inflow/outflow field.
  4. BFS "momentum propagation": push direction from each seeded connector
     out along connected pipe runs until the frontier runs out or hits a
     conflicting seed (flagged, not overwritten).
"""
import json
import math
from collections import defaultdict, deque

IMG_W, IMG_H = 12600, 9000

# ---- tunable thresholds -----------------------------------------------
Y_TOL = 6          # px, how close two horizontal segments' y must be to be "the same line"
X_TOL = 6          # px, same for vertical segments' x
GAP_SOLID = 90     # px, max gap to bridge between solid segments on the same line
GAP_DOTTED = 260   # px, dotted/dashed lines have larger gaps between dashes
NODE_SNAP_DIST = 140  # px, max distance from a pipe-run endpoint to a symbol center
                      # to consider them connected (accounts for stub lines / leaders)
ANGLE_TOL_DEG = 3.5


def load_data(conn_path, line_path, valve_path):
    connectors = json.load(open(conn_path))["connectors"]
    lines = json.load(open(line_path))
    valves = json.load(open(valve_path))
    return connectors, lines, valves


def classify_orientation(angle):
    a = angle % 180
    if a <= ANGLE_TOL_DEG or a >= 180 - ANGLE_TOL_DEG:
        return "h"
    if abs(a - 90) <= ANGLE_TOL_DEG:
        return "v"
    return "d"  # diagonal - left ungrouped, handled as its own run


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def merge_axis_aligned(lines, orientation):
    """Merge collinear same-orientation segments that are close enough to be
    the same physical pipe run (handles dashes + small Hough fragmentation)."""
    idxs = [i for i, l in enumerate(lines) if classify_orientation(l["angle"]) == orientation]
    if not idxs:
        return []

    uf = UnionFind(len(idxs))
    # bucket by rounded cross-axis coordinate to cut down comparisons
    buckets = defaultdict(list)
    for pos, i in enumerate(idxs):
        l = lines[i]
        cross = l["center"]["y"] if orientation == "h" else l["center"]["x"]
        buckets[round(cross / 10)].append(pos)

    def gap_tol(l):
        return GAP_DOTTED if l["line_type"] == "dotted" else GAP_SOLID

    for pos_a, i in enumerate(idxs):
        la = lines[i]
        cross_a = la["center"]["y"] if orientation == "h" else la["center"]["x"]
        bucket_key = round(cross_a / 10)
        candidates = []
        for k in (bucket_key - 1, bucket_key, bucket_key + 1):
            candidates.extend(buckets.get(k, []))
        for pos_b in candidates:
            if pos_b <= pos_a:
                continue
            j = idxs[pos_b]
            lb = lines[j]
            if la["line_type"] != lb["line_type"]:
                continue
            cross_b = lb["center"]["y"] if orientation == "h" else lb["center"]["x"]
            tol = Y_TOL if orientation == "h" else X_TOL
            if abs(cross_a - cross_b) > tol:
                continue
            # check along-axis gap
            if orientation == "h":
                a_min, a_max = sorted([la["start"]["x"], la["end"]["x"]])
                b_min, b_max = sorted([lb["start"]["x"], lb["end"]["x"]])
            else:
                a_min, a_max = sorted([la["start"]["y"], la["end"]["y"]])
                b_min, b_max = sorted([lb["start"]["y"], lb["end"]["y"]])
            gap = max(0, max(a_min, b_min) - min(a_max, b_max))
            if gap <= max(gap_tol(la), gap_tol(lb)):
                uf.union(pos_a, pos_b)

    groups = defaultdict(list)
    for pos, i in enumerate(idxs):
        groups[uf.find(pos)].append(i)

    merged = []
    for group in groups.values():
        members = [lines[i] for i in group]
        if orientation == "h":
            cross = sum(m["center"]["y"] for m in members) / len(members)
            xs = [m["start"]["x"] for m in members] + [m["end"]["x"] for m in members]
            x0, x1 = min(xs), max(xs)
            start, end = {"x": x0, "y": cross}, {"x": x1, "y": cross}
        else:
            cross = sum(m["center"]["x"] for m in members) / len(members)
            ys = [m["start"]["y"] for m in members] + [m["end"]["y"] for m in members]
            y0, y1 = min(ys), max(ys)
            start, end = {"x": cross, "y": y0}, {"x": cross, "y": y1}
        types = set(m["line_type"] for m in members)
        merged.append({
            "orientation": orientation,
            "line_type": types.pop() if len(types) == 1 else "mixed",
            "start": start,
            "end": end,
            "length": math.hypot(end["x"] - start["x"], end["y"] - start["y"]),
            "member_count": len(members),
            "member_ids": [m["id"] for m in members],
        })
    return merged


def merge_diagonal(lines):
    """Diagonal segments are rare here; keep them as individual runs."""
    out = []
    for l in lines:
        if classify_orientation(l["angle"]) != "d":
            continue
        out.append({
            "orientation": "d",
            "line_type": l["line_type"],
            "start": l["start"],
            "end": l["end"],
            "length": l["length"],
            "member_count": 1,
            "member_ids": [l["id"]],
        })
    return out


def build_pipe_runs(lines):
    runs = merge_axis_aligned(lines, "h") + merge_axis_aligned(lines, "v") + merge_diagonal(lines)
    for idx, r in enumerate(runs):
        r["run_id"] = f"run_{idx}"
    return runs


def build_symbol_nodes(connectors, valves):
    nodes = []
    for c in connectors:
        x, y, w, h = c["bbox_xywh"]
        cx, cy = x + w / 2, y + h / 2
        nodes.append({
            "node_id": f"connector_{c['id']}",
            "type": "connector",
            "x": cx, "y": cy,
            "reference_tag": c.get("reference_tag"),
            "seed_direction": c.get("direction"),          # inflow / outflow
            "seed_confidence": c.get("confidence"),
            "nearest_edge": c.get("nearest_edge"),
        })
    for i, v in enumerate(valves):
        bx, by, bw, bh = v["bbox_xywh"]
        cx, cy = (bx + bw / 2) * IMG_W, (by + bh / 2) * IMG_H
        nodes.append({
            "node_id": f"valve_{i}",
            "type": v["class_name"].replace(" ", "_"),
            "x": cx, "y": cy,
            "confidence": v["confidence"],
        })
    return nodes


def dist(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def build_graph(runs, symbol_nodes):
    """
    Graph nodes: every pipe run endpoint gets matched to the nearest symbol
    node within NODE_SNAP_DIST (proximity matching), or, failing that, to
    other pipe-run endpoints that land close together (a junction with no
    symbol on it, e.g. a tee formed purely by two lines touching).
    Returns: adjacency dict node_id -> list of (neighbor_run_id, run) and
    a lookup of run_id -> [endpoint_node_id_A, endpoint_node_id_B]
    """
    endpoints = []  # (run_idx, which_end 'start'/'end', x, y)
    for ridx, r in enumerate(runs):
        endpoints.append((ridx, "start", r["start"]["x"], r["start"]["y"]))
        endpoints.append((ridx, "end", r["end"]["x"], r["end"]["y"]))

    # match each endpoint to nearest symbol node (if within snap distance)
    endpoint_node = {}  # (ridx, which_end) -> node_id
    for ridx, which, ex, ey in endpoints:
        best, best_d = None, NODE_SNAP_DIST
        for n in symbol_nodes:
            d = dist(ex, ey, n["x"], n["y"])
            if d < best_d:
                best, best_d = n["node_id"], d
        if best:
            endpoint_node[(ridx, which)] = best

    # remaining unmatched endpoints: cluster mutually-close ones into junction nodes
    unmatched = [(ridx, which, ex, ey) for ridx, which, ex, ey in endpoints
                 if (ridx, which) not in endpoint_node]
    uf = UnionFind(len(unmatched))
    for a in range(len(unmatched)):
        for b in range(a + 1, len(unmatched)):
            if dist(unmatched[a][2], unmatched[a][3], unmatched[b][2], unmatched[b][3]) <= NODE_SNAP_DIST:
                uf.union(a, b)
    junction_groups = defaultdict(list)
    for i in range(len(unmatched)):
        junction_groups[uf.find(i)].append(i)
    for gi, group in enumerate(junction_groups.values()):
        jid = f"junction_{gi}"
        xs = [unmatched[i][2] for i in group]
        ys = [unmatched[i][3] for i in group]
        for i in group:
            ridx, which, _, _ = unmatched[i]
            endpoint_node[(ridx, which)] = jid

    run_endpoints = {r["run_id"]: (endpoint_node[(ridx, "start")], endpoint_node[(ridx, "end")])
                     for ridx, r in enumerate(runs)}

    adjacency = defaultdict(set)  # node_id -> set of run_ids touching it
    for ridx, r in enumerate(runs):
        n_start, n_end = run_endpoints[r["run_id"]]
        adjacency[n_start].add(r["run_id"])
        adjacency[n_end].add(r["run_id"])

    return run_endpoints, adjacency


def propagate_direction(runs, run_endpoints, adjacency, symbol_nodes):
    """
    BFS momentum propagation, seeded from connector inflow/outflow.
    For a run whose one endpoint is node A and other is node B:
      - if A is an 'outflow' connector -> flow goes B -> A (material leaves via A)
      - if A is an 'inflow' connector  -> flow goes A -> B (material enters via A)
    Propagate the resolved endpoint's direction across junctions to neighboring runs.
    """
    run_by_id = {r["run_id"]: r for r in runs}
    node_seed = {n["node_id"]: n["seed_direction"] for n in symbol_nodes if n.get("seed_direction")}

    flow = {}  # run_id -> (from_node, to_node) resolved direction
    resolved_node_flow = {}  # node_id -> 'source' (flow leaves this node into network) or 'sink'
    conflicts = []

    queue = deque()
    for node_id, seed in node_seed.items():
        resolved_node_flow[node_id] = "sink" if seed == "outflow" else "source"
        queue.append(node_id)

    visited_runs = set()
    while queue:
        node_id = queue.popleft()
        role = resolved_node_flow[node_id]  # 'source' -> flow leaves node into attached runs
        for run_id in adjacency.get(node_id, []):
            if run_id in visited_runs:
                continue
            n_start, n_end = run_endpoints[run_id]
            other = n_end if node_id == n_start else n_start
            if role == "source":
                frm, to = node_id, other
            else:  # sink: flow travels INTO node_id, i.e. FROM the other node
                frm, to = other, node_id
            if run_id in flow and flow[run_id] != (frm, to):
                conflicts.append(run_id)
                continue
            flow[run_id] = (frm, to)
            visited_runs.add(run_id)

            other_role = "source" if to == other else "sink"
            # the "other" node now has flow arriving into it (if to==other) => it becomes a source
            # for its OTHER attached runs, unless it already has a seeded role
            if other not in resolved_node_flow:
                resolved_node_flow[other] = "source" if to == other else "sink"
                queue.append(other)
            elif resolved_node_flow[other] != ("source" if to == other else "sink"):
                conflicts.append(f"node_conflict:{other}")

    unresolved = [r["run_id"] for r in runs if r["run_id"] not in flow]
    return flow, unresolved, conflicts


def main():
    connectors, lines, valves = load_data(
        "/mnt/user-data/uploads/page_61.json",
        "/mnt/user-data/uploads/line_hough_transform_61.json",
        "/mnt/user-data/uploads/page_61_valve.json",
    )
    runs = build_pipe_runs(lines)
    symbol_nodes = build_symbol_nodes(connectors, valves)
    run_endpoints, adjacency = build_graph(runs, symbol_nodes)
    flow, unresolved, conflicts = propagate_direction(runs, run_endpoints, adjacency, symbol_nodes)

    print(f"Raw Hough segments: {len(lines)}")
    print(f"Merged pipe runs:   {len(runs)}")
    print(f"Symbol nodes:       {len(symbol_nodes)} (24 connectors + {len(valves)} valves)")
    print(f"Runs with resolved direction: {len(flow)} / {len(runs)}")
    print(f"Unresolved runs (no path to a seeded connector): {len(unresolved)}")
    print(f"Conflicts flagged: {len(conflicts)}")

    node_lookup = {n["node_id"]: n for n in symbol_nodes}

    def describe(node_id):
        if node_id in node_lookup:
            n = node_lookup[node_id]
            tag = n.get("reference_tag") or n["type"]
            return f"{n['type']}:{tag}"
        return node_id

    print("\nSample resolved runs:")
    for run_id, (frm, to) in list(flow.items())[:15]:
        print(f"  {run_id}: {describe(frm)}  ->  {describe(to)}")

    out = {
        "runs": runs,
        "run_endpoints": run_endpoints,
        "symbol_nodes": symbol_nodes,
        "flow": {k: {"from": v[0], "to": v[1]} for k, v in flow.items()},
        "unresolved_run_ids": unresolved,
        "conflicts": conflicts,
    }
    with open("/home/claude/pnid/flow_graph_output.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()