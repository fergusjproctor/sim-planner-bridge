#!/usr/bin/env python3
"""
plot_sg.py — radial hierarchical scene-graph plotter (no 'in' edge crossings)

- Subtree-based wedge layout => 'in' hierarchy branches cleanly from the center
- Curved 'on' edges to reduce visual collisions
- Same styling as before

Usage:
  python scripts/plot_sg.py --input scene.sg.json --out scene_graph.png --title "scene"
"""

import argparse
import json
import math
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from matplotlib.colors import to_hex

# ---------- helpers ----------

def _label(node):
    attrs = node.get("attributes", {}) or {}
    lbl = attrs.get("label") or node.get("type", "node")
    return f"{lbl}{node['id']}"

def _child_count_map(edges):
    count = defaultdict(int)
    for e in edges:
        if (e.get("relation_type") or "").lower() == "in":
            count[e["from_id"]] += 1
    return count

def _build_graph(sg):
    G = nx.DiGraph()
    for n in sg.get("nodes", []):
        G.add_node(n["id"], **n)
    for e in sg.get("edges", []):
        G.add_edge(e["from_id"], e["to_id"], **e)
    return G

def _find_floor_id(G):
    for nid, data in G.nodes(data=True):
        if data.get("type") == "floor":
            return nid
    return min(G.nodes)

def _depths_from_floor(G, floor_id):
    depths = {floor_id: 0}
    q = deque([floor_id])
    while q:
        u = q.popleft()
        for _, v, ed in G.out_edges(u, data=True):
            if (ed.get("relation_type") or "").lower() != "in":
                continue
            if v not in depths:
                depths[v] = depths[u] + 1
                q.append(v)
    if len(depths) < G.number_of_nodes():
        maxd = max(depths.values())
        for nid in G.nodes:
            if nid not in depths:
                depths[nid] = maxd + 1
    return depths

def _from_polar(r, theta):
    return (r * math.cos(theta), r * math.sin(theta))

def _hierarchical_radial_layout(G, depths, root,
                                base_radius=0.15,     # put root near center
                                dr=1.0,               # radial gap between levels
                                wedge_padding=0.06,   # angular padding between sibling wedges (radians)
                                min_wedge=0.04):      # minimum wedge width (radians)
    """Planar layout for 'in' edges: allocate disjoint angular wedges to subtrees."""
    # Build 'in' tree relations
    children = defaultdict(list)
    for u, v, ed in G.edges(data=True):
        if (ed.get("relation_type") or "").lower() == "in":
            children[u].append(v)

    # stable order by label to keep diagrams consistent
    def name(nid):
        n = G.nodes[nid]
        a = (n.get("attributes") or {})
        return (a.get("label") or n.get("type") or "").lower()
    for u in children:
        children[u] = sorted(children[u], key=name)

    # subtree sizes
    memo = {}
    def subsize(u):
        if u in memo: return memo[u]
        s = 1
        for c in children.get(u, []):
            s += subsize(c)
        memo[u] = s
        return s
    for n in G.nodes: subsize(n)

    pos = {}

    def place(u, th0, th1):
        # ensure minimum angular size
        if th1 - th0 < min_wedge:
            mid = 0.5 * (th0 + th1)
            th0, th1 = mid - min_wedge/2, mid + min_wedge/2

        theta = 0.5 * (th0 + th1)
        r = base_radius + depths[u] * dr
        pos[u] = _from_polar(r, theta)

        kids = children.get(u, [])
        if not kids: return

        total = sum(memo[k] for k in kids)
        span = max(th1 - th0 - wedge_padding * max(len(kids)-1, 0), min_wedge)
        cursor = th0
        # distribute wedges by subtree size, with padding between siblings
        for i, k in enumerate(kids):
            frac = memo[k] / total if total > 0 else 1.0 / len(kids)
            w = max(span * frac, min_wedge)
            left = cursor
            right = cursor + w
            place(k, left, right)
            cursor = right + wedge_padding

    # allocate full circle for the root
    place(root, 0.0, 2.0 * math.pi)

    # any orphan nodes (not reachable via 'in'): place evenly on their depth ring
    unplaced = [n for n in G.nodes if n not in pos]
    by_depth = defaultdict(list)
    for n in unplaced:
        by_depth[depths[n]].append(n)
    for d, nodes in by_depth.items():
        m = max(1, len(nodes))
        r = base_radius + d * dr
        for i, n in enumerate(sorted(nodes, key=name)):
            th = 2.0 * math.pi * (i / m)
            pos[n] = _from_polar(r, th)

    return pos

def _palette_for_objects(names):
    tab = [to_hex(cm.tab20(i)) for i in range(20)]
    m = {}
    for name in sorted(set(names)):
        m[name] = tab[abs(hash(name)) % len(tab)]
    return m

def _node_styles(G, child_counts):
    labels, colors, sizes, textcolors, obj_names = {}, {}, {}, {}, []
    for nid, data in G.nodes(data=True):
        lbl = _label(data)
        labels[nid] = lbl
        t = data.get("type", "")
        if t == "floor":
            colors[nid] = "#9e9e9e"; sizes[nid] = 1800; textcolors[nid] = "white"
        elif t == "room":
            colors[nid] = "#bdbdbd"; sizes[nid] = 1400; textcolors[nid] = "black"
        elif t == "large_object":
            obj_names.append(lbl); sizes[nid] = 1000; textcolors[nid] = "white"
        else:
            obj_names.append(lbl); sizes[nid] = 900; textcolors[nid] = "white"
    pal = _palette_for_objects(obj_names)
    for nid, data in G.nodes(data=True):
        if data.get("type") in ("large_object", "small_object"):
            colors[nid] = pal[_label(data)]
    for nid in G.nodes:
        k = child_counts.get(nid, 0)
        if k >= 6: sizes[nid] *= 1.2
        if k >= 12: sizes[nid] *= 1.3
    return labels, colors, sizes, textcolors

# ---------- main render ----------

def render_scene_graph(sg, out_path, title=None, show_edge_labels=True, dpi=220):
    G = _build_graph(sg)
    floor_id = _find_floor_id(G)
    depths = _depths_from_floor(G, floor_id)
    pos = _hierarchical_radial_layout(G, depths, root=floor_id, base_radius=0.15, dr=1.05,
                                      wedge_padding=0.07, min_wedge=0.045)
    child_counts = _child_count_map(sg.get("edges", []))
    labels, colors, sizes, textcolors = _node_styles(G, child_counts)

    plt.figure(figsize=(10, 10), dpi=dpi)
    ax = plt.gca(); ax.set_axis_off()

    # Draw 'in' (hierarchy) edges first: straight lines, planar with this layout
    in_edges = [(u, v) for u, v, ed in G.edges(data=True)
                if (ed.get("relation_type") or "").lower() == "in"]
    nx.draw_networkx_edges(G, pos, edgelist=in_edges, width=1.5, edge_color="#666666", alpha=0.9, arrows=False)

    # Draw 'on' edges: gentle arcs to avoid slicing across branches
    on_edges = [(u, v) for u, v, ed in G.edges(data=True)
                if (ed.get("relation_type") or "").lower() == "on"]
    if on_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=on_edges, width=1.2, edge_color="#4444aa", alpha=0.75, arrows=False,
            connectionstyle="arc3,rad=0.20"
        )

    # Any other relation edges (rare)
    other_edges = [(u, v) for u, v, ed in G.edges(data=True)
                   if (ed.get("relation_type") or "").lower() not in ("in", "on")]
    if other_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=other_edges, width=1.2, edge_color="#999999", alpha=0.7, arrows=False,
            connectionstyle="arc3,rad=0.15"
        )

    # Nodes
    node_color_list = [colors[n] for n in G.nodes()]
    node_size_list = [sizes[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color_list,
                           node_size=node_size_list,
                           linewidths=1.5,
                           edgecolors="#303030",
                           alpha=0.96)

    # Node labels (name+id)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7.5, font_color="white", font_weight="bold")

    # Child counts (tiny black number slightly below)
    count_labels = {nid: str(child_counts.get(nid, 0)) for nid in G.nodes()}
    nx.draw_networkx_labels(
        G, {k: (x, y - 0.035) for k, (x, y) in pos.items()},
        labels=count_labels, font_size=6, font_color="#000000", font_weight="bold"
    )

    # Edge labels (optional)
    if show_edge_labels:
        elabs = {(u, v): (ed.get("relation_type") or "").lower() for u, v, ed in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=elabs, font_size=6, label_pos=0.5, rotate=False)

    if title: plt.title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"[OK] saved graph to {out_path}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to scene graph JSON (SG)")
    ap.add_argument("--out", required=True, help="Output image path (e.g., scene_graph.png)")
    ap.add_argument("--title", default=None, help="Figure title")
    ap.add_argument("--no-edge-labels", action="store_true", help="Hide edge labels")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        sg = json.load(f)

    render_scene_graph(
        sg,
        out_path=args.out,
        title=args.title,
        show_edge_labels=not args.no_edge_labels
    )

if __name__ == "__main__":
    main()