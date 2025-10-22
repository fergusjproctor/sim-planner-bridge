#!/usr/bin/env python3
"""
plot_sg.py

Render a GRID-style scene-graph JSON to a radial node-link diagram.

Input format expected (our sg JSON):
{
  "version": "1.1",
  "nodes": [
      {"id": 0, "type": "floor", "attributes": {"label": "floor", "color": ""}, "operation": []},
      {"id": 1, "type": "room", "attributes": {"label": "kitchen", "color": ""}, "operation": []},
      {"id": 2, "type": "large_object", "attributes": {"label": "table", "color": ""}, "operation": ["place_to"]},
      ...
  ],
  "edges": [
      {"from_id": 0, "to_id": 1, "relation_type": "in"},
      {"from_id": 1, "to_id": 2, "relation_type": "in"},
      ...
  ]
}

Usage:
  python scripts/plot_sg.py \
      --input /path/to/scene.sg.json \
      --out /path/to/scene_graph.png \
      --title "scene graph" \
      --no-edge-labels
"""

import argparse
import json
import math
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from matplotlib.colors import to_hex

# ------------------------
# Helpers
# ------------------------

def _label(node):
    attrs = node.get("attributes", {}) or {}
    lbl = attrs.get("label") or node.get("type", "node")
    return f"{lbl}{node['id']}"  # e.g., "banana40"

def _child_count_map(edges):
    count = defaultdict(int)
    for e in edges:
        if e.get("relation_type", "").lower() == "in":
            count[e["from_id"]] += 1
    return count

def _build_graph(sg):
    G = nx.DiGraph()
    id2node = {n["id"]: n for n in sg.get("nodes", [])}
    for n in sg.get("nodes", []):
        G.add_node(n["id"], **n)  # stash full node info on the networkx node

    for e in sg.get("edges", []):
        G.add_edge(e["from_id"], e["to_id"], **e)

    return G, id2node

def _find_floor_id(G):
    for nid, data in G.nodes(data=True):
        if data.get("type") == "floor":
            return nid
    # fallback: smallest id
    return min(G.nodes)

def _depths_from_floor(G, floor_id):
    """Compute tree-like depths following 'in' edges from floor->room->object."""
    depths = {floor_id: 0}
    q = deque([floor_id])
    while q:
        u = q.popleft()
        for _, v, edata in G.out_edges(u, data=True):
            if edata.get("relation_type", "").lower() != "in":
                continue
            if v not in depths:
                depths[v] = depths[u] + 1
                q.append(v)
    # any orphan nodes get max depth+1
    if len(depths) < G.number_of_nodes():
        maxd = max(depths.values())
        for nid in G.nodes:
            if nid not in depths:
                depths[nid] = maxd + 1
    return depths

def _shell_layout(G, depths):
    """Radial shells by depth using networkx shell_layout."""
    shells_map = defaultdict(list)
    for nid, d in depths.items():
        shells_map[d].append(nid)
    shells = [shells_map[d] for d in sorted(shells_map)]
    pos = nx.shell_layout(G, nlist=shells, rotate=0)  # dict: node->(x,y)
    # scale outward a bit as depth increases to avoid overlap
    for nid, (x, y) in pos.items():
        r = math.sqrt(x*x + y*y)
        d = depths[nid]
        scale = 1.0 + 0.25 * d
        pos[nid] = (x * scale, y * scale)
    return pos

def _palette_for_objects(names):
    """Stable color assignment using tab20, based on hashed labels."""
    tab = [to_hex(cm.tab20(i)) for i in range(20)]
    color_map = {}
    for name in sorted(set(names)):
        idx = abs(hash(name)) % len(tab)
        color_map[name] = tab[idx]
    return color_map

def _node_styles(G, child_counts):
    labels = {}
    colors = {}
    sizes = {}
    textcolors = {}
    obj_names = []

    for nid, data in G.nodes(data=True):
        lbl = _label(data)
        labels[nid] = lbl
        t = data.get("type", "")

        if t == "floor":
            colors[nid] = "#9e9e9e"
            sizes[nid] = 1800
            textcolors[nid] = "white"
        elif t == "room":
            colors[nid] = "#bdbdbd"
            sizes[nid] = 1400
            textcolors[nid] = "black"
        elif t == "large_object":
            obj_names.append(lbl)
            sizes[nid] = 1000
            textcolors[nid] = "white"
        else:
            # small_object or anything else
            obj_names.append(lbl)
            sizes[nid] = 900
            textcolors[nid] = "white"

    # assign object colors
    obj_palette = _palette_for_objects(obj_names)
    for nid, data in G.nodes(data=True):
        if data.get("type") in ("large_object", "small_object"):
            colors[nid] = obj_palette[_label(data)]

    # degree override: make very high fan-out a little bigger
    for nid in G.nodes:
        k = child_counts.get(nid, 0)
        if k >= 6:
            sizes[nid] *= 1.2
        if k >= 12:
            sizes[nid] *= 1.3

    return labels, colors, sizes, textcolors

# ------------------------
# Main render function
# ------------------------

def render_scene_graph(sg, out_path, title=None, show_edge_labels=True, dpi=220):
    G, id2node = _build_graph(sg)
    floor_id = _find_floor_id(G)
    depths = _depths_from_floor(G, floor_id)
    pos = _shell_layout(G, depths)
    child_counts = _child_count_map(sg.get("edges", []))
    labels, colors, sizes, textcolors = _node_styles(G, child_counts)

    plt.figure(figsize=(10, 10), dpi=dpi)
    ax = plt.gca()
    ax.set_axis_off()

    # Draw edges first
    edge_colors = []
    for u, v, edata in G.edges(data=True):
        rt = (edata.get("relation_type") or "").lower()
        if rt == "in":
            edge_colors.append("#666666")
        elif rt == "on":
            edge_colors.append("#4444aa")
        else:
            edge_colors.append("#999999")
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color=edge_colors, alpha=0.8, arrows=False)

    # Draw nodes
    node_color_list = [colors[n] for n in G.nodes()]
    node_size_list = [sizes[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color_list,
                           node_size=node_size_list,
                           linewidths=1.5,
                           edgecolors="#303030",
                           alpha=0.95)

    # Node labels on top (object name+id)
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=7.5,
        font_color="white",
        font_weight="bold"
    )

    # Child counts in the center of each node (small, like your screenshot)
    count_labels = {nid: str(child_counts.get(nid, 0)) for nid in G.nodes()}
    # draw in a secondary position slightly inward
    nx.draw_networkx_labels(
        G, {k: (x, y - 0.03) for k, (x, y) in pos.items()},
        labels=count_labels,
        font_size=6,
        font_color="#000000",
        font_weight="bold"
    )

    # Edge labels
    if show_edge_labels:
        elabs = {(u, v): (edata.get("relation_type") or "").lower()
                 for u, v, edata in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=elabs, font_size=6, label_pos=0.5, rotate=False)

    if title:
        plt.title(title, fontsize=11, pad=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"[OK] saved graph to {out_path}")

# ------------------------
# CLI
# ------------------------

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