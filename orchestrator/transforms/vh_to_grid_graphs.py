#!/usr/bin/env python3
"""
vh_to_grid_graphs.py

Convert a VirtualHome EnvironmentGraph JSON into:
  (1) GRID-style scene graph (sg)
  (2) GRID-style relation graph (rg) centered on the robot

Robust to:
- Categories like "Characters", "Rooms", "Floor", "Walls", "Ceiling"
- properties containing "SURFACES"
- files that look like:  true, { ...graph... }

Usage:
  python vh_to_grid_graphs.py \
      --input /path/to/vh_env_graph.json \
      --out-sg /path/to/out.sg.json \
      --out-rg /path/to/out.rg.json \
      [--robot-id 1] [--pretty]
"""

import argparse
import json
import re
from collections import defaultdict

# ---------- Normalization & heuristics ----------

def _norm(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

CATEGORY_MAP = {
    "characters": "person",
    "character":  "person",
    "person":     "person",
    "people":     "person",
    "rooms":      "room",
    "room":       "room",
    "floor":      "floor",
    "floors":     "floor",
    "walls":      "wall",
    "wall":       "wall",
    "ceiling":    "ceiling",
    "ceilings":   "ceiling",
}

LONGITUDINAL_HINTS = {"drawer", "dresser", "toolbox", "side table", "cabinet drawer"}
REVOLUTE_HINTS     = {"cabinet", "bookcase", "medicine cabinet", "storage cabinet",
                      "fridge", "refrigerator", "microwave", "oven", "wardrobe",
                      "closet", "cupboard"}
SURFACE_HINTS      = {"table", "desk", "counter", "shelf", "bookcase", "coffee table",
                      "nightstand", "side table", "dresser", "sink", "stove", "hob",
                      "bench", "island"}

def norm_category(raw):
    return CATEGORY_MAP.get(_norm(raw), _norm(raw))

def infer_mechanism_ops(label: str):
    name = _norm(label)
    if any(k in name for k in LONGITUDINAL_HINTS):
        return ["longitudinal_open", "longitudinal_close"]
    if any(k in name for k in REVOLUTE_HINTS):
        return ["revolute_open", "revolute_close"]
    return []

def infer_surface_ops(label: str, props_uc: set):
    name = _norm(label)
    ops = []
    if ("SURFACE" in props_uc) or ("SURFACES" in props_uc) or ("RECIPIENT" in props_uc) \
       or any(k in name for k in SURFACE_HINTS):
        ops.append("place_to")
    return ops

def infer_pick_ops(props_uc: set):
    return ["pick"] if "GRABBABLE" in props_uc else []

# ---------- Input parsing ----------

def load_vh_env_graph(path: str) -> dict:
    """
    Accepts:
      - plain dict JSON: {"nodes":[...],"edges":[...]}
      - list/tuple: [true, {"nodes":...}]
      - text starting with 'true,' followed by {...}
    """
    with open(path, "r") as f:
        raw_text = f.read()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # strip leading "true," or any noise before the first '{'
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise
        data = json.loads(raw_text[start:end+1])

    if isinstance(data, (list, tuple)):
        # common pattern: [true, {...graph...}] or (true, {...})
        for el in data:
            if isinstance(el, dict) and "nodes" in el:
                return el
        # fallback: second element
        if len(data) >= 2 and isinstance(data[1], dict):
            return data[1]
        raise ValueError("Could not locate graph dict inside list/tuple JSON.")

    if isinstance(data, dict) and "nodes" in data:
        return data

    raise ValueError("Unrecognized EnvironmentGraph format.")

# ---------- Converters ----------

def vh_to_grid_sg(vh_graph: dict):
    """
    Returns:
        sg: dict (GRID-style scene graph)
        vh2sg: mapping {vh_id -> sg_id}
    """
    vh_nodes = {n["id"]: n for n in vh_graph.get("nodes", [])}
    vh_edges = vh_graph.get("edges", [])

    # Build category buckets
    rooms_vh = []
    objects_vh = []
    persons_vh = []
    for n in vh_graph.get("nodes", []):
        cat = norm_category(n.get("category", ""))
        if cat == "room":
            rooms_vh.append(n["id"])
        elif cat == "person":
            persons_vh.append(n["id"])
        elif cat in {"floor", "wall", "ceiling"}:
            # structural geometry: ignore as manipulable objects
            continue
        else:
            objects_vh.append(n["id"])

    sg_nodes = []
    vh2sg = {}
    next_id = 0

    # Synthetic single floor root
    floor_id = next_id
    sg_nodes.append({
        "id": next_id,
        "type": "floor",
        "attributes": {"label": "floor", "color": ""},
        "operation": []
    })
    next_id += 1

    # Rooms
    for rid in rooms_vh:
        rnode = vh_nodes[rid]
        label = _norm(rnode.get("class_name", "room"))
        vh2sg[rid] = next_id
        sg_nodes.append({
            "id": next_id,
            "type": "room",
            "attributes": {"label": label, "color": ""},
            "operation": []
        })
        next_id += 1

    # Objects
    for oid in objects_vh:
        onode = vh_nodes[oid]
        label = _norm(onode.get("class_name", "object"))
        props_uc = {p.upper() for p in onode.get("properties", [])}

        node_type = "small_object" if "GRABBABLE" in props_uc else "large_object"

        ops = []
        ops += infer_pick_ops(props_uc)
        ops += infer_surface_ops(label, props_uc)
        ops += infer_mechanism_ops(label)
        # dedupe ops preserving order
        seen = set()
        ops = [x for x in ops if not (x in seen or seen.add(x))]

        # map OPEN/CLOSED state if present
        states_uc = {_norm(s).upper() for s in onode.get("states", [])}
        state_attr = ""
        if "OPEN" in states_uc or "CLOSED" in states_uc:
            mech_ops = infer_mechanism_ops(label)
            if mech_ops:
                if "OPEN" in states_uc and any(op.endswith("_open") for op in mech_ops):
                    state_attr = [op for op in mech_ops if op.endswith("_open")][0]
                elif "CLOSED" in states_uc and any(op.endswith("_close") for op in mech_ops):
                    state_attr = [op for op in mech_ops if op.endswith("_close")][0]
            else:
                state_attr = "open" if "OPEN" in states_uc else "closed"

        vh2sg[oid] = next_id
        attr = {"label": label, "color": ""}
        if state_attr:
            attr["state"] = state_attr
        sg_nodes.append({
            "id": next_id,
            "type": node_type,
            "attributes": attr,
            "operation": ops
        })
        next_id += 1

    # Edges: floor -> rooms
    sg_edges = [{"from_id": floor_id, "to_id": vh2sg[rid], "relation_type": "in"} for rid in rooms_vh if rid in vh2sg]

    # INSIDE relationships (room/container → object)
    for e in vh_edges or []:
        rel = _norm(e.get("relation_type", ""))  # can be "INSIDE"
        a, b = e.get("from_id"), e.get("to_id")
        if rel != "inside":
            continue
        if a not in vh2sg or b not in vh2sg:
            # allow room target not mapped yet
            if a in vh2sg and b in rooms_vh:
                # ensure room mapped
                pass
            else:
                continue
        # don’t create room-in-room
        if a in rooms_vh:
            continue
        # add container/room contains object
        sg_edges.append({"from_id": vh2sg.get(b, vh2sg.get(b, b)), "to_id": vh2sg[a], "relation_type": "in"})

    # Fallback: if there is exactly one room and no INSIDE edges place all objects under it
    if rooms_vh and not any(_norm(e.get("relation_type","")) == "inside" for e in vh_edges or []):
        if len(rooms_vh) == 1:
            only_room = rooms_vh[0]
            for oid in objects_vh:
                sg_edges.append({"from_id": vh2sg[only_room], "to_id": vh2sg[oid], "relation_type": "in"})

    sg = {"version": "1.1", "nodes": sg_nodes, "edges": sg_edges}
    return sg, vh2sg, persons_vh

def auto_robot_id(vh_graph: dict, persons_vh_list):
    # prefer explicit Characters/person nodes; fallback: any node with class_name "character"
    if persons_vh_list:
        return persons_vh_list[0]
    for n in vh_graph.get("nodes", []):
        if _norm(n.get("class_name")) == "character":
            return n["id"]
    return None

def vh_to_grid_rg(vh_graph: dict, vh2sg: dict, robot_vh_id: int):
    vh_nodes = {n["id"]: n for n in vh_graph.get("nodes", [])}
    vh_edges = vh_graph.get("edges", []) or []

    if robot_vh_id not in vh_nodes:
        raise ValueError("Robot id not found in VH nodes.")

    include_ids = set()
    edges_rg_tmp = []

    def add_edge(a, b, rel):
        edges_rg_tmp.append((a, b, rel))
        include_ids.add(a); include_ids.add(b)

    # near / grasp / in
    for e in vh_edges:
        rel = _norm(e.get("relation_type", ""))
        a, b = e.get("from_id"), e.get("to_id")
        if rel == "close":
            if a == robot_vh_id:
                add_edge(a, b, "near")
            elif b == robot_vh_id:
                add_edge(b, a, "near")
        elif rel in ("holds_lh", "holds_rh"):
            if a == robot_vh_id:
                add_edge(a, b, "grasp")
        elif rel == "inside":
            if a == robot_vh_id:
                add_edge(a, b, "in")

    # If we saw any nodes already, also record their room containment
    known = set(include_ids)
    for e in vh_edges:
        if _norm(e.get("relation_type","")) != "inside":
            continue
        a, b = e["from_id"], e["to_id"]
        if a in known or b in known:
            add_edge(a, b, "in")

    # Build RG ids: 0 -> robot
    vh2rg = {robot_vh_id: 0}
    next_id = 1
    for vid in sorted(include_ids):
        if vid == robot_vh_id:
            continue
        vh2rg[vid] = next_id
        next_id += 1

    # Nodes
    rg_nodes = [{"id": 0, "type": "robot"}]
    for vid, rid in vh2rg.items():
        if vid == robot_vh_id:
            continue
        cat = norm_category(vh_nodes[vid].get("category"))
        if cat == "room":
            ntype = "room"
        else:
            ntype = "object"
        rg_nodes.append({"id": rid, "type": ntype})

    # Edges
    rg_edges = []
    for a, b, rel in edges_rg_tmp:
        if a not in vh2rg or (b not in vh2rg and b != robot_vh_id):
            continue
        rg_edges.append({"from_id": vh2rg[a], "to_id": vh2rg[b], "relation_type": rel})

    return {"version": "1.1", "nodes": rg_nodes, "edges": rg_edges}

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to VH EnvironmentGraph JSON")
    ap.add_argument("--out-sg", required=True, help="Output path for GRID scene graph JSON")
    ap.add_argument("--out-rg", required=True, help="Output path for GRID relation graph JSON")
    ap.add_argument("--robot-id", type=int, default=None, help="VH node id for the robot (auto-detect if omitted)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON outputs")
    args = ap.parse_args()

    vh_graph = load_vh_env_graph(args.input)
    sg, vh2sg, persons_vh = vh_to_grid_sg(vh_graph)
    robot_id = args.robot_id or auto_robot_id(vh_graph, persons_vh)
    if robot_id is None:
        raise SystemExit("Could not auto-detect robot/person node. Provide --robot-id explicitly.")

    rg = vh_to_grid_rg(vh_graph, vh2sg, robot_id)

    dump_kwargs = {"indent": 2, "ensure_ascii": False} if args.pretty else {}
    with open(args.out_sg, "w") as f:
        json.dump(sg, f, **dump_kwargs)
    with open(args.out_rg, "w") as f:
        json.dump(rg, f, **dump_kwargs)

    print(f"[OK] Wrote SG -> {args.out_sg}")
    print(f"[OK] Wrote RG -> {args.out_rg}")

if __name__ == "__main__":
    main()