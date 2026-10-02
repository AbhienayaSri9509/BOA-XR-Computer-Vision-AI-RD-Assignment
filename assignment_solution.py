"""
BOA XR - Computer Vision / AI R&D Internship Assignment
Part 1: Oriented Bounding Box (OBB) measurement
Part 2: Gravity-aware 3D packing ("3D Tetris")

Install:
    pip install trimesh numpy matplotlib

Run:
    python assignment_solution.py

Part 1:
    python assignment_solution.py --part 1

Part 2:
    python assignment_solution.py --part 2

The script writes generated images to ./outputs/.
"""

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)
MASTER = np.array([100.0, 100.0, 100.0])


# ----------------------------- PART 1 ---------------------------------

def load_mesh(path: Path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def calculate_obb(path: Path):
    """Calculate Trimesh's oriented bounding box and return dimensions/volume."""
    mesh = load_mesh(path)
    obb = mesh.bounding_box_oriented
    dims = np.asarray(obb.primitive.extents, dtype=float)
    # Sort only for presentation as L >= W >= H; raw OBB axes are preserved
    display_dims = np.sort(dims)[::-1]
    return mesh, obb, display_dims, float(obb.volume)


def draw_obb(mesh, obb, title, save_path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Sample mesh for faster display on very dense models.
    if len(mesh.vertices) > 12000:
        idx = np.linspace(0, len(mesh.vertices) - 1, 12000).astype(int)
        verts = mesh.vertices[idx]
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=1, alpha=0.35)
    else:
        ax.plot_trisurf(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
            triangles=mesh.faces, alpha=0.25, linewidth=0
        )

    # OBB corners -> 12 edges.
    corners = np.asarray(obb.vertices)
    edges = [
        (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
        (3,7),(4,5),(4,6),(5,7),(6,7)
    ]
    for a, b in edges:
        ax.plot(
            [corners[a,0], corners[b,0]],
            [corners[a,1], corners[b,1]],
            [corners[a,2], corners[b,2]],
            linewidth=2.2
        )

    dims = np.sort(obb.primitive.extents)[::-1]
    ax.set_title(
        f"{title}\nOBB: {dims[0]:.3f} × {dims[1]:.3f} × {dims[2]:.3f} | "
        f"Volume: {obb.volume:.3f}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def run_part1(model_dir: Path):
    files = ["CUBE.obj", "CYLINDER.obj", "TEAPOT.obj"]
    print("\nPART 1 - ORIENTED BOUNDING BOX\n")
    results = []

    for name in files:
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing model: {path}")

        mesh, obb, dims, volume = calculate_obb(path)
        results.append((name, dims, volume))
        print(f"{name:12s}  L×W×H = {dims[0]:.6f} × {dims[1]:.6f} × {dims[2]:.6f}"
              f"   Volume = {volume:.6f}")

        draw_obb(
            mesh, obb, name,
            OUTPUT / f"{path.stem.lower()}_obb.png"
        )

    with open(OUTPUT / "obb_results.txt", "w", encoding="utf-8") as f:
        for name, dims, volume in results:
            f.write(
                f"{name}: L={dims[0]:.6f}, W={dims[1]:.6f}, H={dims[2]:.6f}, "
                f"Volume={volume:.6f}\n"
            )


# ----------------------------- PART 2 ---------------------------------

@dataclass
class Placement:
    item_id: int
    item_type: str
    pos: np.ndarray       # x, y, z = minimum corner
    dims: np.ndarray      # dx, dy, dz


def rotations(dims):
    """All unique axis-aligned rotations."""
    return sorted(set(tuple(p) for p in itertools.permutations(map(float, dims))))


def overlaps_xy(a, b):
    """Positive-area footprint intersection in XY."""
    return (
        min(a.pos[0] + a.dims[0], b.pos[0] + b.dims[0])
        > max(a.pos[0], b.pos[0]) + 1e-8
        and
        min(a.pos[1] + a.dims[1], b.pos[1] + b.dims[1])
        > max(a.pos[1], b.pos[1]) + 1e-8
    )


def overlap_volume(a, b):
    dx = min(a.pos[0]+a.dims[0], b.pos[0]+b.dims[0]) - max(a.pos[0], b.pos[0])
    dy = min(a.pos[1]+a.dims[1], b.pos[1]+b.dims[1]) - max(a.pos[1], b.pos[1])
    dz = min(a.pos[2]+a.dims[2], b.pos[2]+b.dims[2]) - max(a.pos[2], b.pos[2])
    return max(0, dx) * max(0, dy) * max(0, dz)


def collides(pos, dims, placed):
    candidate = Placement(-1, "", np.asarray(pos, float), np.asarray(dims, float))
    return any(overlap_volume(candidate, p) > 1e-8 for p in placed)


def in_master_box(pos, dims):
    return (
        np.all(pos >= -1e-8)
        and np.all(pos + dims <= MASTER + 1e-8)
    )


def supported_z(x, y, dims, placed):
    """
    Lowest legal Z at (x,y).
    A box can rest on the floor, or on one/more boxes whose top face
    intersects its footprint with positive area.
    """
    base = 0.0
    candidates = [0.0]
    footprint = Placement(-1, "", np.array([x, y, 0.0]), np.asarray(dims))

    for p in placed:
        if overlaps_xy(footprint, p):
            candidates.append(float(p.pos[2] + p.dims[2]))

    # Try each support level, choosing the lowest level with no 3D collision.
    for z in sorted(set(candidates)):
        if in_master_box(np.array([x, y, z]), dims) and not collides(
            np.array([x, y, z]), dims, placed
        ):
            return z
    return None


def candidate_xy_positions(dims, placed):
    """
    Candidate x/y coordinates come from container boundaries and the
    sides of already placed objects. This is a standard extreme-point
    packing heuristic and avoids an exhaustive 100^3 search.
    """
    xs = {0.0, float(MASTER[0] - dims[0])}
    ys = {0.0, float(MASTER[1] - dims[1])}

    for p in placed:
        xs.update([float(p.pos[0]), float(p.pos[0] + p.dims[0])])
        ys.update([float(p.pos[1]), float(p.pos[1] + p.dims[1])])

    xs = sorted(x for x in xs if -1e-8 <= x <= MASTER[0] - dims[0] + 1e-8)
    ys = sorted(y for y in ys if -1e-8 <= y <= MASTER[1] - dims[1] + 1e-8)
    return xs, ys


def score_candidate(pos, dims, placed):
    """
    Lower is better:
      1) lower height,
      2) lower layer/footprint coordinates,
      3) lower exposed height.
    """
    x, y, z = pos
    # Prefer low, then back-left positions. A small "gap" term favors
    # compact placement without sacrificing the gravity constraint.
    gap = 0.0
    for p in placed:
        if overlaps_xy(Placement(-1,"",np.array([x,y,z]),np.asarray(dims)), p):
            gap += abs((p.pos[2] + p.dims[2]) - z)
    return (round(z, 6), round(x + y, 6), round(gap, 6), round(x, 6), round(y, 6))


def place_item(item, placed):
    best = None

    # Larger dimensions first in the caller; here try every allowed orientation.
    for rot in rotations(item["dims"]):
        dims = np.array(rot, dtype=float)
        xs, ys = candidate_xy_positions(dims, placed)

        # Include coordinates immediately next to existing objects in both axes.
        for x in xs:
            for y in ys:
                z = supported_z(x, y, dims, placed)
                if z is None:
                    continue

                pos = np.array([x, y, z], dtype=float)
                if not in_master_box(pos, dims) or collides(pos, dims, placed):
                    continue

                score = score_candidate(pos, dims, placed)
                if best is None or score < best[0]:
                    best = (score, Placement(item["id"], item["type"], pos, dims))

    return best[1] if best else None


def validate_placements(placements):
    assert len(placements) == 20, f"Expected 20 placements, got {len(placements)}"

    for p in placements:
        assert in_master_box(p.pos, p.dims), f"Item {p.item_id} outside master box"

    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            assert overlap_volume(placements[i], placements[j]) <= 1e-8, (
                f"Items {placements[i].item_id} and {placements[j].item_id} overlap"
            )

    # Gravity/support validation: floor OR positive XY support from another item.
    for p in placements:
        if abs(p.pos[2]) < 1e-8:
            continue
        supported = False
        for q in placements:
            if p.item_id == q.item_id:
                continue
            top = q.pos[2] + q.dims[2]
            if abs(top - p.pos[2]) < 1e-8 and overlaps_xy(p, q):
                supported = True
                break
        assert supported, f"Item {p.item_id} is floating"


def pack_items(items):
    # Volume descending tends to reduce fragmentation.
    ordered = sorted(
        items,
        key=lambda i: np.prod(i["dims"]),
        reverse=True
    )

    placed = []
    for item in ordered:
        p = place_item(item, placed)
        if p is None:
            raise RuntimeError(f"Could not place item {item['id']} ({item['type']})")
        placed.append(p)
        print(
            f"Placed {p.item_id:2d} {p.item_type:14s} "
            f"at ({p.pos[0]:5.1f}, {p.pos[1]:5.1f}, {p.pos[2]:5.1f}) "
            f"dims={tuple(int(x) if x.is_integer() else round(x,2) for x in p.dims)}"
        )

    validate_placements(placed)
    return placed


def cuboid_faces(pos, dims):
    x, y, z = pos
    dx, dy, dz = dims
    v = np.array([
        [x,y,z], [x+dx,y,z], [x+dx,y+dy,z], [x,y+dy,z],
        [x,y,z+dz], [x+dx,y,z+dz], [x+dx,y+dy,z+dz], [x,y+dy,z+dz]
    ])
    return [
        [v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],
        [v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]],
        [v[1],v[2],v[6],v[5]], [v[4],v[7],v[3],v[0]]
    ]


def draw_box(ax, pos, dims, label):
    faces = cuboid_faces(pos, dims)
    poly = Poly3DCollection(faces, alpha=0.55, linewidths=0.5)
    ax.add_collection3d(poly)

    center = pos + dims / 2
    ax.text(center[0], center[1], center[2], str(label), fontsize=7)


def set_box_axes(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def save_packing_snapshots(placements):
    # Final overview.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for p in placements:
        draw_box(ax, p.pos, p.dims, p.item_id)
    set_box_axes(ax)
    ax.set_title("BOA XR - Final 3D Packing (20 Items)")
    plt.tight_layout()
    plt.savefig(OUTPUT / "packing_final.png", dpi=160)
    plt.close(fig)

    # Step-by-step frames, suitable for screen recording or video creation.
    frames = OUTPUT / "packing_frames"
    frames.mkdir(exist_ok=True)

    for i in range(1, len(placements) + 1):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for p in placements[:i]:
            draw_box(ax, p.pos, p.dims, p.item_id)
        set_box_axes(ax)
        ax.set_title(f"3D Tetris Packing — Step {i}/20")
        plt.tight_layout()
        plt.savefig(frames / f"step_{i:02d}.png", dpi=140)
        plt.close(fig)


def run_part2(json_path: Path):
    items = json.loads(json_path.read_text(encoding="utf-8"))
    placements = pack_items(items)

    output = []
    for p in placements:
        output.append({
            "id": p.item_id,
            "type": p.item_type,
            "position_xyz": [round(float(v), 3) for v in p.pos],
            "placed_dimensions": [round(float(v), 3) for v in p.dims]
        })

    (OUTPUT / "packing_coordinates.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8"
    )
    save_packing_snapshots(placements)

    total_volume = sum(float(np.prod(p.dims)) for p in placements)
    master_volume = float(np.prod(MASTER))
    print("\nPART 2 VALIDATION")
    print(f"Items packed: {len(placements)}/20")
    print(f"Total item volume: {total_volume:.1f}")
    print(f"Master box volume: {master_volume:.1f}")
    print(f"Volume utilization: {100*total_volume/master_volume:.2f}%")
    print("No overlaps: PASS")
    print("Inside 100×100×100 master box: PASS")
    print("Gravity/support constraint: PASS")
    print(f"Outputs: {OUTPUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["1", "2", "both"], default="both")
    parser.add_argument(
        "--model-dir",
        default=str(ROOT),
        help="Directory containing CUBE.obj, CYLINDER.obj, TEAPOT.obj"
    )
    parser.add_argument(
        "--json",
        default=str(ROOT / "Item List.json"),
        help="Path to Item List.json"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if args.part in ("1", "both"):
        run_part1(model_dir)
    if args.part in ("2", "both"):
        run_part2(Path(args.json))


if __name__ == "__main__":
    main()
