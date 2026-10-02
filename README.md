# BOA XR — Computer Vision / AI R&D Internship Assignment

## Overview
This project implements both parts of the assignment:

1. **3D OBB measurement:** loads CUBE.obj, CYLINDER.obj and TEAPOT.obj, computes an oriented bounding box, and reports dimensions and volume.
2. **3D Tetris packing:** loads `Item List.json`, places all 20 items inside a 100×100×100 master box using axis-aligned rotations, collision checks, extreme-point candidate positions, and gravity/support constraints.

## Requirements
Python 3.10+ recommended.

```bash
pip install trimesh numpy matplotlib
```

## Run

Run everything:

```bash
python assignment_solution.py
```

Run only Part 1:

```bash
python assignment_solution.py --part 1
```

Run only Part 2:

```bash
python assignment_solution.py --part 2
```

## Part 1 output
Generated under `outputs/`:
- `cube_obb.png`
- `cylinder_obb.png`
- `teapot_obb.png`
- `obb_results.txt`

The OBB is computed with Trimesh's oriented bounding-box implementation rather than a screen-aligned AABB.

## Part 2 output
Generated under `outputs/`:
- `packing_final.png`
- `packing_coordinates.json`
- `packing_frames/step_01.png` ... `step_20.png`

The placement heuristic:
- sorts larger-volume objects first;
- tests all unique axis-aligned rotations;
- generates candidate x/y coordinates from the master-box boundaries and placed-object boundaries;
- drops each candidate to the lowest valid z-level;
- rejects collisions and out-of-box placements;
- validates that every non-floor item has positive-area footprint support from an item directly below.

## Screen recording
For the requested submission video:
1. Open the project in VS Code/PyCharm.
2. Run `python assignment_solution.py --part 1`.
3. Show the terminal results and open the three OBB images.
4. Run `python assignment_solution.py --part 2`.
5. Show the placement coordinates and validation output.
6. Open `outputs/packing_frames` and step through the 20 frames to demonstrate one-by-one stacking.
7. Upload the recording to Google Drive and set **Anyone with the link → Viewer**.


