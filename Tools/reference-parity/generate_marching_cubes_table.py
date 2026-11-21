#!/usr/bin/env python3
"""Record the triangulation table of `NFKHi3DGenMarchingCubes`.

Hi3DGen extracts its mesh with scikit-image's `measure.marching_cubes` (Lewiner et al. 2003). The port
decides each cube's topology itself: the corner sign pattern, one decision a face whose diagonal corners
share a sign (the product of the inside pair against the product of the outside pair), and one interior
decision for the two-isolated-corner cases and the two-ambiguous-face cases (the face decision on the
slice where it peaks). The triangles that fill a decided cube are a choice of the reference's table that
no rule derives, so this script records them by running the reference on single cubes as a black box.

The cube's corner `c` sits at (c & 1, (c >> 1) & 1, (c >> 2) & 1); its twelve edges are the corner
pairs differing in one bit, in ascending order; edge 12 names the vertex the reference adds inside a
cube. A face lists its corners in cycle order, and a cube's decisions are read in face order.

The script samples every sign pattern with random magnitudes, checks that each decided cube always
yields the same triangles, checks the whole extraction on random grids against the reference in order,
and prints the Swift table: one `mask:` token, then one token a face-decision combination (bit k set
when the k-th ambiguous face joins its inside corners), `.` where no sample reached it, and
`without|with` where the interior decision splits it.

Run in the Hi3DGen oracle environment (scikit-image 0.26):
    ~/.inferkit-validation/hi3dgenvenv/bin/python3 -I Tools/reference-parity/generate_marching_cubes_table.py
"""
import numpy as np
from skimage import measure

CORNERS = [(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)]
EDGES = [(a, b) for a in range(8) for b in range(8) if a < b and bin(a ^ b).count("1") == 1]
FACES = [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]]
EPSILON = float(np.finfo(np.float64).eps)
TWO_INSIDE_ISOLATED = {24: (3, 4), 36: (2, 5), 66: (1, 6), 129: (0, 7)}
TWO_OUTSIDE_ISOLATED = {255 - mask: pair for mask, pair in TWO_INSIDE_ISOLATED.items()}
OPPOSITE_AMBIGUOUS_FACES = {60, 90, 102, 153, 165, 195}


def ambiguous_faces(mask):
    found = []
    for index, face in enumerate(FACES):
        signs = [mask >> c & 1 for c in face]
        if signs[0] == signs[2] and signs[1] == signs[3] and signs[0] != signs[1]:
            found.append(index)
    return found


def projected(axis, corner):
    return corner & ~(1 << axis)


def peak_slice(values, axis, a, b, c, d):
    """The four values on the slice across `axis` where a·c − b·d is stationary, or None outside the cube."""
    def ends(corner):
        return values[corner], values[corner | (1 << axis)]
    (a0, a1), (b0, b1), (c0, c1), (d0, d1) = ends(a), ends(b), ends(c), ends(d)
    quadratic = (a1 - a0) * (c1 - c0) - (b1 - b0) * (d1 - d0)
    linear = c0 * (a1 - a0) + a0 * (c1 - c0) - d0 * (b1 - b0) - b0 * (d1 - d0)
    if quadratic == 0:
        return None
    t = -linear / (2 * quadratic)
    if not 0 <= t <= 1:
        return None
    return [x0 + (x1 - x0) * t for x0, x1 in ((a0, a1), (b0, b1), (c0, c1), (d0, d1))]


def tunnel(values, mask, joins):
    if mask in TWO_INSIDE_ISOLATED or mask in TWO_OUTSIDE_ISOLATED:
        p, q = TWO_INSIDE_ISOLATED.get(mask) or TWO_OUTSIDE_ISOLATED[mask]
        a, c = projected(0, p), projected(0, q)
        b, d = [k for k in (0, 2, 4, 6) if k not in (a, c)]
        s = peak_slice(values, 0, a, b, c, d)
        if mask in TWO_INSIDE_ISOLATED:
            return s is None or (s[0] < 0 and s[2] < 0)
        return s is not None and s[0] >= 0 and s[2] >= 0 and s[1] < 0 and s[3] < 0 and s[0] * s[2] > s[1] * s[3]
    if mask in OPPOSITE_AMBIGUOUS_FACES and all(joins):
        face = FACES[ambiguous_faces(mask)[0]]
        axis = [i for i in range(3) if all(CORNERS[k][i] == CORNERS[face[0]][i] for k in face)][0]
        a, c = 0, 7 ^ (1 << axis)
        b, d = [k for k in range(8) if not k >> axis & 1 and k not in (a, c)]
        s = peak_slice(values, axis, a, b, c, d)
        if s is None:
            return False
        if s[0] >= 0 and s[2] >= 0 and s[1] < 0 and s[3] < 0:
            return s[0] * s[2] > s[1] * s[3]
        if s[1] >= 0 and s[3] >= 0 and s[0] < 0 and s[2] < 0:
            return s[1] * s[3] > s[0] * s[2]
        return False
    return None


def decide(corner_values):
    values = [float(x) for x in corner_values]
    values = [-EPSILON if x == 0 else x for x in values]
    mask = sum(1 << c for c in range(8) if values[c] < 0)
    joins = []
    for index in ambiguous_faces(mask):
        w = [values[k] for k in FACES[index]]
        inside, outside = (w[0] * w[2], w[1] * w[3]) if w[0] < 0 else (w[1] * w[3], w[0] * w[2])
        joins.append(inside > outside)
    return (mask, tuple(joins), tunnel(values, mask, joins)), values


def edge_label(point):
    inner = [i for i in range(3) if 0 < point[i] < 1]
    if len(inner) == 3:
        return 12
    if len(inner) != 1:
        raise ValueError(point)
    axis = inner[0]
    base = [int(round(point[i])) if i != axis else 0 for i in range(3)]
    corner = base[0] + 2 * base[1] + 4 * base[2]
    return EDGES.index((corner, corner | (1 << axis)))


def reference_cube(corner_values):
    grid = np.zeros((2, 2, 2), np.float32)
    for c in range(8):
        grid[CORNERS[c]] = corner_values[c]
    vertices, faces, _, _ = measure.marching_cubes(grid, level=0.0, gradient_direction="ascent")
    labels = [edge_label(p) for p in vertices]
    return tuple(tuple(labels[i] for i in face) for face in faces)


def record(seed=0):
    rng = np.random.default_rng(seed)
    table = {}
    for mask in range(1, 255):
        draws = 400 if not ambiguous_faces(mask) else 8000
        if mask in (105, 150):
            draws = 120000
        for draw in range(draws):
            magnitude = 10 ** rng.uniform(-3, 0, 8) if draw % 2 else rng.uniform(0.001, 1, 8)
            corner_values = np.array([-magnitude[c] if mask >> c & 1 else magnitude[c] for c in range(8)], np.float32)
            key, _ = decide(corner_values)
            triangles = reference_cube(corner_values)
            if table.setdefault(key, triangles) != triangles:
                raise SystemExit(f"cube {corner_values.tolist()} decided {key} gave two triangulations")
    return table


def extract(grid, table):
    """The whole extraction as the port runs it: cubes x-major, vertices in order of first use."""
    side = grid.shape[0] - 1
    vertices, faces, by_edge = [], [], {}
    for x in range(side):
        for y in range(side):
            for z in range(side):
                corner_values = [grid[x + o[0], y + o[1], z + o[2]] for o in CORNERS]
                key, values = decide(corner_values)
                if key[0] in (0, 255):
                    continue
                center = None
                for triangle in table[key]:
                    face = []
                    for e in triangle:
                        if e == 12:
                            if center is None:
                                weights = [1 / abs(q) for q in values]
                                total = sum(weights)
                                local = [sum(weights[c] * CORNERS[c][i] for c in range(8)) / total for i in range(3)]
                                vertices.append([np.float32(x + local[0]), np.float32(y + local[1]), np.float32(z + local[2])])
                                center = len(vertices) - 1
                            face.append(center)
                            continue
                        a, b = EDGES[e]
                        axis = (a ^ b).bit_length() - 1
                        lower = (x + CORNERS[a][0], y + CORNERS[a][1], z + CORNERS[a][2])
                        if (lower, axis) not in by_edge:
                            point = [float(q) for q in lower]
                            point[axis] += values[a] / (values[a] - values[b])
                            vertices.append([np.float32(q) for q in point])
                            by_edge[(lower, axis)] = len(vertices) - 1
                        face.append(by_edge[(lower, axis)])
                    faces.append(face)
    return np.array(vertices, np.float32).reshape(-1, 3), np.array(faces, np.int64).reshape(-1, 3)


def check_grids(table, count=300, seed=99):
    rng = np.random.default_rng(seed)
    checked = 0
    for trial in range(count):
        n = int(rng.integers(3, 14))
        if trial % 2:
            grid = rng.uniform(-1, 1, (n, n, n))
        else:
            grid = 10 ** rng.uniform(-3, 0, (n, n, n)) * rng.choice([-1, 1], (n, n, n))
        grid = grid.astype(np.float32)
        if (grid < 0).all() or (grid > 0).all():
            continue
        vertices, faces, _, _ = measure.marching_cubes(grid, level=0.0, gradient_direction="ascent")
        ours, our_faces = extract(grid, table)
        if not (np.array_equal(ours, vertices) and np.array_equal(our_faces, faces)):
            raise SystemExit(f"grid {trial} ({n}³) differs from the reference")
        checked += 1
    return checked


def swift_literal(table):
    symbols = "0123456789abc"
    tokens = []
    for mask in range(1, 255):
        tokens.append(f"{mask:02x}:")
        for combination in range(1 << len(ambiguous_faces(mask))):
            joins = tuple(bool(combination >> k & 1) for k in range(len(ambiguous_faces(mask))))
            def text(split):
                triangles = table.get((mask, joins, split))
                return "." if triangles is None else "".join(symbols[e] for t in triangles for e in t)
            if (mask, joins, True) in table or (mask, joins, False) in table:
                tokens.append(text(False) + "|" + text(True))
            else:
                tokens.append(text(None))
    lines, line = [], ""
    for token in tokens:
        if line and (token.endswith(":") or len(line) + 1 + len(token) > 112):
            lines.append(line)
            line = token
        else:
            line = f"{line} {token}" if line else token
    lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    recorded = record()
    print(f"// {len(recorded)} decided cubes; {check_grids(recorded)} random grids identical to the reference in order")
    print(swift_literal(recorded))
