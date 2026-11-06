"""Rivers and open water, taken from the save's land-use grid and bent into the schematic.

The save stores land use as a square byte grid (``city.lu``); value 7 marks water. Water is
split into two kinds that are drawn differently:

* open water (sea, estuary, lakes): the part of the mask that survives a morphological
  opening. Its outlines are traced on the map and drawn as filled polygons.
* rivers: the thin remainder. Each river becomes one centre line (the longest path along
  its skeleton), mapped into layout space and drawn as a thick stroke.

The map -> layout mapping is piecewise affine over a Delaunay triangulation of the
stations. Outside the stations' convex hull every point uses the affine map fitted to the
nearest stations, which is only an approximation, so far-away shapes are simplified hard.
"""
import base64
import heapq
import math

import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, distance_transform_edt, label
from scipy.spatial import Delaunay, cKDTree
from skimage.measure import find_contours
from skimage.morphology import disk, skeletonize

WATER_CODE = 7
OPEN_RADIUS = 4          # grid cells; water narrower than ~2*radius counts as river
MIN_RIVER_CELLS = 25
MIN_OPEN_CELLS = 120


def water_mask(save):
    """Boolean water grid (rows = y) or None when the save has no land-use layer."""
    lu = (save.get('city') or {}).get('lu')
    if not isinstance(lu, dict) or 'b' not in lu:
        return None
    raw = np.frombuffer(base64.b64decode(lu['b']), np.uint8)
    n = int(round(math.sqrt(raw.size)))
    if n * n != raw.size:
        return None
    return raw.reshape(n, n) == WATER_CODE


class Warp:
    """Piecewise-affine map from map (grid) coordinates to layout coordinates."""

    def __init__(self, geo, lay, k=10, k_far=30):
        self.geo, self.lay = np.asarray(geo, float), np.asarray(lay, float)
        self.k, self.k_far = min(k, len(geo)), min(k_far, len(geo))
        self.tri, self.tree = Delaunay(self.geo), cKDTree(self.geo)

    def forward(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        out = np.empty_like(pts)
        simp = self.tri.find_simplex(pts)
        inside = simp >= 0
        if inside.any():
            T = self.tri.transform[simp[inside]]
            b = np.einsum('nij,nj->ni', T[:, :2], pts[inside] - T[:, 2])
            bary = np.c_[b, 1 - b.sum(1)]
            out[inside] = np.einsum('ni,nij->nj', bary, self.lay[self.tri.simplices[simp[inside]]])
        # outside the hull: affine map fitted to the nearest stations, weighted by distance.
        # Far from the network more stations are used so the extrapolation stays smooth.
        hull_d, _ = self.tree.query(pts)
        for i in np.nonzero(~inside)[0]:
            dd, nn = self.tree.query(pts[i], k=self.k if hull_d[i] < 8 else self.k_far)
            A = np.c_[self.geo[nn], np.ones(len(nn))]; w = 1 / (dd + 1)
            coef, *_ = np.linalg.lstsq(A * w[:, None], self.lay[nn] * w[:, None], rcond=None)
            out[i] = np.append(pts[i], 1) @ coef
        return out


def split(mask):
    """(open water, river cells) masks."""
    open_ = binary_opening(mask, structure=disk(OPEN_RADIUS))
    lab, n = label(open_, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1); sizes[0] = 0
    open_ = np.isin(lab, np.where(sizes >= MIN_OPEN_CELLS)[0])
    thin = mask & ~binary_dilation(open_, iterations=OPEN_RADIUS)   # drop shore fringes
    return open_, thin


def douglas_idx(pts, tol):
    """Indices kept by Douglas-Peucker simplification of an open polyline."""
    def rec(i, j):
        if j <= i + 1: return [i, j]
        a, b = pts[i], pts[j]; ab = b - a; L = np.linalg.norm(ab) + 1e-9; w = pts[i + 1:j] - a
        d = np.abs(ab[0] * w[:, 1] - ab[1] * w[:, 0]) / L; k = int(np.argmax(d)) + i + 1
        return rec(i, k)[:-1] + rec(k, j) if d[k - i - 1] > tol else [i, j]
    return rec(0, len(pts) - 1)


def simplify_ring(pts, tol):
    """Douglas-Peucker for a closed ring (first point == last point). A ring has no chord
    to measure against, so its two halves, split at the vertex farthest from the start,
    are simplified separately."""
    k = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    return np.vstack([pts[douglas_idx(pts[:k + 1], tol)[:-1]], pts[k:][douglas_idx(pts[k:], tol)]])


def open_water_polygons(open_, warp, push=4000.0, tol=1.5):
    """Outlines of open water in layout space.

    Outlines are simplified on the map (tolerance in grid cells) before warping, so only a
    few vertices go through the extrapolated part of the warp. Where water touches the map
    border the outline follows the border; after warping, those points are pushed far out,
    straight across the border they lie on (the schematic keeps the map's orientation), so
    the water runs off the canvas instead of ending in a visible straight edge.
    """
    if not open_.any():
        return []
    n = open_.shape[0]
    grid = np.pad(open_.astype(float), 1)
    polys = []
    for c in find_contours(grid, 0.5):
        if len(c) < 8:
            continue
        g = np.c_[c[:, 1] - 0.5, c[:, 0] - 0.5]
        g = np.clip(g, 0, n)
        g = simplify_ring(g, tol)
        out_dir = np.c_[(g[:, 0] >= n - 0.01).astype(float) - (g[:, 0] <= 0.01),
                        (g[:, 1] >= n - 0.01).astype(float) - (g[:, 1] <= 0.01)]
        pts = warp.forward(g)
        L = np.linalg.norm(out_dir, axis=1, keepdims=True)
        pts += np.divide(out_dir, L, out=np.zeros_like(out_dir), where=L > 0) * push
        polys.append(pts)
    return polys


def river_lines(thin, open_, mask):
    """Centre lines (map coordinates, cell centres) of rivers, each with its median width."""
    dl = distance_transform_edt(mask)
    lab, n = label(thin, structure=np.ones((3, 3)))
    out = []
    dist_open = distance_transform_edt(~open_) if open_.any() else None
    for c in range(1, n + 1):
        comp = lab == c
        if comp.sum() < MIN_RIVER_CELLS:
            continue
        sk = skeletonize(comp)
        cells = list(zip(*np.nonzero(sk)))
        if len(cells) < 4:
            continue
        cell_set = set(cells)

        def farthest(src):
            best = {src: 0.0}; prev = {src: None}; pq = [(0.0, src)]
            while pq:
                d, (y, x) = heapq.heappop(pq)
                if d > best[(y, x)]:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nb = (y + dy, x + dx)
                        if (dy or dx) and nb in cell_set:
                            nd = d + math.hypot(dy, dx)
                            if nd < best.get(nb, 1e18):
                                best[nb] = nd; prev[nb] = (y, x); heapq.heappush(pq, (nd, nb))
            end = max(best, key=best.get)
            return end, prev
        a, _ = farthest(cells[0])
        b, prev = farthest(a)
        path = []; p = b
        while p is not None:
            path.append(p); p = prev[p]
        # extend the river into open water where it flows into it
        if dist_open is not None:
            ys, xs = np.nonzero(open_)
            for at_end in (False, True):
                cell = path[-1] if at_end else path[0]
                if dist_open[cell] < OPEN_RADIUS + 3:
                    k = int(np.argmin((ys - cell[0]) ** 2 + (xs - cell[1]) ** 2))
                    mouth = (int(ys[k]), int(xs[k]))
                    path = path + [mouth] if at_end else [mouth] + path
        width = float(np.median([2 * dl[y, x] for y, x in path]))
        out.append((np.array([(x + 0.5, y + 0.5) for y, x in path]), width))
    return out


def clip_polygon(pts, box):
    """Sutherland-Hodgman clip of a polygon to box = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    out = [np.asarray(p, float) for p in pts]
    for axis, lim, keep_le in ((0, x0, False), (0, x1, True), (1, y0, False), (1, y1, True)):
        if not out:
            break
        src, out = out, []
        inside = (lambda p: p[axis] <= lim) if keep_le else (lambda p: p[axis] >= lim)
        for i, p in enumerate(src):
            q = src[i - 1]
            if inside(p) != inside(q):
                t = (lim - q[axis]) / (p[axis] - q[axis])
                out.append(q + t * (p - q))
            if inside(p):
                out.append(p)
    return np.array(out)


def _dist_to_rings(p, rings):
    best = (np.inf, None)
    for r in rings:
        a, b = r[:-1], r[1:]; ab = b - a
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / (np.einsum('ij,ij->i', ab, ab) + 1e-9), 0, 1)
        q = a + t[:, None] * ab; d = np.linalg.norm(q - p, axis=1); i = int(np.argmin(d))
        if d[i] < best[0]: best = (d[i], q[i])
    return best


def join_to_shore(river, rings, reach):
    """End a river (layout space) where it first comes within ``reach`` of open water,
    counted from its inland end.

    The warp can squeeze a river against the shore it runs beside on the map; following
    the bank further would draw two parallel strips, so the river is cut where it gets
    close and continued straight into the water."""
    if not rings or len(river) < 2:
        return river
    rings = [np.vstack([r, r[:1]]) for r in rings]
    d = [_dist_to_rings(p, rings) for p in river]
    near = [x[0] < reach for x in d]
    if all(near):
        return river
    river = list(river)
    def cut(i):
        q = d[i][1]; v = q - river[i]; v = v / (np.linalg.norm(v) + 1e-9)
        return q + v * reach
    if near[-1] and not near[0]:          # flows out at the end: cut at the first approach
        i = near.index(True)
        river = river[:i + 1] + [cut(i)]
    elif near[0] and not near[-1]:        # flows out at the start
        j = len(near) - 1 - near[::-1].index(True)
        river = [cut(j)] + river[j:]
    return np.array(river)
