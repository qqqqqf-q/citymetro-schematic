"""Octilinear schematic layout for a city-metro save.

Least-squares iteration: every edge is pulled toward an octilinear port direction
assigned per station (distinct ports via assignment), same-line neighbours are
pulled straight, geography is a weak anchor, and close station pairs / stations
sitting on foreign edges are pushed apart.
"""
import json, sys, math, collections
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

save_path, out_path = sys.argv[1], sys.argv[2]
d = json.loads(json.load(open(save_path))['body'])
net = d['net']; S = net['stations']; G = net['segments']; L = net['lines']

# ---- graph -------------------------------------------------------------
used = set()
for g in G.values(): used.add(g['a']); used.add(g['b'])
ids = [k for k in S if k in used]
idx = {k: i for i, k in enumerate(ids)}
N = len(ids)
geo = np.array([[S[k]['x'], S[k]['y']] for k in ids], float)

edge_lines = collections.OrderedDict()
for g in G.values():
    a, b = idx[g['a']], idx[g['b']]
    key = (min(a, b), max(a, b))
    edge_lines.setdefault(key, [])
    if g['line'] not in edge_lines[key]: edge_lines[key].append(g['line'])
E = list(edge_lines)
M = len(E)
inc = collections.defaultdict(list)          # node -> [(edge index, other node)]
for e, (a, b) in enumerate(E):
    inc[a].append((e, b)); inc[b].append((e, a))

# same-line consecutive triples (a, i, b) for straightness
line_adj = collections.defaultdict(lambda: collections.defaultdict(set))
for g in G.values():
    a, b = idx[g['a']], idx[g['b']]
    line_adj[g['line']][a].add(b); line_adj[g['line']][b].add(a)
triples = []
for lid, adj in line_adj.items():
    for i, nb in adj.items():
        nb = sorted(nb)
        if len(nb) == 2: triples.append((nb[0], i, nb[1]))

# ---- geography preprocessing: compress outskirts ----------------------
center = geo[[idx[k] for k in ids if S[k]['name'] in ('大世界',)]][0]
rel = geo - center
r = np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9
R0 = 18.0
r2 = R0 * (r / R0) ** float(sys.argv[3] if len(sys.argv) > 3 else 0.55)   # power fisheye: expands the dense core, compresses outskirts
geo2 = rel / r * r2
elen = np.array([np.linalg.norm(geo2[a] - geo2[b]) for a, b in E])
UNIT = 40.0
geo2 *= UNIT / np.median(elen)
elen = np.array([np.linalg.norm(geo2[a] - geo2[b]) for a, b in E])
target_len = np.clip(elen, UNIT * 0.85, UNIT * 2.2)
# short edges get a uniform length; long ones keep some of their length
target_len = UNIT * 0.9 + 0.35 * np.maximum(target_len - UNIT * 0.9, 0)

DIRS = np.array([[math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)] for k in range(8)])

def assign_ports(P):
    """For each node, map incident edge -> port (0..7), distinct where possible."""
    port = {}
    for i, lst in inc.items():
        vecs = np.array([P[o] - P[i] for _, o in lst])
        ang = np.arctan2(vecs[:, 1], vecs[:, 0])
        cap = 1 if len(lst) <= 8 else 2
        cols = [k for k in range(8) for _ in range(cap)]
        cost = np.zeros((len(lst), len(cols)))
        for c, k in enumerate(cols):
            da = np.abs((ang - k * math.pi / 4 + math.pi) % (2 * math.pi) - math.pi)
            cost[:, c] = da ** 2
        rr, cc = linear_sum_assignment(cost)
        for r_, c_ in zip(rr, cc):
            port[(lst[r_][0], i)] = cols[c_]
    return port

def solve(P, w_oct, w_str, w_geo, w_rep, port, extra):
    rows = []      # (coeff dict, rhs_x, rhs_y, weight)
    for e, (a, b) in enumerate(E):
        la = target_len[e]
        da = DIRS[port[(e, a)]]; db = DIRS[port[(e, b)]]
        # p_b - p_a ≈ la*da  and  p_a - p_b ≈ la*db
        tgt = (la * da - la * db) / 2
        rows.append(({b: 1, a: -1}, tgt[0], tgt[1], w_oct))
    for a, i, b in triples:
        rows.append(({i: 2, a: -1, b: -1}, 0, 0, w_str))
    for i in range(N):
        rows.append(({i: 1}, geo2[i, 0], geo2[i, 1], w_geo))
    for (i, j, vx, vy) in extra:
        rows.append(({i: 1, j: -1}, vx, vy, w_rep))
    A = lil_matrix((len(rows), N)); bx = np.zeros(len(rows)); by = np.zeros(len(rows))
    for r_, (cf, x, y, w) in enumerate(rows):
        for k, v in cf.items(): A[r_, k] = v * w
        bx[r_] = x * w; by[r_] = y * w
    A = A.tocsr()
    nx = lsqr(A, bx, x0=P[:, 0], atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    ny = lsqr(A, by, x0=P[:, 1], atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    return np.stack([nx, ny], 1)

def seg_point_dist(p, a, b):
    ab = b - a; t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    q = a + t * ab
    return np.linalg.norm(p - q), q, t

def repulsions(P, dmin):
    extra = []
    for i in range(N):
        for j in range(i + 1, N):
            v = P[i] - P[j]; dd = np.linalg.norm(v)
            if dd < dmin:
                u = v / (dd + 1e-9) if dd > 1e-6 else np.array([1.0, 0])
                extra.append((i, j, *(u * dmin)))
    # station too close to a foreign edge: push station away from the edge's nearest point
    for e, (a, b) in enumerate(E):
        for i in range(N):
            if i == a or i == b: continue
            dist, q, t = seg_point_dist(P[i], P[a], P[b])
            if dist < dmin * 0.6 and 0.05 < t < 0.95:
                n = P[i] - q; n = n / (np.linalg.norm(n) + 1e-9) if np.linalg.norm(n) > 1e-6 else np.array([0, 1.0])
                near = a if t < 0.5 else b
                extra.append((i, near, *(P[near] - P[near] + (P[i] - P[near]) + n * (dmin * 0.6 - dist))))
    return extra

P = geo2.copy()
port = assign_ports(P)
schedule = [(0.3, 0.6, 0.3), (0.6, 0.6, 0.2), (1.0, 0.7, 0.12), (2.0, 0.8, 0.08), (4.0, 0.8, 0.05), (8.0, 0.8, 0.03), (16.0, 0.6, 0.02), (40.0, 0.4, 0.01)]
for w_oct, w_str, w_geo in schedule:
    for it in range(12):
        extra = repulsions(P, UNIT * 0.8)
        P = solve(P, w_oct, w_str, w_geo, 2.5 * max(1, w_oct / 4), port, extra)
        port = assign_ports(P)


# ---- phase 2: hard octilinear solve -------------------------------------
BENDPEN = {4: 0.0, 3: 0.12, 5: 0.12, 2: 0.7, 6: 0.7, 1: 4.0, 7: 4.0}

def angdev(v, k):
    a = math.atan2(v[1], v[0])
    return abs((a - k * math.pi / 4 + math.pi) % (2 * math.pi) - math.pi)

def assign_ports2(P, prev):
    port = {}
    for i, lst in inc.items():
        if len(lst) == 2:
            (e1, o1), (e2, o2) = lst
            v1, v2 = P[o1] - P[i], P[o2] - P[i]
            best = None
            for k1 in range(8):
                for k2 in range(8):
                    if k1 == k2: continue
                    c = angdev(v1, k1) ** 2 + angdev(v2, k2) ** 2 + BENDPEN[(k1 - k2) % 8]
                    if prev is not None:
                        c -= 0.02 * ((prev.get((e1, i)) == k1) + (prev.get((e2, i)) == k2))
                    if best is None or c < best[0]: best = (c, k1, k2)
            port[(e1, i)] = best[1]; port[(e2, i)] = best[2]
        else:
            vecs = [P[o] - P[i] for _, o in lst]
            cap = 1 if len(lst) <= 8 else 2
            cols = [k for k in range(8) for _ in range(cap)]
            cost = np.array([[angdev(v, k) ** 2 for k in cols] for v in vecs])
            rr, cc = linear_sum_assignment(cost)
            for r_, c_ in zip(rr, cc): port[(lst[r_][0], i)] = cols[c_]
    # an edge whose two ends nearly agree should agree exactly
    for e, (a, b) in enumerate(E):
        pa, pb = port[(e, a)], port[(e, b)]
        if pa != (pb + 4) % 8:
            v = P[b] - P[a]
            k = int(round(math.atan2(v[1], v[0]) / (math.pi / 4))) % 8
            free_a = all(port[(e2, a)] != k for e2, _ in inc[a] if e2 != e)
            free_b = all(port[(e2, b)] != (k + 4) % 8 for e2, _ in inc[b] if e2 != e)
            if angdev(v, k) < math.radians(20) and free_a and free_b:
                port[(e, a)] = k; port[(e, b)] = (k + 4) % 8
    return port

def repulsions_vec(P, dmin):
    extra = []
    diff = P[:, None, :] - P[None, :, :]
    dist = np.linalg.norm(diff, axis=2) + np.eye(N) * 1e9
    ii, jj = np.where(np.triu(dist < dmin, 1))
    for i, j in zip(ii, jj):
        v = diff[i, j]; dd = dist[i, j]
        u = v / dd if dd > 1e-6 else np.array([1.0, 0])
        extra.append((i, j, *(u * dmin)))
    A = P[[a for a, b in E]]; B = P[[b for a, b in E]]
    AB = B - A; L2 = (AB ** 2).sum(1) + 1e-9
    for i in range(N):
        t = np.clip(((P[i] - A) * AB).sum(1) / L2, 0, 1)
        Q = A + t[:, None] * AB
        dd = np.linalg.norm(P[i] - Q, axis=1)
        for e in np.where((dd < dmin * 0.55) & (t > 0.02) & (t < 0.98))[0]:
            a, b = E[e]
            if i in (a, b): continue
            n = P[i] - Q[e]; nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-6 else np.array([-AB[e, 1], AB[e, 0]]) / math.sqrt(L2[e])
            near = a if t[e] < 0.5 else b
            extra.append((i, near, *((P[i] - P[near]) + n * (dmin * 0.55 - dd[e]))))
    return extra

def solve_hard(P, port, extra, w_geo, w_c=60.0, w_rep=None):
    w_rep = w_rep or w_c * 0.3
    rows, cols, vals, rhs = [], [], [], []
    r = [0]
    def add(coefs, b, w):
        for c, v in coefs: rows.append(r[0]); cols.append(c); vals.append(v * w)
        rhs.append(b * w); r[0] += 1
    X = lambda i: i; Y = lambda i: N + i
    for e, (a, b) in enumerate(E):
        pa, pb = port[(e, a)], port[(e, b)]
        v = P[b] - P[a]
        if pa == (pb + 4) % 8:
            dx, dy = DIRS[pa]
            add([(X(b), dy), (X(a), -dy), (Y(b), -dx), (Y(a), dx)], 0.0, w_c)
            cur = v @ DIRS[pa]
            ell = float(np.clip(cur, UNIT * 0.95, UNIT * 2.4))
            add([(X(b), dx), (X(a), -dx), (Y(b), dy), (Y(a), -dy)], ell, w_c * 0.6 if cur < UNIT * 0.9 else 2.0)
        else:
            # bent edge: keep both legs reasonably long
            da, db = DIRS[pa], DIRS[pb]
            M2 = np.array([[da[0], -db[0]], [da[1], -db[1]]])
            try:
                s_, t_ = np.linalg.solve(M2, v)
            except np.linalg.LinAlgError:
                s_, t_ = 1, 1
            ls = max(s_, UNIT * 0.5); lt = max(t_, UNIT * 0.5)
            tgt = ls * da - lt * db
            add([(X(b), 1), (X(a), -1)], tgt[0], 0.6)
            add([(Y(b), 1), (Y(a), -1)], tgt[1], 0.6)
    for i in range(N):
        add([(X(i), 1)], geo2[i, 0], w_geo); add([(Y(i), 1)], geo2[i, 1], w_geo)
    for (i, j, vx, vy) in extra:
        add([(X(i), 1), (X(j), -1)], vx, w_rep); add([(Y(i), 1), (Y(j), -1)], vy, w_rep)
    from scipy.sparse import csr_matrix
    A = csr_matrix((vals, (rows, cols)), shape=(r[0], 2 * N))
    z = lsqr(A, np.array(rhs), x0=np.concatenate([P[:, 0], P[:, 1]]), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    return np.stack([z[:N], z[N:]], 1)

port2 = None
for it in range(60):
    port2 = assign_ports2(P, port2)
    extra = repulsions_vec(P, UNIT * float(sys.argv[4] if len(sys.argv) > 4 else 0.95))
    P = solve_hard(P, port2, extra, w_geo=0.01 if it < 40 else 0.002, w_c=60.0 if it < 40 else 400.0)
port = port2
nbend = sum(1 for e, (a, b) in enumerate(E) if port[(e, a)] != (port[(e, b)] + 4) % 8)
print('bent edges', nbend, 'of', M)

# report octilinearity
devs = []
for e, (a, b) in enumerate(E):
    v = P[b] - P[a]; ang = math.degrees(math.atan2(v[1], v[0]))
    devs.append(abs((ang + 22.5) % 45 - 22.5))
devs = np.array(devs)
print('edge angle deviation: median %.2f p90 %.2f max %.2f' % (np.median(devs), np.percentile(devs, 90), devs.max()))
mind = min(np.linalg.norm(P[i] - P[j]) for i in range(N) for j in range(i + 1, N))
print('min station distance %.1f (unit %.0f)' % (mind, UNIT))

out = {
    'unit': UNIT,
    'stations': [{'id': k, 'name': S[k]['name'], 'x': float(P[i, 0]), 'y': float(P[i, 1]),
                  'lines': sorted({l for e, _ in inc[i] for l in edge_lines[E[e]]}, key=lambda l: L[l]['num'])}
                 for k, i in idx.items()],
    'edges': [{'a': ids[a], 'b': ids[b], 'lines': edge_lines[(a, b)],
               'pa': port[(e, a)], 'pb': port[(e, b)]} for e, (a, b) in enumerate(E)],
    'geo': {k: [float(geo[i,0]), float(geo[i,1])] for k, i in idx.items()},
    'lines': {k: {'num': v['num'], 'name': v['name'], 'color': v['color']} for k, v in L.items()},
    'meta': {'name': d['meta']['name'], 'date': '%d年%d月' % (d['clock']['y'], d['clock']['m'])},
}
json.dump(out, open(out_path, 'w'), ensure_ascii=False, indent=1)
