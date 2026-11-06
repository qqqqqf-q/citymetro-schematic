"""Turn a near-octilinear layout into a strictly octilinear one with few bends.

Every line is cut into chains between key stations (interchanges, termini, branch
points). A chain is drawn either straight or with exactly one bend; the directions are
chosen from the input layout. Coordinates are then solved as an LP: edges are exactly
horizontal / vertical / 45deg, every edge is at least one unit long, total length is
minimised. Edges whose direction cannot be satisfied (inconsistent cycles) become
'free' edges drawn with one bend. Overlaps are repaired by lazily adding linear
ordering constraints taken from the input layout.
"""
import math, collections
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def run(lay, STRAIGHT_TOL=12.0, DP_TOL=0.8, log=print):
    """Strictly octilinear layout with few bends, from a pre-layout."""
    ids = [s['id'] for s in lay['stations']]; idx = {k: i for i, k in enumerate(ids)}; N = len(ids)
    U = lay['unit']
    P0 = np.array([[s['x'] / U, s['y'] / U] for s in lay['stations']])
    E = [(idx[e['a']], idx[e['b']]) for e in lay['edges']]; M = len(E)
    elines = [e['lines'] for e in lay['edges']]
    eid = {}
    for e, (a, b) in enumerate(E): eid[(a, b)] = e; eid[(b, a)] = e
    inc = collections.defaultdict(list)
    for e, (a, b) in enumerate(E): inc[a].append(e); inc[b].append(e)
    nlines = collections.Counter()
    st_lines = collections.defaultdict(set)
    for e, (a, b) in enumerate(E):
        for l in elines[e]: st_lines[a].add(l); st_lines[b].add(l)

    DV = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)], float)
    def angk(v): return math.atan2(v[1], v[0]) / (math.pi / 4)

    # ---- chains per line
    la = collections.defaultdict(lambda: collections.defaultdict(list))
    for e, (a, b) in enumerate(E):
        for l in elines[e]: la[l][a].append(b); la[l][b].append(a)
    def douglas(pts_idx, tol):
        """Douglas-Peucker on station indices; returns kept indices (positions in list)."""
        pts = P0[pts_idx]
        def rec(i, j):
            if j <= i + 1: return [i, j]
            a, b = pts[i], pts[j]; ab = b - a; L = np.linalg.norm(ab) + 1e-12
            w_ = pts[i + 1:j] - a; d = np.abs(ab[0] * w_[:, 1] - ab[1] * w_[:, 0]) / L
            k = int(np.argmax(d)) + i + 1
            if d[k - i - 1] > tol:
                left = rec(i, k); right = rec(k, j); return left[:-1] + right
            return [i, j]
        return rec(0, len(pts) - 1)
    chains = []
    done = set()
    for l, adj in la.items():
        lkey = {n for n, nb in adj.items() if len(nb) != 2}
        for s in (lkey or [next(iter(adj))]):
            for n in adj[s]:
                if (l, s, n) in done: continue
                path = [s, n]
                while path[-1] not in lkey and len(adj[path[-1]]) == 2:
                    nxt = [m for m in adj[path[-1]] if m != path[-2]][0]
                    path.append(nxt)
                    if nxt == s: break
                done.add((l, path[0], path[1])); done.add((l, path[-1], path[-2]))
                keep = douglas(path, DP_TOL)
                for i, j in zip(keep, keep[1:]):
                    chains.append((l, path[i:j + 1]))
    # direction (as integer k, oriented a->b of the stored edge) for each edge
    want = {}
    for l, ch in chains:
        v = P0[ch[-1]] - P0[ch[0]]
        t = angk(v); k = int(round(t)); dev = abs(t - k) * 45
        steps = list(zip(ch, ch[1:]))
        if dev <= STRAIGHT_TOL or len(steps) == 1 and dev <= 22.5 and False:
            seq = [k] * len(steps)
        else:
            k1, k2 = math.floor(t), math.ceil(t)
            best = None
            for first, second in ((k1, k2), (k2, k1)):
                for j in range(len(steps) + 1):
                    seq_ = [first] * j + [second] * (len(steps) - j)
                    if len(steps) > 1 and j in (0, len(steps)) : pass
                    cost = 0
                    for (a, b), kk in zip(steps, seq_):
                        tt = angk(P0[b] - P0[a]); cost += min(abs(tt - kk), abs(tt - kk + 8), abs(tt - kk - 8)) ** 2
                    # the chain as a whole must land on the right end point: both directions used
                    if len(steps) > 1 and (j == 0 or j == len(steps)): cost += 3
                    if best is None or cost < best[0]: best = (cost, seq_)
            seq = best[1]
            if len(steps) == 1:
                seq = [k]                         # the LP decides whether it can stay straight
        for (a, b), kk in zip(steps, seq):
            e = eid[(a, b)]
            if e in want: continue
            if kk is None: want[e] = None
            else: want[e] = kk % 8 if E[e][0] == a else (kk + 4) % 8

    # ---- port conflicts: two edges leaving a node in the same direction
    def port(e, n):
        k = want[e]
        if k is None: return None
        return k if E[e][0] == n else (k + 4) % 8
    for rnd in range(5):
        changed = 0
        for n in range(N):
            byp = collections.defaultdict(list)
            for e in inc[n]:
                p = port(e, n)
                if p is not None: byp[p].append(e)
            for p, es in byp.items():
                if len(es) < 2: continue
                # keep the edge closest to its direction, rotate the others to free neighbours
                def dev_of(e):
                    o = E[e][1] if E[e][0] == n else E[e][0]
                    t = angk(P0[o] - P0[n]); return min(abs(t - p), abs(t - p + 8), abs(t - p - 8)), t
                es.sort(key=lambda e: dev_of(e)[0])
                used = {port(e2, n) for e2 in inc[n]}
                for e in es[1:]:
                    _, t = dev_of(e)
                    side = 1 if ((t - p + 4) % 8 - 4) > 0 else -1
                    for q in ((p + side) % 8, (p - side) % 8):
                        if q not in used:
                            want[e] = q if E[e][0] == n else (q + 4) % 8; used.add(q); changed += 1; break
                    else:
                        want[e] = None; changed += 1
        if not changed: break

    # ---- LP
    def solve(free, seps):
        rows, cols, vals, rhs = [], [], [], []
        r = [0]
        def le(coefs, b):            # sum coefs <= b
            for c, v in coefs: rows.append(r[0]); cols.append(c); vals.append(v)
            rhs.append(b); r[0] += 1
        X = lambda i: i; Y = lambda i: N + i
        nS = M; S_ = lambda e: 2 * N + e          # slack on perpendicular equality
        L_ = lambda e: 2 * N + nS + e             # length proxy
        NV = 2 * N + nS + M
        for e, (a, b) in enumerate(E):
            k = want[e]
            if k is None or e in free:
                v = P0[b] - P0[a]
                for ax, comp in ((X, v[0]), (Y, v[1])):
                    if abs(comp) > 0.25 * np.abs(v).max():
                        sg = np.sign(comp); le([(ax(b), -sg), (ax(a), sg)], -0.5)
                le([(L_(e), -1), (X(b), 1), (X(a), -1)], 0); le([(L_(e), -1), (X(b), -1), (X(a), 1)], 0)
                le([(L_(e), -1), (Y(b), 1), (Y(a), -1)], 0); le([(L_(e), -1), (Y(b), -1), (Y(a), 1)], 0)
                continue
            dx, dy = DV[k]
            if dy == 0: perp = [(Y(b), 1), (Y(a), -1)]
            elif dx == 0: perp = [(X(b), 1), (X(a), -1)]
            elif dx == dy: perp = [(X(b), 1), (X(a), -1), (Y(b), -1), (Y(a), 1)]
            else: perp = [(X(b), 1), (X(a), -1), (Y(b), 1), (Y(a), -1)]
            le(perp + [(S_(e), -1)], 0); le([(c, -v) for c, v in perp] + [(S_(e), -1)], 0)
            along = [(X(b), -dx), (X(a), dx)] if dx else [(Y(b), -dy), (Y(a), dy)]
            le(along, -1.0)
            le([(L_(e), -1)] + [(c, -v) for c, v in along], 0)
        for coefs, b in seps: le(coefs, b)
        A = coo_matrix((vals, (rows, cols)), shape=(r[0], NV)).tocsr()
        c = np.zeros(NV)
        for e in range(M):
            c[S_(e)] = 60.0; c[L_(e)] = 1.0
        bounds = [(None, None)] * (2 * N) + [(0, None)] * (nS + M)
        bounds[0] = (0, 0); bounds[N] = (0, 0)
        res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method='highs')
        if res.status != 0:
            raise RuntimeError('LP failed: %s' % res.message)
        z = res.x
        return np.stack([z[:N], z[N:2 * N]], 1), z[2 * N:2 * N + M]

    AX = [np.array(v) / np.linalg.norm(v) for v in ((1, 0), (0, 1), (1, 1), (1, -1))]
    def sep_constraint_nodes(i, j, D=0.8):
        v = P0[j] - P0[i]
        best = max(range(4), key=lambda a: abs(v @ AX[a]) / (1 if a < 2 else 1.0))
        ax = AX[best] * (1 if best < 2 else math.sqrt(2))   # unnormalised (1,1) for diagonals
        s = np.sign(v @ ax) or 1
        # s*ax.(p_j - p_i) >= D
        co = []
        for comp, base in ((0, 0), (1, N)):
            if ax[comp]: co += [(base + j, -s * ax[comp]), (base + i, s * ax[comp])]
        return (co, -D)
    def sep_constraint_node_edge(i, e, D=0.6):
        a, b = E[e]; k = want[e]
        dx, dy = DV[k]; n_ = np.array([-dy, dx])           # perpendicular (unnormalised)
        s = np.sign((P0[i] - P0[a]) @ n_) or 1
        co = [(a, s * n_[0]), (N + a, s * n_[1]), (i, -s * n_[0]), (N + i, -s * n_[1])]
        return (co, -D * np.linalg.norm(n_))

    def violations(P, free):
        bad = []
        d = np.abs(P[:, None, :] - P[None, :, :]).max(2) + np.eye(N) * 99
        adjset = {(a, b) for a, b in E} | {(b, a) for a, b in E}
        for i, j in zip(*np.where(np.triu(d < 0.79, 1))):
            bad.append(('nn', i, j))
        for e, (a, b) in enumerate(E):
            A_, B_ = P[a], P[b]; AB = B_ - A_; L2 = AB @ AB + 1e-12
            for i in range(N):
                if i in (a, b): continue
                t = (P[i] - A_) @ AB / L2
                if -0.05 < t < 1.05 and np.linalg.norm(P[i] - (A_ + np.clip(t, 0, 1) * AB)) < 0.45:
                    bad.append(('ne', i, e))
        def cross(p1, p2, p3, p4):
            def o(a, b, c): return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            return o(p1, p2, p3) * o(p1, p2, p4) < 0 and o(p3, p4, p1) * o(p3, p4, p2) < 0
        for e1 in range(M):
            for e2 in range(e1 + 1, M):
                a, b = E[e1]; c_, d_ = E[e2]
                if {a, b} & {c_, d_}: continue
                if cross(P[a], P[b], P[c_], P[d_]) and not cross(P0[a], P0[b], P0[c_], P0[d_]):
                    bad.append(('ee', e1, e2))
        return bad

    free = set(); seps = []; seen = set()
    for it in range(40):
        P, slack = solve(free, seps)
        newfree = {e for e in range(M) if slack[e] > 1e-6 and want[e] is not None and e not in free}
        if newfree:
            free |= newfree
            log(f'iter {it}: {len(newfree)} edges cannot be straight -> free'); continue
        bad = violations(P, free)
        log(f'iter {it}: violations {len(bad)} {dict(collections.Counter(k for k, _, _ in bad))}')
        if not bad: break
        for kind, x, y in bad:
            if (kind, x, y) in seen: continue
            seen.add((kind, x, y))
            if kind == 'nn': seps.append(sep_constraint_nodes(x, y))
            elif kind == 'ne':
                if want[y] is not None and y not in free: seps.append(sep_constraint_node_edge(x, y))
                else:
                    a, b = E[y]; seps.append(sep_constraint_nodes(x, a, 0.9)); seps.append(sep_constraint_nodes(x, b, 0.9))
            else:
                for e_line, e_other in ((x, y), (y, x)):
                    if want[e_line] is None or e_line in free: continue
                    for i in E[e_other]: seps.append(sep_constraint_node_edge(i, e_line, 0.3))

    nfree = sum(1 for e in range(M) if want[e] is None or e in free)
    log(f'free (bent) edges {nfree} of {M}')
    out = dict(lay)
    UNIT = 40.0
    out['stations'] = [dict(s, x=float(P[i, 0] * UNIT), y=float(P[i, 1] * UNIT)) for i, s in enumerate(lay['stations'])]
    out['edges'] = []
    for e, ed in enumerate(lay['edges']):
        k = want[e]
        f = k is None or e in free
        if f:
            v = P[E[e][1]] - P[E[e][0]]; k = int(round(angk(v))) % 8
        out['edges'].append(dict(ed, pa=k, pb=(k + 4) % 8, free=f))
    return out
