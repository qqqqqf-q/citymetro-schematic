"""Render a strictly octilinear layout in the style of the Shanghai official network map."""
import json, sys, math, base64, collections, re, heapq
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.ndimage import distance_transform_edt
from pypinyin import lazy_pinyin

lay_path, save_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
lay = json.load(open(lay_path))
save = json.loads(json.load(open(save_path))['body'])

SCALE = 2.0                       # layout unit 40 -> 80 px between adjacent stations
LW = 7.0                          # line width
PAR = LW + 2.0                    # spacing of parallel lines on shared track
FONT = '"PingFang SC", "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif'
FONT_EN = 'Arial, Helvetica, sans-serif'
FS, FS_EN = 16.0, 8.0
INK = '#1a1a1a'
WATER = '#bfe2f5'

st = {s['id']: dict(s) for s in lay['stations']}
for s in st.values(): s['x'] *= SCALE; s['y'] *= SCALE
LINES = lay['lines']; edges = lay['edges']
def P(sid): return np.array([st[sid]['x'], st[sid]['y']])

# ---------------------------------------------------------------- colours
def hex2rgb(h): h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
def rgb2lab(c):
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M_ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    x, y, z = M_ @ c / np.array([0.95047, 1.0, 1.08883])
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.array([116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))])
def hsl(h, s, l):
    import colorsys
    return '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in colorsys.hls_to_rgb(h, l, s))
by_num = sorted(LINES, key=lambda k: LINES[k]['num'])
# lines 1-18 keep the game's (Shanghai) colours; later lines reuse them in the save,
# which makes e.g. line 1 and 19 indistinguishable -> pick distinct ones
taken, recolored, seen = [], {}, {}
for k in by_num:
    c = LINES[k]['color'].lower()
    if c in seen: recolored[k] = None
    else: seen[c] = k; taken.append(rgb2lab(hex2rgb(c)))
pool = [hsl(h / 72, s, l) for h in range(72) for s in (0.42, 0.55, 0.65) for l in (0.34, 0.42, 0.5)]
pool_lab = [rgb2lab(hex2rgb(c)) for c in pool]
for k in by_num:
    if k in recolored:
        j = int(np.argmax([min(np.linalg.norm(p - t) for t in taken) for p in pool_lab]))
        LINES[k]['color'] = pool[j]; taken.append(pool_lab[j]); recolored[k] = pool[j]

# ---------------------------------------------------------------- English names
SPECIAL = {
    '一大会址': 'Site of the 1st CPC National Congress', '外滩': 'The Bund', '豫园': 'Yuyuan Garden',
    '大世界': 'Great World', '静安寺': "Jing'an Temple", '世博会博物馆': 'World Expo Museum',
    '世博大道': 'Expo Ave.', '中华艺术宫': 'China Art Museum', '东方体育中心': 'Oriental Sports Center',
    '上海': 'Shanghai Railway Station', '上海南': 'Shanghai South Railway Station',
    '浦东机场': 'Pudong International Airport', '虹桥火车站': 'Hongqiao Railway Station',
    '上海科技馆': 'Shanghai Science & Technology Museum', '上海儿童医学中心': "Shanghai Children's Medical Center",
    '上海游泳馆': 'Shanghai Swimming Center', '上海动物园': 'Shanghai Zoo', '十六铺': 'Shiliupu',
}
SUFFIX = [('1号航站楼', 'Airport Terminal 1'), ('2号航站楼', 'Airport Terminal 2'), ('火车站', 'Railway Station'),
          ('工业园区', 'Industrial Park'), ('体育中心', 'Sports Center'), ('足球场', 'Football Stadium'),
          ('博物馆', 'Museum'), ('公园', 'Park'), ('花园', 'Garden'), ('大学', 'University'), ('大道', 'Ave.'),
          ('机场', 'Airport'), ('新镇', 'New Town'), ('1站', 'Station 1'), ('路', 'Rd.'), ('镇', 'Town')]
DIRN = {'东': 'E', '西': 'W', '南': 'S', '北': 'N'}
def py(s):
    out = ''
    for i, syl in enumerate(lazy_pinyin(s)):
        if i and syl[:1] in 'aoe': out += "'"
        out += syl
    return out[:1].upper() + out[1:]
def en_part(n):
    if n in SPECIAL: return SPECIAL[n]
    d = ''
    if len(n) >= 3 and n[-1] in DIRN and n[:-1] not in ('杨浦',):
        base = n[:-1]
        if base in SPECIAL or any(base.endswith(z) for z, _ in SUFFIX) or len(base) >= 2:
            d = ' (%s)' % DIRN[n[-1]]; n = base
    if n in SPECIAL: return SPECIAL[n] + d
    for z, e in SUFFIX:
        if n.endswith(z) and len(n) > len(z):
            head = n[:-len(z)]
            head_en = py(head) if head.startswith('上海') else (SPECIAL.get(head) or py(head))
            return f'{head_en} {e}{d}'
    return py(n) + d
def en_name(n): return ' · '.join(en_part(p) for p in n.split('·'))

# ---------------------------------------------------------------- per-line topology
line_adj = collections.defaultdict(lambda: collections.defaultdict(list))
edge_of = {}
for e in edges:
    for l in e['lines']:
        line_adj[l][e['a']].append(e['b']); line_adj[l][e['b']].append(e['a'])
    edge_of[(e['a'], e['b'])] = e; edge_of[(e['b'], e['a'])] = e

def chains(adj):
    seen_ = set(); out = []
    for s in [n for n, nb in adj.items() if len(nb) != 2]:
        for n in adj[s]:
            if (s, n) in seen_: continue
            ch = [s, n]; seen_.add((s, n)); seen_.add((n, s))
            while len(adj[ch[-1]]) == 2:
                nxt = [m for m in adj[ch[-1]] if m != ch[-2]][0]
                if (ch[-1], nxt) in seen_: break
                seen_.add((ch[-1], nxt)); seen_.add((nxt, ch[-1])); ch.append(nxt)
            out.append(ch)
    return out

DIRS = np.array([[math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)] for k in range(8)])
def two_leg(p, q, prefer_first=None):
    v = q - p; t = math.atan2(v[1], v[0]) / (math.pi / 4)
    if abs(t - round(t)) * 45 < 0.5: return [p, q]
    k1, k2 = math.floor(t), math.ceil(t); best = None
    for f, g in ((k1, k2), (k2, k1)):
        d1, d2 = DIRS[f % 8], DIRS[g % 8]
        s_, t_ = np.linalg.solve(np.array([[d1[0], d2[0]], [d1[1], d2[1]]]), v)
        score = min(s_, t_) + (1e3 if prefer_first is not None and f % 8 == prefer_first else 0)
        if best is None or score > best[0]: best = (score, p + s_ * d1)
    return [p, best[1], q]

# parallel offsets for track shared by several lines
def offset_of(e, l):
    return np.zeros(2)
    if len(e['lines']) < 2: return np.zeros(2)
    a, b = P(e['a']), P(e['b']); v = (b - a) / np.linalg.norm(b - a); n = np.array([-v[1], v[0]])
    order = sorted(e['lines'], key=lambda x: LINES[x]['num'])
    i = order.index(l)
    return n * (i - (len(order) - 1) / 2) * PAR

# directions already taken at each station by straight edges; bent edges must leave/arrive
# through a free direction, otherwise their first leg would run on top of another line
def dir_idx(v): return int(round(math.atan2(v[1], v[0]) / (math.pi / 4))) % 8
used_dir = collections.defaultdict(set)
def is_straight(e):
    v = P(e['b']) - P(e['a']); t = math.atan2(v[1], v[0]) / (math.pi / 4)
    return abs(t - round(t)) * 45 < 0.5
for e in edges:
    if e.get('via'):
        vs = [np.array(v) * SCALE for v in e['via']]
        used_dir[e['a']].add(dir_idx(vs[0] - P(e['a']))); used_dir[e['b']].add(dir_idx(vs[-1] - P(e['b'])))
for e in edges:
    if is_straight(e) and not e.get('via'):
        used_dir[e['a']].add(dir_idx(P(e['b']) - P(e['a']))); used_dir[e['b']].add(dir_idx(P(e['a']) - P(e['b'])))
bent_route = {}
for e in edges:
    if is_straight(e) or e.get('via'): continue
    p, q = P(e['a']), P(e['b']); v = q - p; t = math.atan2(v[1], v[0]) / (math.pi / 4)
    k1, k2 = math.floor(t), math.ceil(t); best = None
    for f, g in ((k1, k2), (k2, k1)):
        d1, d2 = DIRS[f % 8], DIRS[g % 8]
        s_, t_ = np.linalg.solve(np.array([[d1[0], d2[0]], [d1[1], d2[1]]]), v)
        clash = (f % 8 in used_dir[e['a']]) + ((g + 4) % 8 in used_dir[e['b']])
        score = -100 * clash + min(s_, t_)
        if best is None or score > best[0]: best = (score, f % 8, (g + 4) % 8, p + s_ * d1)
    used_dir[e['a']].add(best[1]); used_dir[e['b']].add(best[2])
    bent_route[(e['a'], e['b'])] = best[3]
VIA = {}
for e in edges:
    if e.get('via'):
        pts_ = [np.array(v) * SCALE for v in e['via']]
        VIA[(e['a'], e['b'])] = pts_; VIA[(e['b'], e['a'])] = pts_[::-1]
def route(a, b, off):
    if (a, b) in VIA: return [P(a) + off] + [v + off for v in VIA[(a, b)]] + [P(b) + off]
    if (a, b) in bent_route: return [P(a) + off, bent_route[(a, b)] + off, P(b) + off]
    if (b, a) in bent_route: return [P(a) + off, bent_route[(b, a)] + off, P(b) + off]
    return [P(a) + off, P(b) + off]

line_polys = []
for l in by_num:
    for ch in chains(line_adj[l]):
        pts = []
        for a, b in zip(ch, ch[1:]):
            e = edge_of[(a, b)]; off = offset_of(e, l)
            seg = route(a, b, off)
            pts += seg if not pts else seg[1:] if np.linalg.norm(seg[0] - pts[-1]) < 0.5 else seg
        clean = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - clean[-1]) > 0.5: clean.append(p)
        # drop collinear midpoints
        out = [clean[0]]
        for i in range(1, len(clean) - 1):
            u1 = clean[i] - out[-1]; u2 = clean[i + 1] - clean[i]
            if abs(u1[0] * u2[1] - u1[1] * u2[0]) > 1e-6 * np.linalg.norm(u1) * np.linalg.norm(u2) or u1 @ u2 < 0:
                out.append(clean[i])
        out.append(clean[-1])
        line_polys.append((l, out, ch))
        seg_owner_chain = None

# rounded corners: true circular arcs whose radius depends on the turn angle
def turn_radius(deg):
    if deg < 50: return 55.0      # 45deg turn: long gentle arc
    if deg < 100: return 38.0     # 90deg turn
    return 12.0                   # sharp 135deg turn
def rounded_path(pts, rmul=1.0):
    d = 'M%.2f %.2f' % tuple(pts[0])
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        u1 = p1 - p0; u2 = p2 - p1; l1, l2 = np.linalg.norm(u1), np.linalg.norm(u2)
        u1 /= l1; u2 /= l2
        cr = u1[0] * u2[1] - u1[1] * u2[0]
        th = math.acos(max(-1, min(1, u1 @ u2)))     # turning angle
        if th < 1e-3: continue
        r = turn_radius(math.degrees(th)) * rmul
        t = r * math.tan(th / 2)
        tmax = min(l1 * 0.5 if i - 1 > 0 else l1 * 0.9, l2 * 0.5 if i + 1 < len(pts) - 1 else l2 * 0.9)
        if t > tmax: t = tmax; r = t / math.tan(th / 2)
        a_, b_ = p1 - u1 * t, p1 + u2 * t
        d += ' L%.2f %.2f A%.2f %.2f 0 0 %d %.2f %.2f' % (a_[0], a_[1], r, r, 1 if cr > 0 else 0, b_[0], b_[1])
    d += ' L%.2f %.2f' % tuple(pts[-1])
    return d

def _overlaps():
    segs = [(l, p, q) for l, pts, _ in line_polys for p, q in zip(pts, pts[1:])]
    out = []
    for i in range(len(segs)):
        l1, p1, p2 = segs[i]; v = p2 - p1; L = np.linalg.norm(v)
        if L < 1: continue
        u = v / L
        for j in range(i + 1, len(segs)):
            l2, p3, p4 = segs[j]
            if l1 == l2: continue
            w = p4 - p3
            if abs(u[0] * w[1] - u[1] * w[0]) > 0.02 * np.linalg.norm(w): continue
            if abs((p3 - p1) @ np.array([-u[1], u[0]])) > LW * 0.8: continue
            a3, a4 = sorted(((p3 - p1) @ u, (p4 - p1) @ u))
            if min(L, a4) - max(0, a3) > 3:
                def near_st(p):
                    return min(st, key=lambda k: np.linalg.norm(P(k) - p))
                out.append((LINES[l1]['num'], LINES[l2]['num'], st[near_st(p1)]['name'], st[near_st(p2)]['name'], '|', st[near_st(p3)]['name'], st[near_st(p4)]['name']))
    return out
_ov = _overlaps(); print('overlapping line pieces:', len(_ov)); [print('  ', x) for x in _ov]

# ---------------------------------------------------------------- parallel bundling
# Wherever pieces of different lines run on top of each other, draw them side by side
# (like shared track on the official map): overlapping collinear pieces form a cluster,
# each line gets a slot, and every polyline is rebuilt from its offset pieces.
segs = []                      # (poly index, seg index)
for pi, (l, pts, _) in enumerate(line_polys):
    for si in range(len(pts) - 1): segs.append((pi, si))
def seg_pts(x): pi, si = x; pts = line_polys[pi][1]; return pts[si], pts[si + 1]
parent = list(range(len(segs)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i in range(len(segs)):
    p1, p2 = seg_pts(segs[i]); v = p2 - p1; L = np.linalg.norm(v)
    if L < 1: continue
    u = v / L; n = np.array([-u[1], u[0]])
    for j in range(i + 1, len(segs)):
        if line_polys[segs[i][0]][0] == line_polys[segs[j][0]][0]: continue
        p3, p4 = seg_pts(segs[j]); w = p4 - p3
        if abs(u[0] * w[1] - u[1] * w[0]) > 0.02 * (np.linalg.norm(w) + 1e-9): continue
        if abs((p3 - p1) @ n) > 1.0: continue
        a3, a4 = sorted(((p3 - p1) @ u, (p4 - p1) @ u))
        if min(L, a4) - max(0, a3) > 3: parent[find(i)] = find(j)
clusters = collections.defaultdict(list)
for i in range(len(segs)): clusters[find(i)].append(i)
seg_off = {}
bundled = []
for root, mem in clusters.items():
    lines_in = sorted({line_polys[segs[i][0]][0] for i in mem}, key=lambda l: LINES[l]['num'])
    if len(lines_in) < 2: continue
    p1, p2 = seg_pts(segs[mem[0]]); u = (p2 - p1) / np.linalg.norm(p2 - p1)
    if u[0] < -1e-9 or (abs(u[0]) < 1e-9 and u[1] < 0): u = -u          # canonical orientation
    n = np.array([-u[1], u[0]])
    for i in mem:
        slot = lines_in.index(line_polys[segs[i][0]][0])
        seg_off[segs[i]] = n * (slot - (len(lines_in) - 1) / 2) * PAR
    bundled.append((lines_in, [seg_pts(segs[i]) for i in mem], n))
new_polys = []
for pi, (l, pts, ch) in enumerate(line_polys):
    offs = [seg_off.get((pi, si), np.zeros(2)) for si in range(len(pts) - 1)]
    out = [pts[0] + offs[0]]
    for si in range(1, len(pts) - 1):
        a0, a1, b1 = pts[si - 1] + offs[si - 1], pts[si] + offs[si - 1], pts[si + 1] + offs[si]
        b0 = pts[si] + offs[si]
        d1, d2 = a1 - a0, b1 - b0
        M_ = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(M_)) > 1e-6:
            tt = np.linalg.solve(M_, b0 - a0); out.append(a0 + tt[0] * d1)
        else:
            out += [a1, b0]
    out.append(pts[-1] + offs[-1])
    new_polys.append((l, out, ch))
line_polys = new_polys
print('bundled corridors:', len(bundled))

SEGS = [(p, q) for _, pts, _ in line_polys for p, q in zip(pts, pts[1:])]
SEGS_A = np.array([s[0] for s in SEGS]); SEGS_B = np.array([s[1] for s in SEGS])

# local line direction at each station
st_dir = collections.defaultdict(dict)
for l in by_num:
    for sid, nb in line_adj[l].items():
        v = sum(((P(sid) - P(o)) / np.linalg.norm(P(sid) - P(o)) for o in nb), np.zeros(2)) if len(nb) == 2 else P(nb[0]) - P(sid)
        if np.linalg.norm(v) < 1e-6: v = P(nb[0]) - P(sid)
        if len(nb) == 2:
            a = P(nb[0]) - P(sid); v = a / np.linalg.norm(a)
        st_dir[sid][l] = v / np.linalg.norm(v)

# ---------------------------------------------------------------- water
ids_geo = list(lay['geo'])
G_ = np.array([lay['geo'][k] for k in ids_geo]); Pp = np.array([P(k) for k in ids_geo])
tri = Delaunay(G_); tree = cKDTree(G_)
def warp(pts):
    pts = np.asarray(pts, float); out = np.zeros_like(pts); simp = tri.find_simplex(pts)
    for i, (p, s_) in enumerate(zip(pts, simp)):
        if s_ >= 0:
            T = tri.transform[s_]; b = T[:2] @ (p - T[2]); out[i] = np.append(b, 1 - b.sum()) @ Pp[tri.simplices[s_]]
        else:
            dd, nn = tree.query(p, k=10); A = np.c_[G_[nn], np.ones(len(nn))]; w = 1 / (dd + 1)
            coef, *_ = np.linalg.lstsq(A * w[:, None], Pp[nn] * w[:, None], rcond=None); out[i] = np.append(p, 1) @ coef
    return out
lu = np.frombuffer(base64.b64decode(save['city']['lu']['b']), np.uint8).reshape(152, 152)
water = lu == 7
dl = distance_transform_edt(water)
start = None
for y in range(151, 100, -1):
    xs_ = np.where(water[y])[0]
    if len(xs_): start = (y, int(xs_[np.argmax(dl[y, xs_])])); break
best = {start: 0.0}; prev = {}; pq = [(0.0, start)]; end = None
while pq:
    c, (y, x) = heapq.heappop(pq)
    if c > best.get((y, x), 1e18): continue
    if y < 40 and dl[y, x] >= 5: end = (y, x); break
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny, nx = y + dy, x + dx
            if (dy or dx) and 0 <= ny < 152 and 0 <= nx < 152 and water[ny, nx]:
                nc = c + math.hypot(dy, dx) / min(dl[ny, nx], 3.0) ** 2
                if nc < best.get((ny, nx), 1e18): best[(ny, nx)] = nc; prev[(ny, nx)] = (y, x); heapq.heappush(pq, (nc, (ny, nx)))
river = []; n_ = end
while n_ is not None: river.append((n_[1] + 0.5, n_[0] + 0.5)); n_ = prev.get(n_)
river_w = warp(np.array(river[::-1]))
def douglas(pts, tol):
    def rec(i, j):
        if j <= i + 1: return [i, j]
        a, b = pts[i], pts[j]; ab = b - a; L = np.linalg.norm(ab) + 1e-9; w = pts[i + 1:j] - a
        d = np.abs(ab[0] * w[:, 1] - ab[1] * w[:, 0]) / L; k = int(np.argmax(d)) + i + 1
        return rec(i, k)[:-1] + rec(k, j) if d[k - i - 1] > tol else [i, j]
    return [pts[i] for i in rec(0, len(pts) - 1)]
def octi_poly(pts):
    """Polyline through the given vertices using only 0/45/90deg legs."""
    out = [pts[0]]
    for q in pts[1:]:
        out += two_leg(out[-1], q)[1:]
    return out

# Yangtze: southern shore of the wide water body in the north-east, simplified the same way
shore = []
for x in range(152):
    col = water[:, x]
    if not col[0]: continue
    y = 0
    while y + 1 < 152 and col[y + 1]: y += 1
    if y > 3: shore.append((x + 0.5, y + 0.5))
shore_s = octi_poly(douglas(warp(np.array(shore)), 260.0)) if shore else None
# the Huangpu ends where it meets the Yangtze. Cut the traced path while it is still inside
# the station area (the warp extrapolates poorly further out) and join it to the shore.
river_g = np.array(river[::-1])
cut = next((i for i, p in enumerate(river_g) if p[1] < 42), len(river_g) - 1)
river_w = river_w[:cut + 1]
if shore:
    sw = np.array(shore_s)
    best_ = None
    for a_, b_ in zip(sw, sw[1:]):
        ab = b_ - a_; t = np.clip((river_w[-1] - a_) @ ab / (ab @ ab + 1e-9), 0, 1); q = a_ + t * ab
        if best_ is None or np.linalg.norm(q - river_w[-1]) < best_[0]: best_ = (np.linalg.norm(q - river_w[-1]), q)
    river_w = np.vstack([river_w, best_[1] + (best_[1] - river_w[-1]) / (best_[0] + 1e-9) * 30])
river_s = octi_poly(douglas(river_w, 55.0))


# ---------------------------------------------------------------- labels & badges
def w_cn(t, fs=FS): return sum(fs if ord(c) > 0x2E80 else fs * 0.6 for c in t)
def w_en(t, fs=FS_EN): return sum(fs * (0.3 if c in "il.'·() " else 0.62 if c.isupper() else 0.54) for c in t)

JUNCTION = {sid for sid, s in st.items() if len(s['lines']) > 1 or any(len(line_adj[l].get(sid, [])) > 2 for l in s['lines'])}
R_INT = 7.5
# capsule stations: ends of shared track
capsule = {}
for lines_in, pieces, n in bundled:
    for sid, s_ in st.items():
        served = [l for l in lines_in if l in s_['lines']]
        if len(served) < 2: continue
        p = P(sid)
        for a, b in pieces:
            ab = b - a; t = (p - a) @ ab / (ab @ ab)
            if -0.01 <= t <= 1.01 and np.linalg.norm(a + np.clip(t, 0, 1) * ab - p) < 1.0:
                capsule[sid] = (n, (len(lines_in) - 1) / 2 * PAR); break
for e in []:
    if len(e['lines']) > 1:
        a, b = P(e['a']), P(e['b']); v = (b - a) / np.linalg.norm(b - a)
        half = (len(e['lines']) - 1) / 2 * PAR
        for sid in (e['a'], e['b']): capsule[sid] = (np.array([-v[1], v[0]]), half)
def st_box(sid):
    x, y = P(sid); r = R_INT if sid in JUNCTION else 4.5
    if sid in capsule:
        n, h = capsule[sid]; ex = abs(n[0]) * h; ey = abs(n[1]) * h
        return (x - r - ex, y - r - ey, x + r + ex, y + r + ey)
    return (x - r, y - r, x + r, y + r)
st_boxes = {sid: st_box(sid) for sid in st}

def rect_hits_segs(b, pad=1.5):
    x0, y0, x1, y1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    A, B = SEGS_A, SEGS_B
    m = (np.minimum(A[:, 0], B[:, 0]) <= x1) & (np.maximum(A[:, 0], B[:, 0]) >= x0) & \
        (np.minimum(A[:, 1], B[:, 1]) <= y1) & (np.maximum(A[:, 1], B[:, 1]) >= y0)
    cnt = 0
    for a, b_ in zip(A[m], B[m]):
        t0, t1 = 0.0, 1.0; dx, dy = b_[0] - a[0], b_[1] - a[1]; ok = True
        for pp, qq in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
            if abs(pp) < 1e-9:
                if qq < 0: ok = False; break
            else:
                t = qq / pp
                if pp < 0: t0 = max(t0, t)
                else: t1 = min(t1, t)
                if t0 > t1: ok = False; break
        cnt += ok
    return cnt
def ov(a, b): return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

OCT = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
PREF = [0.0, 0.35, 0.3, 0.45, 0.1, 0.45, 0.3, 0.35]
def cands(cx, cy, w, h, gap):
    out = []
    for ring, g in enumerate((gap, gap + 10, gap + 24)):
        for k, (ux, uy) in enumerate(OCT):
            x0 = cx + g * (0.72 if uy else 1) if ux > 0 else cx - g * (0.72 if uy else 1) - w if ux < 0 else cx - w / 2
            y0 = cy + g * (0.72 if ux else 1) if uy > 0 else cy - g * (0.72 if ux else 1) - h if uy < 0 else cy - h / 2
            out.append((k, ring, (x0, y0, x0 + w, y0 + h)))
    return out

items = []
for sid, s in st.items():
    en = en_name(s['name']); s['en'] = en
    w = max(w_cn(s['name']), w_en(en)); h = FS * 1.08 + FS_EN * 1.25
    gap = (R_INT if sid in JUNCTION else 4.5) + 5 + (capsule[sid][1] if sid in capsule else 0)
    items.append({'kind': 'label', 'sid': sid, 'w': w, 'h': h, 'gap': gap})
BADGE_H = 30.0
def badge_w(num): return 18 + 13 * len(num)
for l in by_num:
    for n, nb in line_adj[l].items():
        if len(nb) == 1:
            num = str(LINES[l]['num']); bw = badge_w(num)
            v = P(n) - P(nb[0]); v /= np.linalg.norm(v)
            items.append({'kind': 'badge', 'sid': n, 'line': l, 'bw': bw, 'w': bw + 4 + 38, 'h': BADGE_H,
                          'gap': R_INT + 12, 'ext': v})
for it in items:
    s = st[it['sid']]; it['cands'] = cands(s['x'], s['y'], it['w'], it['h'], it['gap'])

def cost(it, cand, others):
    k, ring, box = cand
    c = 18 * rect_hits_segs(box) + 25 * sum(1 for b in st_boxes.values() if ov(box, b) > 0)
    c += sum(45 + ov(box, o) * 0.2 for o in others if ov(box, o) > 0)
    c += ring * 1.5
    if it['kind'] == 'label':
        c += PREF[k] * 2
    else:
        cx_ = (box[0] + box[2]) / 2 - st[it['sid']]['x']; cy_ = (box[1] + box[3]) / 2 - st[it['sid']]['y']
        c += 3 * (1 - (cx_ * it['ext'][0] + cy_ * it['ext'][1]) / (math.hypot(cx_, cy_) + 1e-9))
    return c
items.sort(key=lambda it: (sum(1 for c in it['cands'] if rect_hits_segs(c[2]) == 0), it['kind'] != 'badge'))
for it in items: it['choice'] = None; it['box'] = (0, 0, 0, 0)
for rnd in range(8):
    for it in items:
        others = [o['box'] for o in items if o is not it and o['choice'] is not None]
        best_ = min(it['cands'], key=lambda c: cost(it, c, others))
        it['choice'] = best_; it['box'] = best_[2]; it['cost'] = cost(it, best_, others)
bad = [(st[it['sid']]['name'], it['kind']) for it in items if it['cost'] >= 12]
print('labels/badges touching something:', len(bad), bad[:30])

# ---------------------------------------------------------------- SVG
xs = [s['x'] for s in st.values()] + [it['box'][0] for it in items] + [it['box'][2] for it in items]
ys = [s['y'] for s in st.values()] + [it['box'][1] for it in items] + [it['box'][3] for it in items]
PADL, PADR, PADT, PADB = 80, 80, 190, 330
X0, Y0 = min(xs) - PADL, min(ys) - PADT
W, H = max(xs) - X0 + PADR, max(ys) - Y0 + PADB
def esc(t): return t.replace('&', '&amp;').replace("'", '&#39;')

o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{X0:.0f} {Y0:.0f} {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" font-family=\'{FONT}\'>',
     f'<rect x="{X0:.0f}" y="{Y0:.0f}" width="{W:.0f}" height="{H:.0f}" fill="#fff"/>']
if shore_s is not None:
    pts = list(shore_s)
    poly = pts + [np.array([X0 + W, pts[-1][1]]), np.array([X0 + W, Y0]), np.array([pts[0][0], Y0])]
    o.append(f'<path d="{rounded_path([np.array(p) for p in poly], 3.0)} Z" fill="{WATER}"/>')
o.append(f'<path d="{rounded_path(river_s, 4.0)}" fill="none" stroke="{WATER}" stroke-width="40" stroke-linecap="butt" stroke-linejoin="round"/>')

for l, pts, _ in line_polys:
    o.append(f'<path d="{rounded_path(pts)}" fill="none" stroke="{LINES[l]["color"]}" stroke-width="{LW}" stroke-linecap="butt" stroke-linejoin="round"/>')

lab = {it['sid']: it for it in items if it['kind'] == 'label'}
for sid, s in st.items():
    x, y = s['x'], s['y']
    if sid in capsule:
        n, h = capsule[sid]; ang = math.degrees(math.atan2(n[1], n[0]))
        o.append(f'<rect x="{-h - R_INT:.1f}" y="{-R_INT:.1f}" width="{2 * h + 2 * R_INT:.1f}" height="{2 * R_INT:.1f}" rx="{R_INT}" '
                 f'transform="translate({x:.1f} {y:.1f}) rotate({ang:.1f})" fill="#fff" stroke="{INK}" stroke-width="1.8"/>')
    elif sid in JUNCTION:
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{R_INT}" fill="#fff" stroke="{INK}" stroke-width="1.8"/>')
    else:
        l = s['lines'][0]; v = st_dir[sid].get(l, np.array([1.0, 0])); n = np.array([-v[1], v[0]])
        b = lab[sid]['box']; c = np.array([(b[0] + b[2]) / 2 - x, (b[1] + b[3]) / 2 - y])
        if n @ c < 0: n = -n
        a_ = np.array([x, y]); bestd = None
        for l2, pts2, _ in line_polys:            # the line may be shifted sideways inside a bundle
            if l2 != l: continue
            for p_, q_ in zip(pts2, pts2[1:]):
                ab = q_ - p_; t = np.clip((np.array([x, y]) - p_) @ ab / (ab @ ab + 1e-9), 0, 1); c_ = p_ + t * ab
                dd = np.linalg.norm(c_ - [x, y])
                if bestd is None or dd < bestd: bestd = dd; a_ = c_
        b_ = a_ + n * (LW / 2 + 5.5)
        o.append(f'<line x1="{a_[0]:.1f}" y1="{a_[1]:.1f}" x2="{b_[0]:.1f}" y2="{b_[1]:.1f}" stroke="{LINES[l]["color"]}" stroke-width="{LW * 0.8:.1f}"/>')

for it in items:
    x0, y0, x1, y1 = it['box']
    if it['kind'] == 'label':
        s = st[it['sid']]; ux = OCT[it['choice'][0]][0]
        anchor, tx = ('start', x0) if ux > 0 else ('end', x1) if ux < 0 else ('middle', (x0 + x1) / 2)
        o.append(f'<text x="{tx:.1f}" y="{y0 + FS * 0.93:.1f}" font-size="{FS}" text-anchor="{anchor}" fill="{INK}" '
                 f'stroke="#fff" stroke-width="3" stroke-linejoin="round" paint-order="stroke">{s["name"]}</text>')
        o.append(f'<text x="{tx:.1f}" y="{y1 - 1:.1f}" font-size="{FS_EN}" text-anchor="{anchor}" fill="#333" font-family="{FONT_EN}" '
                 f'stroke="#fff" stroke-width="2" stroke-linejoin="round" paint-order="stroke">{esc(s["en"])}</text>')
    else:
        l = it['line']; num = str(LINES[l]['num']); col = LINES[l]['color']; bw = it['bw']
        o.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{BADGE_H}" fill="{col}"/>')
        o.append(f'<text x="{x0 + bw / 2:.1f}" y="{y0 + 24:.1f}" font-size="25" font-family="{FONT_EN}" fill="#fff" text-anchor="middle">{num}</text>')
        o.append(f'<text x="{x0 + bw + 3:.1f}" y="{y0 + 17:.1f}" font-size="17" fill="{INK}" font-weight="600">号线</text>')
        o.append(f'<text x="{x0 + bw + 4:.1f}" y="{y0 + 28:.1f}" font-size="8" font-family="{FONT_EN}" fill="{INK}">Line {num}</text>')

nst = len(st); nl = len(LINES)
o.append(f'<text x="{X0 + 70:.0f}" y="{Y0 + 80:.0f}" font-size="40" font-weight="700" fill="{INK}">上海轨道交通网络示意图</text>')
o.append(f'<text x="{X0 + 72:.0f}" y="{Y0 + 108:.0f}" font-size="16" font-family="{FONT_EN}" fill="{INK}" letter-spacing="1">SHANGHAI METRO NETWORK MAP</text>')
o.append(f'<text x="{X0 + 72:.0f}" y="{Y0 + 136:.0f}" font-size="15" fill="#666">{lay["meta"]["name"]} · 游戏时间 {lay["meta"]["date"]} · {nl} 条线路 · {nst} 座车站</text>')
lx0, ly0 = X0 + 70, Y0 + H - PADB + 80
cols = 8; cw = (W - 140) / cols
for i, l in enumerate(by_num):
    cx_ = lx0 + (i % cols) * cw; cy_ = ly0 + (i // cols) * 40
    num = str(LINES[l]['num']); bw = badge_w(num) * 0.8
    o.append(f'<rect x="{cx_:.1f}" y="{cy_ - 20:.1f}" width="{bw:.1f}" height="24" fill="{LINES[l]["color"]}"/>')
    o.append(f'<text x="{cx_ + bw / 2:.1f}" y="{cy_:.1f}" font-size="20" font-family="{FONT_EN}" fill="#fff" text-anchor="middle">{num}</text>')
    o.append(f'<text x="{cx_ + bw + 5:.1f}" y="{cy_ - 6:.1f}" font-size="14" fill="{INK}" font-weight="600">号线 <tspan font-weight="400" fill="#888" font-size="12">{len(line_adj[l])} 站</tspan></text>')
    o.append(f'<text x="{cx_ + bw + 5:.1f}" y="{cy_ + 4:.1f}" font-size="8" font-family="{FONT_EN}" fill="#555">Line {num}</text>')
o.append(f'<line x1="{lx0:.0f}" y1="{ly0 - 46:.0f}" x2="{X0 + W - 70:.0f}" y2="{ly0 - 46:.0f}" stroke="#ddd"/>')
ly1 = ly0 + ((len(by_num) - 1) // cols + 1) * 40 + 18
o.append(f'<circle cx="{lx0 + 8:.0f}" cy="{ly1:.0f}" r="{R_INT}" fill="#fff" stroke="{INK}" stroke-width="1.8"/><text x="{lx0 + 24:.0f}" y="{ly1 + 5:.0f}" font-size="14" fill="{INK}">换乘车站 <tspan font-family="{FONT_EN}" font-size="9">Transfer Station</tspan></text>')
o.append(f'<line x1="{lx0 + 210:.0f}" y1="{ly1 + 3:.0f}" x2="{lx0 + 240:.0f}" y2="{ly1 + 3:.0f}" stroke="#888" stroke-width="{LW}"/><line x1="{lx0 + 225:.0f}" y1="{ly1:.0f}" x2="{lx0 + 225:.0f}" y2="{ly1 - 9:.0f}" stroke="#888" stroke-width="{LW * 0.8:.1f}"/><text x="{lx0 + 250:.0f}" y="{ly1 + 5:.0f}" font-size="14" fill="{INK}">单线车站 <tspan font-family="{FONT_EN}" font-size="9">Non-transfer Station</tspan></text>')
note = '本图经变形处理，不反映实际地理位置。'
if recolored: note += '存档中 19 号线起沿用了 1–18 号线的颜色，本图为其重新配色以便区分。'
o.append(f'<text x="{lx0 + 470:.0f}" y="{ly1 + 5:.0f}" font-size="13" fill="#888">{note}</text>')
o.append('</svg>')
open(out_path, 'w').write('\n'.join(o))
print('size', round(W), round(H))
