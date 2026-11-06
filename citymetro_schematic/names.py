"""English station names and city names.

Station names are romanised with pypinyin plus a few suffix rules (路 -> Rd., 公园 -> Park,
a trailing 东/西/南/北 -> "(E)" ...). Each game map may ship a JSON file in ``maps/``
(named after the save's ``meta.map`` id) with the city name and exact translations for
names the rules get wrong; users can add more with ``--names``.
"""
import json
import re
from importlib import resources

from pypinyin import lazy_pinyin

SUFFIX = [('1号航站楼', 'Airport Terminal 1'), ('2号航站楼', 'Airport Terminal 2'), ('火车站', 'Railway Station'),
          ('工业园区', 'Industrial Park'), ('体育中心', 'Sports Center'), ('足球场', 'Football Stadium'),
          ('博物馆', 'Museum'), ('公园', 'Park'), ('花园', 'Garden'), ('大学', 'University'), ('大道', 'Ave.'),
          ('机场', 'Airport'), ('新镇', 'New Town'), ('1站', 'Station 1'), ('路', 'Rd.'), ('镇', 'Town')]
DIRN = {'东': 'E', '西': 'W', '南': 'S', '北': 'N'}


def load_map_data(map_id):
    """Bundled data for a game map id (e.g. 'sh'), or {} if there is none."""
    if not map_id or not re.fullmatch(r'[A-Za-z0-9_-]+', map_id):
        return {}
    f = resources.files(__package__).joinpath('maps', f'{map_id}.json')
    return json.loads(f.read_text('utf-8')) if f.is_file() else {}


def pinyin(s):
    out = ''
    for i, syl in enumerate(lazy_pinyin(s)):
        if i and syl[:1] in 'aoe': out += "'"
        out += syl
    return out[:1].upper() + out[1:]


def city_names(meta, map_data):
    """(Chinese, English) city name for the title."""
    city = map_data.get('city') or {}
    zh = city.get('zh')
    if not zh:
        # scenario names look like "经营挑战 · 上海 1990": take the last part, drop the year
        part = re.split(r'\s*[·・]\s*', meta.get('name') or '')[-1]
        zh = re.sub(r'\s*\d+\s*$', '', part).strip()
    en = city.get('en') or (pinyin(zh) if zh else '')
    return zh, en


def translator(exact):
    """Return a function mapping a Chinese station name to an English one.

    ``exact`` maps whole names (or whole parts of a name joined with '·') to English;
    it is only used for exact matches, so an entry like 上海 -> "Shanghai Railway Station"
    does not leak into 上海大学.
    """
    def part(n):
        if n in exact: return exact[n]
        d = ''
        if len(n) >= 3 and n[-1] in DIRN:
            base = n[:-1]
            if base in exact or any(base.endswith(z) for z, _ in SUFFIX) or len(base) >= 2:
                d = ' (%s)' % DIRN[n[-1]]; n = base
        if n in exact: return exact[n] + d
        for z, e in SUFFIX:
            if n.endswith(z) and len(n) > len(z):
                return f'{pinyin(n[:-len(z)])} {e}{d}'
        return pinyin(n) + d
    return lambda n: ' · '.join(part(p) for p in n.split('·'))
