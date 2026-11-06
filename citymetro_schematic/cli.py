"""Command line: save file in, schematic SVG (and PNG) out."""
import argparse
import json
import os
import shutil
import subprocess
import sys

from . import prelayout, render, straighten
from .names import load_map_data
from .save import SaveError, load


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='citymetro-schematic',
        description='把《城市与地铁》存档画成官方风格的地铁线路示意图（SVG，可选 PNG）。')
    ap.add_argument('save', help='游戏导出的 .cmsave.json 存档')
    ap.add_argument('-o', '--output', help='输出文件，.svg 或 .png；默认与存档同名的 .svg 和 .png')
    ap.add_argument('--title', help='自定义中文标题（默认「<城市>轨道交通网络示意图」）')
    ap.add_argument('--names', help='英文站名对照 JSON，格式 {"中文站名": "English"}，覆盖自动译名')
    ap.add_argument('--png-width', type=int, default=4200, help='PNG 宽度（像素），默认 4200')
    ap.add_argument('--no-png', action='store_true', help='只输出 SVG')
    ap.add_argument('--fisheye', type=float, default=0.5,
                    help='市中心放大程度，0 不变形，越大中心越疏、郊区越紧，默认 0.5')
    ap.add_argument('--center', metavar='站名', help='以哪个车站为市中心（默认取地图预设，没有则自动找车站最密处）')
    ap.add_argument('--gap', type=float, default=1.1, help='车站最小间距（布局单位），默认 1.1')
    ap.add_argument('--straight-tol', type=float, default=12.0,
                    help='一段线路偏离直线多少度以内拉成一条直线，默认 12')
    ap.add_argument('--simplify', type=float, default=0.8,
                    help='拉直线路时允许偏离原走向的程度，越大转折越少，默认 0.8')
    ap.add_argument('--keep-layout', metavar='JSON', help='另存最终布局（调试用）')
    ap.add_argument('-q', '--quiet', action='store_true', help='不输出过程信息')
    a = ap.parse_args(argv)

    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, file=sys.stderr))
    try:
        save = load(a.save)
    except (OSError, SaveError) as e:
        print(f'无法读取存档：{e}', file=sys.stderr)
        return 2
    names = None
    if a.names:
        with open(a.names, encoding='utf-8') as f:
            names = json.load(f)

    base = os.path.splitext(a.save[:-len('.json')] if a.save.endswith('.cmsave.json') else a.save)[0]
    svg_path, png_path = base + '.svg', None if a.no_png else base + '.png'
    if a.output:
        stem, ext = os.path.splitext(a.output)
        if ext.lower() == '.png':
            svg_path, png_path = stem + '.svg', a.output
        else:
            svg_path, png_path = a.output, None if a.no_png else stem + '.png'

    log('1/3 初步排版…')
    center = a.center or load_map_data(save.get('meta', {}).get('map')).get('center')
    lay = prelayout.run(save, fisheye=a.fisheye, min_gap=a.gap, center_name=center, log=log)
    log('2/3 拉直线路…')
    lay = straighten.run(lay, STRAIGHT_TOL=a.straight_tol, DP_TOL=a.simplify, log=log)
    if a.keep_layout:
        with open(a.keep_layout, 'w', encoding='utf-8') as f:
            json.dump(lay, f, ensure_ascii=False, indent=1)
    log('3/3 绘制…')
    svg = render.run(lay, save, title=a.title, names=names, log=log)
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(svg_path)

    if png_path:
        tool = shutil.which('rsvg-convert')
        if tool:
            subprocess.run([tool, '-w', str(a.png_width), '-o', png_path, svg_path], check=True)
            print(png_path)
        else:
            print('未找到 rsvg-convert，跳过 PNG（macOS: brew install librsvg；Debian/Ubuntu: apt install librsvg2-bin）。'
                  '也可以用浏览器打开 SVG 后截图或打印。', file=sys.stderr)
    return 0
