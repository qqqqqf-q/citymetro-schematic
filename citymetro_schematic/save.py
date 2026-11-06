"""Reading 城市与地铁 (City & Metro) save files."""
import json


class SaveError(ValueError):
    pass


def load(path):
    """Parsed save body: the dict holding ``net``, ``city``, ``meta`` and ``clock``."""
    with open(path, encoding='utf-8') as f:
        try:
            outer = json.load(f)
        except json.JSONDecodeError as e:
            raise SaveError(f'not a JSON file: {e}') from None
    body = outer.get('body') if isinstance(outer, dict) else None
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError as e:
            raise SaveError(f'save body is not valid JSON: {e}') from None
    if not isinstance(body, dict) or not isinstance(body.get('net'), dict):
        raise SaveError('no metro network found; expected a .cmsave.json exported from the game')
    net = body['net']
    for key in ('stations', 'segments', 'lines'):
        if not isinstance(net.get(key), dict):
            raise SaveError(f'save has no net.{key}')
    if not net['segments']:
        raise SaveError('the save has no built track yet')
    return body
