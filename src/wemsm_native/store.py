"""
store.py
========

Cached artifacts and the reuse rule.

Every product is a pickle plus a small JSON sidecar (``<file>.meta.json``)
holding:

    id        unique id of this build
    stage     producing stage
    params    the configuration values that define the product
    upstream  {name: id} of the artifacts it was built from
    created   timestamp

Reuse rule
----------
An existing artifact is always reused unless its stage's rebuild flag is set.
It is never silently replaced: if its recorded params or upstream ids disagree
with the current configuration, ``check`` reports a conflict naming the flag
that must be set. Sidecars make scanning an output directory cheap, so the
whole plan is checked before any expensive computation starts.

When a flag forces a rebuild, the previous file is kept as ``<file>.prev``.
"""

from __future__ import annotations

import json
import os
import pickle as pkl
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class StaleArtifactError(RuntimeError):
    """A cached artifact disagrees with the configuration or its inputs."""


def meta_path(path: str) -> str:
    return path + '.meta.json'


def read_meta(path: str) -> Optional[Dict[str, Any]]:
    mp = meta_path(path)
    if not os.path.isfile(path):
        return None
    if not os.path.isfile(mp):
        return {'id': None, 'stage': None, 'params': None, 'upstream': {},
                'created': None, 'missing_sidecar': True}
    with open(mp) as fh:
        return json.load(fh)


def save(path: str, data: Any, stage: str, params: Dict[str, Any],
         upstream: Dict[str, str]) -> Dict[str, Any]:
    """Write ``data`` and its sidecar; keep any previous version as ``.prev``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.isfile(path):
        shutil.move(path, path + '.prev')
        if os.path.isfile(meta_path(path)):
            shutil.move(meta_path(path), meta_path(path + '.prev'))
    meta = {'id': uuid.uuid4().hex, 'stage': stage,
            'params': json.loads(json.dumps(params, default=str)),
            'upstream': dict(upstream),
            'created': time.strftime('%Y-%m-%d %H:%M:%S')}
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        pkl.dump(data, fh, protocol=pkl.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    with open(meta_path(path), 'w') as fh:
        json.dump(meta, fh, indent=1)
    return meta


def load(path: str) -> Any:
    with open(path, 'rb') as fh:
        return pkl.load(fh)


def _diff(old: Any, new: Any, prefix: str = '') -> List[str]:
    """Human-readable differences between two JSON-like values."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new)):
            p = f'{prefix}.{k}' if prefix else str(k)
            if k not in old:
                out.append(f'{p}: (absent) -> {new[k]!r}')
            elif k not in new:
                out.append(f'{p}: {old[k]!r} -> (absent)')
            else:
                out.extend(_diff(old[k], new[k], p))
        return out
    return [] if old == new else [f'{prefix}: {old!r} -> {new!r}']


# --------------------------------------------------------------------------- #
@dataclass
class Item:
    """One planned artifact."""
    name: str                     # unique key, e.g. 'features:clust2'
    stage: str
    path: str
    params: Dict[str, Any]
    flag: str                     # config flag that forces a rebuild
    force: bool
    upstream: List[str] = field(default_factory=list)   # names of other Items
    action: str = ''              # 'reuse' | 'build' | 'rebuild' | 'conflict'
    reasons: List[str] = field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None


class Plan:
    """Decides REUSE / BUILD / REBUILD for every artifact before running."""

    def __init__(self) -> None:
        self.items: Dict[str, Item] = {}

    def add(self, item: Item) -> Item:
        item.params = json.loads(json.dumps(item.params, default=str))
        self.items[item.name] = item
        return item

    def resolve(self) -> None:
        for it in self.items.values():                 # insertion order = DAG order
            it.meta = read_meta(it.path)
            it.reasons = []
            if it.force:
                it.action = 'rebuild' if it.meta else 'build'
                continue
            if it.meta is None:
                it.action = 'build'
                continue
            if it.meta.get('missing_sidecar'):
                it.action = 'conflict'
                it.reasons.append(f'{it.path} has no metadata sidecar, so it '
                                  f'cannot be checked against this configuration')
                continue
            it.reasons.extend(_diff(it.meta['params'], it.params))
            for up_name in it.upstream:
                up = self.items[up_name]
                if up.action in ('build', 'rebuild'):
                    it.reasons.append(f'input {up_name} will be '
                                      f'{"re" if up.action == "rebuild" else ""}built')
                elif up.action == 'conflict':
                    it.reasons.append(f'input {up_name} is itself out of date')
                elif up.meta and it.meta['upstream'].get(up_name) != up.meta.get('id'):
                    it.reasons.append(f'input {up_name} changed since this file '
                                      f'was built')
            it.action = 'conflict' if it.reasons else 'reuse'

    def report(self) -> str:
        lines = []
        for it in self.items.values():
            lines.append(f'  {it.action.upper():8s} {it.name:40s} {it.path}')
        return '\n'.join(lines)

    def raise_on_conflict(self) -> None:
        bad = [it for it in self.items.values() if it.action == 'conflict']
        if not bad:
            return
        flags = sorted({it.flag for it in bad})
        msg = ['Existing output files do not match this configuration. Nothing '
               'was recomputed.']
        for it in bad:
            msg.append(f'\n  {it.path}')
            msg.extend(f'      - {r}' for r in it.reasons)
        msg.append('\nTo rebuild them explicitly, set ' +
                   ', '.join(f'`{f}=True`' for f in flags) +
                   ' (in the config, or with --set ' +
                   ' --set '.join(f'{f}=true' for f in flags) +
                   '), or use a different output_dir.')
        raise StaleArtifactError('\n'.join(msg))

    def upstream_ids(self, name: str) -> Dict[str, str]:
        out = {}
        for up in self.items[name].upstream:
            m = read_meta(self.items[up].path)
            if m is None or m.get('id') is None:
                raise StaleArtifactError(f'{name}: input {up} is missing')
            out[up] = m['id']
        return out

    def run(self, name: str, build) -> Tuple[Any, Dict[str, Any]]:
        """Load the artifact if planned for reuse, else build and save it."""
        it = self.items[name]
        if it.action == 'reuse':
            print(f'  [reuse] {it.path}')
            return load(it.path), it.meta
        if it.action == 'conflict':
            raise StaleArtifactError(f'{name} is in conflict; resolve the plan first')
        print(f'  [{it.action}] {it.path}')
        data = build()
        it.meta = save(it.path, data, it.stage, it.params, self.upstream_ids(name))
        return data, it.meta


def scan(output_dir: str) -> List[Dict[str, Any]]:
    """Every artifact sidecar found under ``output_dir``."""
    found = []
    for root, _, files in os.walk(output_dir):
        for f in sorted(files):
            if f.endswith('.meta.json') and not f.endswith('.prev.meta.json'):
                p = os.path.join(root, f[:-len('.meta.json')])
                with open(os.path.join(root, f)) as fh:
                    m = json.load(fh)
                m['path'] = p
                m['exists'] = os.path.isfile(p)
                found.append(m)
    return found
