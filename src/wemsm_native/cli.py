"""
Command line interface.

    wemsm-native status     --config cfg.py   # scan output_dir, show the plan
    wemsm-native features   --config cfg.py
    wemsm-native tica       --config cfg.py
    wemsm-native cluster    --config cfg.py
    wemsm-native properties --config cfg.py
    wemsm-native msm        --config cfg.py
    wemsm-native fe         --config cfg.py
    wemsm-native all        --config cfg.py

    # explicit rebuilds / quick parameter changes without editing the file
    wemsm-native all --config cfg.py --set cluster.n_clusters=200 --set msm.rebuild=true

Also available as ``python -m wemsm_native ...``.
"""

from __future__ import annotations

import argparse
import sys

from . import store
from .config import apply_override, load_config
from .pipeline import STAGES, Pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='wemsm-native',
                                 description='MSMs from weighted-ensemble simulations')
    ap.add_argument('command', choices=['status', 'all'] + STAGES)
    ap.add_argument('--config', required=True, help='.py (defining CONFIG), .toml or .json')
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                    help='override a config value (repeatable)')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for a in args.set:
        apply_override(cfg, a)
    cfg.validate()

    try:
        pipe = Pipeline(cfg)
        if args.command == 'status':
            found = store.scan(cfg.output_dir)
            print(f'{len(found)} artifact(s) found under {cfg.output_dir}:')
            for m in found:
                print(f'  {m.get("created")}  {m.get("stage") or "?":10s} {m["path"]}')
            print()
            pipe.check(STAGES)
            return 0
        pipe.run(STAGES if args.command == 'all' else [args.command])
    except store.StaleArtifactError as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
