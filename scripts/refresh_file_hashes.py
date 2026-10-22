"""Record the SHA-256 of local code and newly created files in the configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def refresh(root: Path) -> list[str]:
    root = root.resolve()
    configurations = {p.resolve() for p in (root / 'configs').rglob('*.json')}
    visited: set[Path] = set()
    active: set[Path] = set()
    changed: list[str] = []

    def resolve(value: object) -> Path | None:
        if not isinstance(value, str) or '<' in value:
            return None
        path = Path(value)
        if not path.is_absolute(): path = root / path
        return path.resolve() if path.is_file() else None

    def bound_hash(path: Path) -> str:
        if path in configurations: visit(path)
        return digest(path)

    def update(value: object) -> None:
        if isinstance(value, list):
            for item in value: update(item)
        elif isinstance(value, dict):
            for item in list(value.values()): update(item)
            for path_key in ("compressed_artifact_path", "reconstruction_path", "builder_path",
                             "online_adapter_path", "roundtrip_verifier_path", "training_metrics_path",
                             "source_artifact_path", "source_builder_path", "source_training_metrics_path",
                             "source_model_weights_path"):
                if path := resolve(value.get(path_key)):
                    value[path_key.removesuffix("_path") + "_sha256"] = bound_hash(path)
            # Explicit path/hash pairs, including the protocol's locked-file map.
            for key, old in list(value.items()):
                if key == 'locked_files' and isinstance(old, dict):
                    for name in old:
                        if path := resolve(name): old[name] = bound_hash(path)
                elif isinstance(old, str) and len(old) == 64 and all(c in '0123456789abcdef' for c in old):
                    prefix = key.removesuffix('_sha256')
                    candidates = ['path', 'file'] if key == 'sha256' else [prefix, prefix + '_path', prefix + '_file']
                    if key.endswith('_sha256') or key == 'sha256':
                        for candidate in candidates:
                            if path := resolve(value.get(candidate)):
                                value[key] = bound_hash(path)
                                break
                    elif path := resolve(key):
                        value[key] = bound_hash(path)
            if 'root' in value and isinstance(value.get('files'), dict):
                base = Path(value['root'])
                if not base.is_absolute(): base = root / base
                for name in value['files']:
                    if (base / name).is_file(): value['files'][name] = bound_hash((base / name).resolve())

    def visit(path: Path) -> None:
        if path in visited: return
        if path in active:
            raise ValueError(f'Circular configuration hash reference: {path}')
        active.add(path)
        original = path.read_text()
        value = json.loads(original)
        update(value)
        rendered = json.dumps(value, indent=2) + '\n'
        if rendered != original:
            path.write_text(rendered)
            changed.append(str(path.relative_to(root)))
        active.remove(path)
        visited.add(path)

    for path in sorted(configurations): visit(path)
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps({'updated': refresh(args.root)}, indent=2))
