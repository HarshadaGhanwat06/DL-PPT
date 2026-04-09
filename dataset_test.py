from __future__ import annotations

from pathlib import Path

import numpy as np


def inspect_split(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    print(f"\nSplit: {path.stem}")
    print(f"x shape: {data['x'].shape}")
    print(f"y shape: {data['y'].shape}")
    if 'target_names' in data.files:
        print(f"targets: {list(data['target_names'])}")
    print('first 3 labels:')
    print(data['y'][:3])


def main() -> None:
    dataset_dir = Path('outputs/datasets')
    for split_name in ('train', 'val', 'test'):
        inspect_split(dataset_dir / f'{split_name}.npz')


if __name__ == '__main__':
    main()
