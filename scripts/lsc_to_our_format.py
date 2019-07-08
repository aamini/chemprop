from argparse import ArgumentParser
import os
import shutil

import h5py
import numpy as np


DATASETS = [
    'qm7',
    # 'qm8',
    # 'qm9',
    # 'delaney',
    # 'freesolv',
    # 'lipo',
    # 'pdbbind_full',
    # 'pdbbind_core',
    # 'pdbbind_refined',
    # 'pcba',
    # 'muv',
    # 'hiv',
    # 'bace',
    # 'bbbp',
    # 'tox21',
    # 'toxcast',
    # 'sider',
    # 'clintox',
    # 'chembl'
]


def lsc_to_our_format(lsc_dir: str, ckpt_dir: str, save_dir: str):
    os.makedirs(save_dir)

    for dataset in DATASETS:
        # Create directory where things will be copied
        dataset_save_dir = os.path.join(save_dir, dataset, 'scaffold')
        os.makedirs(dataset_save_dir)

        # Convert preds and copy over preds and targets
        for fold in range(10):
            lsc_preds_path = os.path.join(lsc_dir, dataset, 'test', f'fold_{fold}', 'semi', 'o0003.evalPredict.hdf5')
            ckpt_targets_path = os.path.join(ckpt_dir, '417_default', dataset, 'scaffold', str(fold), 'targets.npy')

            if not (os.path.exists(lsc_preds_path) and os.path.exists(ckpt_targets_path)):
                continue

            save_preds_path = os.path.join(dataset_save_dir, str(fold), 'preds.npy')
            save_targets_path = os.path.join(dataset_save_dir, str(fold), 'targets.npy')

            # Copy targets
            shutil.copy(ckpt_targets_path, save_targets_path)

            # Convert and copy preds
            preds_file = h5py.File(lsc_preds_path)
            preds = np.array(preds_file['predictions'])
            np.save(save_preds_path, preds)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lsc_dir', type=str, required=True,
                        help='Path to directory in lsc save format')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to directory with targets saved in our format')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory where lsc files will be saved in our format')
    args = parser.parse_args()

    lsc_to_our_format(
        lsc_dir=args.lsc_dir,
        ckpt_dir=args.ckpt_dir,
        save_dir=args.save_dir
    )
