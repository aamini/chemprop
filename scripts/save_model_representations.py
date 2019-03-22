from argparse import ArgumentParser
import torch
import numpy as np
import pickle
import os

from chemprop.parsing import update_checkpoint_args
from chemprop.features import save_features
from chemprop.utils import load_checkpoint

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV')
    parser.add_argument('--node_features', action='store_true', default=False, help='Whether to save node level features instead')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--save_path', type=str,
                        help='Path to .pckl file where features will be saved as a Python pickle file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                                '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    args = parser.parse_args()
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None
    update_checkpoint_args(args)

    model = load_checkpoint(args.checkpoint_paths[0], cuda=True) # just use first one
    smiles = []
    with open(args.data_path, 'r') as f:
        f.readline()
        for line in f:
            smiles.append(line.strip().split(',')[0])
    vectors = []

    if args.node_features:
        atom_vectors = []
        for i in range(0, len(smiles), args.batch_size):
            batch_vectors, a_scope = model(smiles[i:i+args.batch_size], output_type='node_final')
            batch_vectors = batch_vectors.cpu().detach().numpy()
            for (a_start, a_size) in a_scope:
                atom_vectors.append(batch_vectors[a_start:a_start+a_size])
        ext = os.path.splitext(args.save_path)[1]
        if ext not in ['pkl', 'pckl']:
            args.save_path = args.save_path + '.pckl'
        with open(args.save_path, 'wb') as f:
            pickle.dump(atom_vectors, f)
    else:
        for i in range(0, len(smiles), args.batch_size):
            batch_vectors = model(smiles[i:i+args.batch_size], output_type='encoder')
            batch_vectors = batch_vectors.cpu().detach().numpy()
            for j in range(len(batch_vectors)):
                vectors.append(batch_vectors[j])   
        save_features(args.save_path, vectors)
