from argparse import Namespace
import csv
import pickle

from tqdm import tqdm

from chemprop.data.utils import get_data, get_task_names
from chemprop.features import get_features_generator
from chemprop.utils import makedirs


def predict_sklearn(args: Namespace):
    assert args.checkpoint_paths is not None and len(args.checkpoint_paths) == 1

    print('Loading data')
    data = get_data(path=args.test_path)

    print('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    for datapoint in tqdm(data, total=len(data)):
        datapoint.set_features(morgan_fingerprint(mol=datapoint.smiles, radius=args.radius, num_bits=args.num_bits))

    print('Loading model')
    model = pickle.load(args.checkpoint_paths[0])

    print('Predicting')
    preds = model.predict(data.features())

    if data.num_tasks() == 1:
        preds = [[pred] for pred in preds]

    print('Saving predictions')
    makedirs(args.preds_path, isfile=True)

    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles'] + get_task_names(args.test_path))

        for smiles, pred in zip(data.smiles(), preds):
            writer.writerow([smiles] + pred)
