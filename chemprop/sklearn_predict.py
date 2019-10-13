from argparse import Namespace
import csv
import pickle

from tqdm import tqdm

from chemprop.data.utils import get_data, get_task_names
from chemprop.features import get_features_generator
from chemprop.utils import makedirs


def predict_sklearn(args: Namespace):
    print('Loading data')
    data = get_data(path=args.test_path)

    if data.num_tasks() != 1:
        raise ValueError(f'Currently only one task is supported but found {data.num_tasks()}')

    print('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    for datapoint in tqdm(data, total=len(data)):
        datapoint.set_features(morgan_fingerprint(mol=datapoint.smiles, radius=args.radius, num_bits=args.num_bits))

    print('Loading model')
    with open(args.checkpoint_path, 'rb') as f:
        model = pickle.load(f)

    print('Predicting')
    # preds = model.predict_proba(data.features())
    preds = model.decision_function(data.features())
    import pdb; pdb.set_trace()

    if data.num_tasks() == 1:
        preds = [[pred] for pred in preds]

    print('Saving predictions')
    makedirs(args.preds_path, isfile=True)

    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles'] + get_task_names(args.test_path))

        for smiles, pred in zip(data.smiles(), preds):
            writer.writerow([smiles] + pred)
