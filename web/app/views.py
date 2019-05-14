"""Defines a number of routes/views for the flask app."""

from argparse import ArgumentParser
from functools import wraps
import io
import os
import sys
import shutil
import sqlite3
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Any, Callable, Dict, List
import werkzeug
import zipfile

from flask import jsonify, redirect, Response, send_file, send_from_directory, url_for
from flask_restful import Resource, reqparse
from rdkit import Chem
from werkzeug.utils import secure_filename

from app import app, db

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from chemprop.data.utils import get_data, get_header, get_smiles, validate_data
from chemprop.parsing import add_predict_args, add_train_args, modify_predict_args, modify_train_args
from chemprop.train.make_predictions import make_predictions
from chemprop.train.run_training import run_training
from chemprop.utils import create_logger, load_task_names, load_args


def check_not_demo(func: Callable) -> Callable:
    """
    View wrapper, which will redirect request to site
    homepage if app is run in DEMO mode.
    :param func: A view which performs sensitive behavior.
    :return: A view with behavior adjusted based on DEMO flag.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if app.config['DEMO']:
            return redirect(url_for('home'))
        return func(*args, **kwargs)

    return decorated_function


def name_already_exists_message(thing_being_named: str, original_name: str, new_name: str) -> str:
    """
    Creates a message about a path already existing and therefore being renamed.

    :param thing_being_named: The thing being renamed (ex. Data, Checkpoint).
    :param original_name: The original name of the object.
    :param new_name: The new name of the object.
    :return: A string with a message about the changed name.
    """
    return f'{thing_being_named} "{original_name} already exists. ' \
           f'Saving to "{new_name}".'


def row_to_json(row: sqlite3.Row, j: bool = True):
    row_to_dict = {key: row[key] for key in row.keys()}

    if j:
        return jsonify(row_to_dict)
    return row_to_dict


def rows_to_json(rows: List[sqlite3.Row], j: bool = True):
    rows_to_dicts = [row_to_json(row, j = False) for row in rows]

    if j:
        return jsonify(rows_to_dicts)
    return rows_to_dicts


def render_error(status: int, message: str):
    """
    :param status The error code.
    :param message A helpful message to explain the error.
    """
    return {"status": status, "message": message}, status


parser = reqparse.RequestParser()
parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
parser.add_argument('userId', type=int, help='The id of the user.')
parser.add_argument('userName', type=str, help='The name of the user.')
parser.add_argument('datasetName', type=str, help='The name of the dataset.')
parser.add_argument('datasetId', type=int, help='The id of the dataset.')
parser.add_argument('checkpointId', type=str, help='The id of the checkpoint.')
parser.add_argument('checkpointName', type=str, help='The name of the checkpoint.')
parser.add_argument('ensembleSize', type=int, help='Training ensemble size.')
parser.add_argument('epochs', type=int, help='Number of training epochs.')
parser.add_argument('gpuId', type=int, help='Id of gpu used for training.')
parser.add_argument('smiles', type=str, action='append', help='Smiles for prediction.')


class Users(Resource):
    def get(self):
        """
        :return: A list of all users.
        """
        return rows_to_json(db.get_users())
    
    def post(self):
        """
        Creates a new user in the database.
        :return: The newly created user object.
        """
        args = parser.parse_args(strict=True)

        if not args.userName:
            return render_error(400, "Must specify a userName.")

        new_user = db.insert_user(args.userName)
        response = row_to_json(new_user)
        response.status_code = 201
        return response


class User(Resource):
    def get(self, user_id: int):
        """

        :param user_id: The id of the user to be returned.
        :return: A requested user object.
        """
        user = db.get_user(int(user_id))
        if not user:
            return render_error(404, "User with specified userId not found.")

        return row_to_json(user)
    
    def delete(self, user_id):
        deleted = db.delete_user(user_id)

        if not deleted:
            return render_error(404, "User with specified userId not found.")
        return {}, 204


class Datasets(Resource):
    def get(self):
        args = parser.parse_args(strict=True)
        
        return rows_to_json(db.get_datasets(args.userId))
    
    def post(self):
        args = parser.parse_args(strict=True)

        if not args.userId:
            return render_error(400, "Must specify a userId.")

        if not args.datasetName:
            return render_error(400, "Must specify a datasetName.")

        if not args.file:
            return render_error(400, "Must specify a file.")

        warnings, errors = [], []

        dataset = args['file']

        with NamedTemporaryFile() as temp_file:
            dataset.save(temp_file.name)
            dataset_errors = validate_data(temp_file.name)

            if len(dataset_errors) > 0:
                return render_error(422, "Invalid dataset file.")
            else:
                targets = get_data(temp_file.name).targets()
                unique_targets = {target for row in targets for target in row if target is not None}

                if unique_targets <= {0, 1}:
                    dataset_type = 'classification'
                else:
                    dataset_type = 'regression'

                new_dataset = db.insert_dataset(args.datasetName, args.userId, dataset_type, len(targets))

                if not new_dataset:
                    return render_error(400, "User with specified userId not found.")

                dataset_path = os.path.join(app.config['DATA_FOLDER'], f'{new_dataset["id"]}.csv')

                if args.datasetName != new_dataset["datasetName"]:
                    warnings.append(name_already_exists_message('Data', args.datasetName, new_dataset["datasetName"]))

                shutil.copy(temp_file.name, dataset_path)
        
        if len(errors) != 0:
            return render_error(415, errors)

        new_dataset = row_to_json(new_dataset, j = False)
        new_dataset['warnings'] = warnings
        response = jsonify(new_dataset)
        response.status_code = 201
        return response


class Dataset(Resource):
    def get(self, dataset_id):
        """
        Downloads a dataset as a .csv file.

        :param dataset: The id of the dataset to download.
        """
        dataset = db.get_dataset(int(dataset_id))
        if not dataset:
            return render_error(404, "Dataset with specified datasetId not found.")

        return row_to_json(dataset)
    
    def delete(self, dataset_id):
        """
        Deletes a dataset.

        :param dataset_id: The id of the dataset to delete.
        """
        deleted = db.delete_dataset(dataset_id)

        if not deleted:
            return render_error(404, "Dataset with specified datasetId not found.")

        os.remove(os.path.join(app.config['DATA_FOLDER'], f'{dataset_id}.csv'))
        return {}, 204


class DatasetFile(Resource):
    def get(self, dataset_id):
        """
        Downloads a dataset as a .csv file.

        :param dataset_id: The id of the dataset to download.
        """
        dataset = db.get_dataset(dataset_id)

        if not dataset:
            return render_error(404, 'Dataset with specified datasetId not found.')

        dataset_file = send_from_directory(app.config['DATA_FOLDER'],
                                           f'{dataset_id}.csv',
                                           as_attachment=True,
                                           attachment_filename=f'{dataset["datasetName"]}.csv',
                                           cache_timeout=-1)

        return dataset_file


class Checkpoints(Resource):
    def get(self):
        args = parser.parse_args(strict=True)

        return rows_to_json(db.get_checkpoints(args.userId))        

    def post(self):
        args = parser.parse_args(strict=True)

        if not args.userId:
            return render_error(400, "Must specify a userId.")
        
        if not args.checkpointName:
            return render_error(400, "Must specify a checkpointName.")

        if not args.file:
            return render_error(400, "Must specify a file.")

        warnings = []

        # Create temporary file to get checkpoint_args without losing data.
        with NamedTemporaryFile() as temp_file:
            args.file.save(temp_file.name)

            checkpoint_args = load_args(temp_file)

            checkpoint = db.insert_checkpoint(args.checkpointName,
                                  args.userId,
                                  checkpoint_args.dataset_type,
                                  checkpoint_args.epochs,
                                  1,
                                  checkpoint_args.train_data_size)

            if not checkpoint:
                return render_error(400, "User with specified userId not found.")

            model = db.insert_model(checkpoint['id'])

            model_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')

            if checkpoint["checkpointName"] != args.checkpointName:
                warnings.append(name_already_exists_message('Checkpoint', checkpoint["checkpointName"], args.checkpointName))

            shutil.copy(temp_file.name, model_path)

        checkpoint = row_to_json(checkpoint, j = False)
        checkpoint['warnings'] = warnings
        response = jsonify(checkpoint)
        response.status_code = 201
        return response


class Checkpoint(Resource):
    def get(self, checkpoint_id):
        checkpoint = db.get_checkpoint(int(checkpoint_id))
        if not checkpoint:
            return render_error(404, "Checkpoint with specified checkpointId not found.")

        return row_to_json(checkpoint)  

    def delete(self, checkpoint_id):
        deleted = db.delete_checkpoint(checkpoint_id)

        if not deleted:
            return render_error(404, "Checkpoint with specified checkpointId not found.")
        
        return {}, 204


class CheckpointFile(Resource):
    def get(self, checkpoint_id):
        checkpoint = db.get_checkpoint(checkpoint_id)

        if not checkpoint:
            return render_error(404, "Checkpoint with specified checkpointId not found.")
        
        models = db.get_models(checkpoint_id)

        model_data = io.BytesIO()

        with zipfile.ZipFile(model_data, mode='w') as z:
            for model in models:
                model_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')
                z.write(model_path, os.path.basename(model_path))

        model_data.seek(0)

        return send_file(
            model_data,
            mimetype='application/zip',
            as_attachment=True,
            attachment_filename=f'{checkpoint["checkpointName"]}.zip',
            cache_timeout=-1
        )


class Train(Resource):
    def post(self):
        args = parser.parse_args(strict=True)

        if not args.userId:
            return render_error(400, "Must specify a userId.")
        
        if not args.checkpointName:
            return render_error(400, "Must specify a checkpointName.")
        
        if not args.datasetId:
            return render_error(400, "Must specify a datasetId.")
        
        if not args.epochs:
            args.epochs = 30
        
        if not args.ensembleSize:
            args.ensembleSize = 1

        warnings = []

        data_path = os.path.join(app.config['DATA_FOLDER'], f'{args.datasetId}.csv')

        data = db.get_dataset(args.datasetId)

        if not data:
            return render_error(404, "Dataset with specified datasetId not found.")

        # Create and modify args
        arg_parser = ArgumentParser()
        add_train_args(arg_parser)
        checkpoint_args = arg_parser.parse_args([])

        checkpoint_args.data_path = data_path
        checkpoint_args.dataset_type = data['class']
        checkpoint_args.epochs = args.epochs
        checkpoint_args.ensemble_size = args.ensembleSize

        if args.gpuId:
            checkpoint_args.gpu = args.gpuId
        else:
            checkpoint_args.no_cuda = True

        checkpoint = db.insert_checkpoint(args.checkpointName,
                              args.userId,
                              checkpoint_args.dataset_type,
                              checkpoint_args.epochs,
                              checkpoint_args.ensemble_size,
                              data['size'])

        if not checkpoint:
            return render_error(400, "User with specified userId not found.")

        with TemporaryDirectory() as temp_dir:
            args.save_dir = temp_dir
            modify_train_args(checkpoint_args)

            # Run training
            logger = create_logger(name='train', save_dir=checkpoint_args.save_dir, quiet=checkpoint_args.quiet)
            task_scores = run_training(checkpoint_args, logger)

            # Check if name overlap
            if args.checkpointName != checkpoint['checkpointName']:
                warnings.append(name_already_exists_message('Checkpoint', args.checkpointName, checkpoint['checkpointName']))

            # Move models
            for root, _, files in os.walk(checkpoint_args.save_dir):
                for fname in files:
                    if fname.endswith('.pt'):
                        model = db.insert_model(checkpoint['id'])
                        save_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')
                        shutil.move(os.path.join(checkpoint_args.save_dir, root, fname), save_path)

        checkpoint = row_to_json(checkpoint, j = False)
        checkpoint['warnings'] = warnings
        response = jsonify(checkpoint)
        response.status_code = 201
        return response


class Predict(Resource):
    def post(self):
        args = parser.parse_args(strict=True)

        if not args.checkpointId:
            return render_error(400, "Must specify a checkpointId.")   

        if args.smiles:
            smiles = args.smiles
        elif args.file:
            data_name = secure_filename(args.file.filename)
            data_path = os.path.join(app.config['TEMP_FOLDER'], data_name)
            args.file.save(data_path)

            # Check if header is smiles
            possible_smiles = get_header(data_path)[0]
            smiles = [possible_smiles] if Chem.MolFromSmiles(possible_smiles) is not None else []

            # Get remaining smiles
            smiles.extend(get_smiles(data_path))
        else:
            return render_error(400, "Must specify smiles or file.")
        
        models = db.get_models(args.checkpointId)

        if len(models) == 0:
            return render_error(404, "No checkpoint found with specified checkpointId.")

        model_paths = [os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt') for model in models]

        task_names = load_task_names(model_paths[0])
        num_tasks = len(task_names)

        # Create and modify args
        predict_parser = ArgumentParser()
        add_predict_args(predict_parser)
        predict_args = predict_parser.parse_args([])

        preds_path = os.path.join(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'])
        predict_args.test_path = 'None'  # TODO: Remove this hack to avoid assert crashing in modify_predict_args
        predict_args.preds_path = preds_path
        predict_args.checkpoint_paths = model_paths

        if args.gpuId:
            predict_args.gpu = args.gpuId
        else:
            predict_args.no_cuda = True

        modify_predict_args(predict_args)

        # Run predictions
        preds = make_predictions(predict_args, smiles=smiles)

        if all(p is None for p in preds):
            return render_error(422, "All SMILES invalid.")

        invalid_smiles_warning = "Invalid SMILE"
        preds = [pred if pred is not None else [invalid_smiles_warning] * num_tasks for pred in preds]

        return jsonify(preds)
