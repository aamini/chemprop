"""Defines a number of database helper functions."""

import os
import shutil
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import current_app, Flask, g

from app import app


def init_app(app: Flask):
    app.teardown_appcontext(close_db)


def init_db():
    """
    Initializes the database by running schema.sql.
    This will wipe existing tables and the corresponding files.
    """
    shutil.rmtree(app.config['DATA_FOLDER'])
    os.makedirs(app.config['DATA_FOLDER'])

    shutil.rmtree(app.config['CHECKPOINT_FOLDER'])
    os.makedirs(app.config['CHECKPOINT_FOLDER'])

    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))


def get_db():
    """
    Connects to the database.
    Returns a database object that can be queried.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            'chemprop.sqlite3',
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def query_db(query: str, args: Tuple[Any] = (), one: bool = False) -> Union[Optional[sqlite3.Row], List[sqlite3.Row]]:
    """
    Helper function to allow for easy queries.
    
    :param query: The query to be executed.
    :param args: The arguments to be passed into the query.
    :param one: Whether the query should return all results or just the first. 
    :return The results of the query.
    """
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def close_db(e: Optional[Any] = None):
    """
    Closes the connection to the database. Called after every request.

    :param e: Error object from last call.
    """
    db = g.pop('db', None)

    if db is not None:
        db.close()


# Table Specific Functions
def get_users() -> List[sqlite3.Row]:
    """
    Returns all users.

    :return A list of users.
    """
    return query_db("SELECT * FROM user")


def get_user(user_id) -> sqlite3.Row:
    """
    Returns a specific user.

    :user_id The id of the user to be returned.
    :return A dict representing the desired user.
    """
    return query_db(f'SELECT * FROM user WHERE id = {user_id}', one = True)


def insert_user(user_name: str) -> sqlite3.Row:
    """
    Inserts a new user. If the desired username is already taken,  
    appends integers incrementally until an open name is found.

    :param username: The desired username for the new user.
    :return A dictionary containing the id and name of the new user.
    """
    db = get_db()

    new_user_id = None
    count = 0
    while new_user_id is None:
        temp_name = user_name
        
        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO user (userName) VALUES (?)', [temp_name])
            new_user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            count += 1

    db.commit()
    cur.close()

    return get_user(new_user_id)


def delete_user(user_id: int):
    """
    Removes the user with the specified id from the database,
    associated checkpoints, and the corresponding files.

    :param user_id: The id of the user to be deleted.
    :return Boolean identifying if the selected dataset was deleted.
    """
    exists = get_user(user_id)

    if not exists:
        return False

    ids = query_db(f'SELECT id FROM checkpoint WHERE userId = {user_id}')

    for id_ in ids:
        delete_checkpoint(id_)

    db = get_db()
    cur = db.cursor()
    db.execute(f'DELETE FROM user WHERE id = {user_id}')
    db.commit()
    cur.close()
    return True


def get_checkpoints(user_id: int) -> List[sqlite3.Row]:
    """
    Returns the checkpoints associated with the given user.
    If no user_id is provided, return the checkpoints associated
    with the default user.

    :param user_id: The id of the user whose checkpoints are returned.
    :return A list of checkpoints.
    """
    if not user_id:
        return query_db(f'SELECT * FROM checkpoint')
 
    return query_db(f'SELECT * FROM checkpoint WHERE userId = {user_id}')


def get_checkpoint(checkpoint_id) -> sqlite3.Row:
    """
    Returns a specific checkpoint.

    :user_id The id of the checkpoint to be returned.
    :return A dict representing the desired checkpoint.
    """
    return query_db(f'SELECT * FROM checkpoint WHERE id = {checkpoint_id}', one = True)


def insert_checkpoint(checkpoint_name: str, 
                user_id: str, 
                model_class: str, 
                num_epochs: int,
                ensemble_size: int,
                training_size: int) -> sqlite3.Row:
    """
    Inserts a new checkpoint. If the desired name is already taken,  
    appends integers incrementally until an open name is found.   

    :param checkpoint_name: The desired name for the new checkpoint.
    :param user_id: The user that should be associated with the new checkpoint.
    :param model_class: The class of the new checkpoint.
    :param num_epochs: The number of epochs the new checkpoint will run for.
    :param ensemble_size: The number of models included in the ensemble.
    :param training_size: The number of molecules used for training.
    :return A tuple containing the id and name of the new checkpoint.   
    """
    exists = get_user(user_id)

    if not exists:
        return None

    db = get_db()

    new_checkpoint_id = None
    count = 0
    while new_checkpoint_id is None:
        temp_name = checkpoint_name

        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO checkpoint '
                             '(checkpointName, userId, class, epochs, ensembleSize, trainingSize) '
                             'VALUES (?, ?, ?, ?, ?, ?)',
                             [temp_name, user_id, model_class, num_epochs, ensemble_size, training_size])
            new_checkpoint_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
            count += 1
            continue
    
    db.commit()
    cur.close()

    return get_checkpoint(new_checkpoint_id)


def delete_checkpoint(checkpoint_id: int):
    """
    Removes the checkpoint with the specified id from the database,
    associated model columns, and the corresponding files.

    :return Boolean identifying if the selected checkpoint was deleted.
    """
    exists = get_checkpoint(checkpoint_id)

    if not exists:
        return False

    rows = query_db(f'SELECT * FROM model WHERE checkpointId = {checkpoint_id}')

    for row in rows:
        os.remove(os.path.join(app.config['CHECKPOINT_FOLDER'], f'{row["id"]}.pt'))

    db = get_db()
    cur = db.cursor()
    db.execute(f'DELETE FROM checkpoint WHERE id = {checkpoint_id}')
    db.execute(f'DELETE FROM model WHERE checkpointId = {checkpoint_id}')
    db.commit()
    cur.close()
    return True


def get_models(checkpoint_id: int) -> List[sqlite3.Row]:
    """
    Returns the models associated with the given checkpoint.

    :param checkpoint_id: The id of the checkpoint whose component models are returned.
    :return A list of models.
    """
    if not checkpoint_id:
        return query_db('SELECT * FROM model')

    return query_db(f'SELECT * FROM model WHERE checkpointId = {checkpoint_id}')


def get_model(model_id: int) -> sqlite3.Row:
    """
    Returns a specific model.

    :model_id The id of the model to be returned.
    :return A dict representing the desired model.
    """
    return query_db(f'SELECT * FROM model WHERE id = {model_id}', one = True)


def insert_model(checkpoint_id: int) -> str:
    """
    Inserts a new model.

    :param checkpoint_id: The id of the checkpoint this model should be associated with.
    :return: The id of the new model.
    """
    exists = get_checkpoint(checkpoint_id)
    if not exists:
        return None

    db = get_db()
    cur = db.execute('INSERT INTO model (checkpointId) VALUES (?)', [checkpoint_id])
    new_model_id = cur.lastrowid
    db.commit()
    cur.close()

    return get_model(new_model_id)


def get_datasets(user_id: int) -> List[sqlite3.Row]:
    """
    Returns the datasets associated with the given user.
    If no user_id is provided, return the datasets associated
    with the default user.

    :param user_id: The id of the user whose datasets are returned.
    :return A list of datasets.
    """
    if not user_id:
        return query_db('SELECT * FROM dataset')

    return query_db(f'SELECT * FROM dataset WHERE userId = {user_id}')


def get_dataset(dataset_id: int) -> sqlite3.Row:
    """
    Returns a specific dataset.

    :dataset_id The id of the dataset to be returned.
    :return A dict representing the desired dataset.
    """
    return query_db(f'SELECT * FROM dataset WHERE id = {dataset_id}', one = True)


def insert_dataset(dataset_name: str, 
                   user_id: str, 
                   dataset_class: str,
                   dataset_size: int) -> sqlite3.Row:
    """
    Inserts a new dataset. If the desired name is already taken,  
    appends integers incrementally until an open name is found.   

    :param dataset_name: The desired name for the new dataset.
    :param associated_user: The user to be associated with the new dataset.
    :param dataset_class: The class of the new dataset.
    :param dataset_size: The number of molecules in the dataset.
    :return A tuple containing the id and name of the new dataset.   
    """
    exists = get_user(user_id)
    if not exists:
        return None

    db = get_db()

    new_dataset_id = None
    count = 0
    while new_dataset_id == None:
        temp_name = dataset_name

        if count != 0:
            temp_name += str(count)
        try:
            cur = db.execute('INSERT INTO dataset (datasetName, userId, class, size) VALUES (?, ?, ?, ?)',
                             [temp_name, user_id, dataset_class, dataset_size])
            new_dataset_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
            count += 1
            continue
    
    db.commit()
    cur.close()

    return get_dataset(new_dataset_id)


def delete_dataset(dataset_id: int) -> bool:
    """
    Removes the dataset with the specified id from the database,
    and deletes the corresponding file.

    :param dataset_id: The id of the dataset to be deleted.
    :return Boolean identifying if the selected dataset was deleted.
    """
    exists = get_dataset(dataset_id)

    if not exists:
        return False

    db = get_db()
    cur = db.execute(f'DELETE FROM dataset WHERE id = {dataset_id}')
    db.commit()
    cur.close()
    return True