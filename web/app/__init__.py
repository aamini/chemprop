"""Runs the web interface version of chemprop, allowing for training and predicting in a web browser."""
import os
from flask import Flask
from flask_restful import Api

app = Flask(__name__)
app.config.from_object('config')
api = Api(app)

os.makedirs(app.config['CHECKPOINT_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

from app import views

api.add_resource(views.Users, '/users')
api.add_resource(views.User, '/users/<user_id>')
api.add_resource(views.Datasets, '/datasets')
api.add_resource(views.Dataset, '/datasets/<dataset_id>')