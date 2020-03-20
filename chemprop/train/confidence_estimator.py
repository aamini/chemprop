import random
import numpy as np
import GPy
import heapq
from argparse import Namespace
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.optimize import minimize
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from chemprop.utils import negative_log_likelihood


def confidence_estimator_builder(confidence_method: str):
    return {
        'nn': NNEstimator,
        'gaussian': GaussianProcessEstimator,
        'random_forest': RandomForestEstimator,
        'tanimoto': TanimotoEstimator,
        'ensemble': EnsembleEstimator,
        'latent_space': LatentSpaceEstimator,
        'bootstrap': BootstrapEstimator,
        'snapshot': SnapshotEstimator,
        'dropout': DropoutEstimator,
        'fp_random_forest': FPRandomForestEstimator,
        'fp_gaussian': FPGaussianProcessEstimator
    }[confidence_method]


class ConfidenceEstimator:
    def __init__(self, train_data, val_data, test_data, scaler, args):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.scaler = scaler
        self.args = args

    def process_model(self, model, predict):
        pass

    def compute_confidence(self, test_predictions):
        pass

    def _scale_confidence(self, confidence):
        return self.scaler.stds * confidence


class DroppingEstimator(ConfidenceEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)

        self.sum_last_hidden_train = np.zeros(
            (len(self.train_data.smiles()), self.args.last_hidden_size))

        self.sum_last_hidden_val = np.zeros(
            (len(self.val_data.smiles()), self.args.last_hidden_size))

        self.sum_last_hidden_test = np.zeros(
            (len(self.test_data.smiles()), self.args.last_hidden_size))

    def process_model(self, model, predict):
        model.eval()
        model.use_last_hidden = False

        last_hidden_train = predict(
            model=model,
            data=self.train_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden_train += np.array(last_hidden_train)

        last_hidden_val = predict(
            model=model,
            data=self.val_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden_val += np.array(last_hidden_val)

        last_hidden_test = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden_test += np.array(last_hidden_test)

    def _compute_hidden_vals(self):
        avg_last_hidden_train = self.sum_last_hidden_train / self.args.ensemble_size
        avg_last_hidden_val = self.sum_last_hidden_val / self.args.ensemble_size
        avg_last_hidden_test = self.sum_last_hidden_test / self.args.ensemble_size

        return avg_last_hidden_train, avg_last_hidden_val, avg_last_hidden_test


class NNEstimator(ConfidenceEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)

        self.sum_val_confidence = np.zeros(
            (len(val_data.smiles()), args.num_tasks))

        self.sum_test_confidence = np.zeros(
            (len(test_data.smiles()), args.num_tasks))

    def process_model(self, model, predict):
        val_preds, val_confidence = predict(
            model=model,
            data=self.val_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            confidence=True
        )

        if len(val_preds) != 0:
            self.sum_val_confidence += np.array(val_confidence).clip(min=0)

        test_preds, test_confidence = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            confidence=True
        )

        if len(test_preds) != 0:
            self.sum_test_confidence += np.array(test_confidence).clip(min=0)

    def compute_confidence(self, val_predictions, test_predictions):
        return (val_predictions,
                np.sqrt(self.sum_val_confidence / self.args.ensemble_size),
                test_predictions,
                np.sqrt(self.sum_test_confidence / self.args.ensemble_size))


class GaussianProcessEstimator(DroppingEstimator):
    def compute_confidence(self, val_predictions, test_predictions):
        _, avg_last_hidden_val, avg_last_hidden_test = self._compute_hidden_vals()

        val_predictions = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))
        val_confidence = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_confidence = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        transformed_val = self.scaler.transform(
            np.array(self.val_data.targets()))

        for task in range(self.args.num_tasks):
            kernel = GPy.kern.Linear(input_dim=self.args.last_hidden_size)
            gaussian = GPy.models.SparseGPRegression(
                avg_last_hidden_val,
                transformed_val[:, task:task + 1], kernel)
            gaussian.optimize()

            avg_val_preds, avg_val_var = gaussian.predict(
                avg_last_hidden_val)

            val_predictions[:, task:task + 1] = avg_val_preds
            val_confidence[:, task:task + 1] = np.sqrt(avg_val_var)

            avg_test_preds, avg_test_var = gaussian.predict(
                avg_last_hidden_test)

            test_predictions[:, task:task + 1] = avg_test_preds
            test_confidence[:, task:task + 1] = np.sqrt(avg_test_var)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_confidence(val_confidence),
                test_predictions, self._scale_confidence(test_confidence))


class FPGaussianProcessEstimator(ConfidenceEstimator):
    def compute_confidence(self, val_predictions, test_predictions):
        train_smiles = self.train_data.smiles()
        val_smiles = self.val_data.smiles()
        test_smiles = self.test_data.smiles()

        # Train targets are already scaled.
        scaled_train_targets = np.array(self.train_data.targets())

        train_fps = np.array([morgan_fingerprint(s) for s in train_smiles])
        val_fps = np.array([morgan_fingerprint(s) for s in val_smiles])
        test_fps = np.array([morgan_fingerprint(s) for s in test_smiles])

        val_predictions = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))
        val_confidence = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_confidence = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        for task in range(self.args.num_tasks):
            kernel = GPy.kern.Linear(input_dim=train_fps.shape[1])
            gaussian = GPy.models.SparseGPRegression(
                train_fps,
                scaled_train_targets[:, task:task + 1], kernel)
            gaussian.optimize()

            val_preds, val_var = gaussian.predict(
                val_fps)

            val_predictions[:, task:task + 1] = val_preds
            val_confidence[:, task:task + 1] = np.sqrt(val_var)

            test_preds, test_var = gaussian.predict(
                test_fps)

            test_predictions[:, task:task + 1] = test_preds
            test_confidence[:, task:task + 1] = np.sqrt(test_var)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_confidence(val_confidence),
                test_predictions, self._scale_confidence(test_confidence))


class RandomForestEstimator(DroppingEstimator):
    def compute_confidence(self, val_predictions, test_predictions):
        _, avg_last_hidden_val, avg_last_hidden_test = self._compute_hidden_vals()

        val_predictions = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))
        val_confidence = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_confidence = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        transformed_val = self.scaler.transform(
            np.array(self.val_data.targets()))

        n_trees = 128
        for task in range(self.args.num_tasks):
            forest = RandomForestRegressor(n_estimators=n_trees)
            forest.fit(avg_last_hidden_val, transformed_val[:, task])

            avg_val_preds = forest.predict(avg_last_hidden_val)
            val_predictions[:, task] = avg_val_preds

            individual_val_predictions = np.array([estimator.predict(avg_last_hidden_val) for estimator in forest.estimators_])
            val_confidence[:, task] = np.std(individual_val_predictions, axis=0)

            avg_test_preds = forest.predict(avg_last_hidden_test)
            test_predictions[:, task] = avg_test_preds

            individual_test_predictions = np.array([estimator.predict(avg_last_hidden_test) for estimator in forest.estimators_])
            test_confidence[:, task] = np.std(individual_test_predictions, axis=0)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_confidence(val_confidence),
                test_predictions, self._scale_confidence(test_confidence))


class FPRandomForestEstimator(ConfidenceEstimator):
    def compute_confidence(self, val_predictions, test_predictions):
        train_smiles = self.train_data.smiles()
        val_smiles = self.val_data.smiles()
        test_smiles = self.test_data.smiles()

        # Train targets are already scaled.
        scaled_train_targets = np.array(self.train_data.targets())

        train_fps = np.array([morgan_fingerprint(s) for s in train_smiles])
        val_fps = np.array([morgan_fingerprint(s) for s in val_smiles])
        test_fps = np.array([morgan_fingerprint(s) for s in test_smiles])

        val_predictions = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))
        val_confidence = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_confidence = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        n_trees = 128
        for task in range(self.args.num_tasks):
            forest = RandomForestRegressor(n_estimators=n_trees)
            forest.fit(train_fps, scaled_train_targets[:, task])

            avg_val_preds = forest.predict(val_fps)
            val_predictions[:, task] = avg_val_preds

            individual_val_predictions = np.array([estimator.predict(val_fps) for estimator in forest.estimators_])
            val_confidence[:, task] = np.std(individual_val_predictions, axis=0)

            avg_test_preds = forest.predict(test_fps)
            test_predictions[:, task] = avg_test_preds

            individual_test_predictions = np.array([estimator.predict(test_fps) for estimator in forest.estimators_])
            test_confidence[:, task] = np.std(individual_test_predictions, axis=0)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_confidence(val_confidence),
                test_predictions, self._scale_confidence(test_confidence))


class LatentSpaceEstimator(DroppingEstimator):
    def compute_confidence(self, val_predictions, test_predictions):
        avg_last_hidden_train, avg_last_hidden_val, avg_last_hidden_test = self._compute_hidden_vals()

        val_confidence = np.zeros((len(self.val_data.smiles()), self.args.num_tasks))
        test_confidence = np.zeros((len(self.test_data.smiles()), self.args.num_tasks))

        for val_input in range(len(avg_last_hidden_val)):
            distances = np.zeros(len(avg_last_hidden_train))
            for train_input in range(len(avg_last_hidden_train)):
                difference = avg_last_hidden_val[val_input] - avg_last_hidden_train[train_input]
                distances[train_input] = np.sqrt(np.sum(difference * difference))

            val_confidence[val_input, :] = sum(heapq.nsmallest(5, distances))/5

        for test_input in range(len(avg_last_hidden_test)):
            distances = np.zeros(len(avg_last_hidden_train))
            for train_input in range(len(avg_last_hidden_train)):
                difference = avg_last_hidden_test[test_input] - avg_last_hidden_train[train_input]
                distances[train_input] = np.sqrt(np.sum(difference * difference))

            test_confidence[test_input, :] = sum(heapq.nsmallest(5, distances))/5

        return val_predictions, val_confidence, test_predictions, test_confidence


class EnsembleEstimator(ConfidenceEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)
        self.all_val_preds = None
        self.all_test_preds = None

    def process_model(self, model, predict):
        val_preds = predict(
            model=model,
            data=self.val_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
        )

        test_preds = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
        )

        reshaped_val_preds = np.array(val_preds).reshape((len(self.val_data.smiles()), self.args.num_tasks, 1))
        if self.all_val_preds is not None:
            self.all_val_preds = np.concatenate((self.all_val_preds, reshaped_val_preds), axis=2)
        else:
            self.all_val_preds = reshaped_val_preds

        reshaped_test_preds = np.array(test_preds).reshape((len(self.test_data.smiles()), self.args.num_tasks, 1))
        if self.all_test_preds is not None:
            self.all_test_preds = np.concatenate((self.all_test_preds, reshaped_test_preds), axis=2)
        else:
            self.all_test_preds = reshaped_test_preds

    def compute_confidence(self, val_predictions, test_predictions):
        val_confidence = np.sqrt(np.var(self.all_val_preds, axis=2))
        test_confidence = np.sqrt(np.var(self.all_test_preds, axis=2))

        return val_predictions, val_confidence, test_predictions, test_confidence


class BootstrapEstimator(EnsembleEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)   


class SnapshotEstimator(EnsembleEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)   


class DropoutEstimator(EnsembleEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)   


class TanimotoEstimator(ConfidenceEstimator):
    def compute_confidence(self, val_predictions, test_predictions):
        train_smiles = self.train_data.smiles()
        val_smiles = self.val_data.smiles()
        test_smiles = self.test_data.smiles()

        val_confidence = np.ndarray(
            shape=(len(val_smiles), self.args.num_tasks))
        test_confidence = np.ndarray(
            shape=(len(test_smiles), self.args.num_tasks))

        train_smiles_sfp = [morgan_fingerprint(s) for s in train_smiles]

        for i in range(len(val_smiles)):
            val_confidence[i, :] = np.ones((self.args.num_tasks)) * tanimoto(
                val_smiles[i], train_smiles_sfp, lambda x: sum(heapq.nsmallest(8, x))/8)

        for i in range(len(test_smiles)):
            test_confidence[i, :] = np.ones((self.args.num_tasks)) * tanimoto(
                test_smiles[i], train_smiles_sfp, lambda x: sum(heapq.nsmallest(8, x))/8)

        return val_predictions, val_confidence, test_predictions, test_confidence


# Classification methods.
class ConformalEstimator(DroppingEstimator):
    pass


class BoostEstimator(DroppingEstimator):
    pass


def morgan_fingerprint(smiles: str, radius: int = 3, num_bits: int = 2048,
                       use_counts: bool = False) -> np.ndarray:
    """
    Generates a morgan fingerprint for a smiles string.

    :param smiles: A smiles string for a molecule.
    :param radius: The radius of the fingerprint.
    :param num_bits: The number of bits to use in the fingerprint.
    :param use_counts: Whether to use counts or just a bit vector for the fingerprint
    :return: A 1-D numpy array containing the morgan fingerprint.
    """
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if use_counts:
        fp_vect = AllChem.GetHashedMorganFingerprint(
            mol, radius, nBits=num_bits, useChirality=True)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=num_bits, useChirality=True)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)

    return fp


def tanimoto(smile, train_smiles_sfp, operation):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
    fp = morgan_fingerprint(smiles)
    tanimoto_distance = []

    for sfp in train_smiles_sfp:
        tsim = np.dot(fp, sfp) / (fp.sum() +
                                  sfp.sum() - np.dot(fp, sfp))
        tanimoto_distance.append(-np.log2(max(0.0001, tsim)))

    return operation(tanimoto_distance)
