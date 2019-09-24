import numpy as np
import GPy
import forestci as fci
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def confidence_estimator_builder(confidence_method: str):
    return {
        'nn': NNEstimator,
        'gaussian': GaussianProcessEstimator,
        'random_forest': RandomForestEstimator,
        'tanimoto': TanimotoEstimator
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


class DroppingEstimator(ConfidenceEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)

        self.sum_last_hidden = np.zeros(
            (len(self.val_data.smiles()), self.args.last_hidden_size))

        self.sum_last_hidden_test = np.zeros(
            (len(self.test_data.smiles()), self.args.last_hidden_size))

    def process_model(self, model, predict):
        model.eval()
        model.use_last_hidden = False

        last_hidden = predict(
            model=model,
            data=self.val_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden += np.array(last_hidden)

        last_hidden_test = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden_test += np.array(last_hidden_test)

    def compute_hidden_vals(self):
        avg_last_hidden = self.sum_last_hidden / self.args.ensemble_size
        avg_last_hidden_test = self.sum_last_hidden_test / self.args.ensemble_size

        transformed_val = self.scaler.transform(
            np.array(self.val_data.targets()))

        return avg_last_hidden, avg_last_hidden_test, transformed_val


class NNEstimator(ConfidenceEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)

        self.sum_test_confidence = np.zeros(
            (len(test_data.smiles()), args.num_tasks))

    def process_model(self, model, predict):
        test_preds, test_confidence = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            confidence=True
        )

        if len(test_preds) != 0:
            self.sum_test_confidence += np.array(test_confidence)

    def compute_confidence(self, test_predictions):
        return test_predictions, \
               self.sum_test_confidence / self.args.ensemble_size * (-1)


class GaussianProcessEstimator(DroppingEstimator):
    def compute_confidence(self, test_predictions):
        avg_last_hidden, avg_last_hidden_test, transformed_val = self.compute_hidden_vals()
        super().compute_confidence(test_predictions)
        if self.args.dataset_type == "regression":
            # TODO: Scale confidence to reflect scaled predictions.
            predictions = np.ndarray(
                shape=(len(self.test_data.smiles()), self.args.num_tasks))
            confidence = np.ndarray(
                shape=(len(self.test_data.smiles()), self.args.num_tasks))

            for task in range(self.args.num_tasks):
                kernel = GPy.kern.Linear(input_dim=self.args.last_hidden_size)
                gaussian = GPy.models.SparseGPRegression(
                    avg_last_hidden,
                    transformed_val[:, task:task + 1], kernel)
                gaussian.optimize()

                avg_test_preds, avg_test_var = gaussian.predict(
                    avg_last_hidden_test)

                # Scale Data
                domain = np.max(avg_test_var) - np.min(avg_test_var)
                # Shift.
                avg_test_var = avg_test_var - np.min(avg_test_var)
                # Scale domain to 1.
                avg_test_var = avg_test_var / domain
                # Apply log scale and flip.
                avg_test_var = np.maximum(
                    0, -np.log(avg_test_var + np.exp(-10)))

                predictions[:, task:task + 1] = avg_test_preds
                confidence[:, task:task + 1] = avg_test_var

            predictions = self.scaler.inverse_transform(predictions)
            return predictions, confidence


class RandomForestEstimator(DroppingEstimator):
    def compute_confidence(self, test_predictions):
        avg_last_hidden, avg_last_hidden_test, transformed_val = self.compute_hidden_vals()
        # TODO: Scale confidence to reflect scaled predictions.
        predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        confidence = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        n_trees = 100
        for task in range(self.args.num_tasks):
            forest = RandomForestRegressor(n_estimators=n_trees)
            forest.fit(avg_last_hidden, transformed_val[:, task])

            avg_test_preds = forest.predict(avg_last_hidden_test)
            predictions[:, task] = avg_test_preds

            avg_test_var = fci.random_forest_error(
                forest, avg_last_hidden, avg_last_hidden_test) * (-1)
            confidence[:, task] = avg_test_var

        predictions = self.scaler.inverse_transform(predictions)
        return predictions, confidence


class EnsembleEstimator(ConfidenceEstimator):
    def __init__(self, train_data, val_data, test_data, scaler, args):
        super().__init__(train_data, val_data, test_data, scaler, args)
        self.all_test_preds = None

    def process_model(self, model, predict):
        test_preds = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
        )

        reshaped_test_preds = test_preds.reshape((len(self.test_data.smiles()), self.args.num_tasks, 1))
        if self.all_test_preds:
            self.all_test_preds = np.concatenate(self.all_test_preds, reshaped_test_preds, axis=2)
        else:
            self.all_test_preds = reshaped_test_preds

    def compute_confidence(self, test_predictions):
        return test_predictions, np.var(self.all_test_preds, axis=2) * -1


class TanimotoEstimator(ConfidenceEstimator):
    def compute_confidence(self, test_predictions):
        train_smiles = self.train_data.smiles()
        test_smiles = self.test_data.smiles()
        confidence = np.ndarray(
            shape=(len(test_smiles), self.args.num_tasks))

        train_smiles_sfp = [morgan_fingerprint(s) for s in train_smiles]
        for i in range(len(test_smiles)):
            confidence[i, :] = np.ones((self.args.num_tasks)) * tanimoto(
                test_smiles[i], train_smiles_sfp, lambda x: max(x))

        return test_predictions, confidence


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
            mol, radius, nBits=num_bits)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=num_bits)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)

    return fp


def tanimoto(smile, train_smiles_sfp, operation):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
    fp = morgan_fingerprint(smiles)
    morgan_sim = []
    for sfp in train_smiles_sfp:
        tsim = np.dot(fp, sfp) / (fp.sum() +
                                  sfp.sum() - np.dot(fp, sfp))
        morgan_sim.append(tsim)
    return operation(morgan_sim)
