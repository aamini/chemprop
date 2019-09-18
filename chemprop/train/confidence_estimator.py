import numpy as np

def confidence_estimator_builder(confidence_method : str) -> ConfidenceEstimator:
    return {
        'nn': NNEstimator,
        'gaussian': GaussianProcessEstimator,
        'random_forest': RandomForestEstimator,
        'conformal': ConformalEstimator,
        'boost': BoostEstimator
    }[confidence_method]


class ConfidenceEstimator:
    def __init__(self, dataset_type : str, val_data, test_data, scaler, args):
        self.dataset_type = dataset_type
        self.val_data = val_data
        self.test_data = test_data
        self.scaler = scaler
        self.args = args
    
    def process_model(self, model, predict, batch_size):
        pass

    def compute_confidence(self):
        pass


class DroppingEstimator(ConfidenceEstimator):
    def __init__(self, dataset_type : str, val_data, test_data, scaler, args):
        super(self, dataset_type, val_data, test_data, args)

        self.sum_last_hidden = np.zeros(
            (len(self.val_data.smiles()), self.args.last_hidden_size))
        
        self.sum_last_hidden_test = np.zeros(
            (len(self.test_data.smiles()), self.args.last_hidden_size))

    def process_model(self, model, predict, batch_size):
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

    def compute_confidence(self):
        avg_last_hidden = self.sum_last_hidden / self.args.ensemble_size
        avg_last_hidden_test = self.sum_last_hidden_test / self.args.ensemble_size

        if self.args.dataset_type == 'regression':
            transformed_val = self.scaler.transform(np.array(self.val_data.targets()))              
    
class NNEstimator(ConfidenceEstimator):
    def __init__(self, dataset_type : str, val_data, test_data, scaler, args):
        super(self, dataset_type, val_data, test_data, scaler, args)

        self.sum_test_confidence = np.zeros((len(test_data.smiles()), args.num_tasks))
    
    def process_model(self, model, predict, batch_size):
        test_preds, test_confidence = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            confidence=True
        )

        if len(test_preds) != 0:
            self.sum_test_confidence += np.array(test_confidence)


class GaussianProcessEstimator(DroppingEstimator):
    pass


class RandomForestEstimator(DroppingEstimator):
    pass


class ConformalEstimator(DroppingEstimator):
    pass


class BoostEstimator(DroppingEstimator):
    pass