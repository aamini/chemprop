from argparse import Namespace
from scipy import stats
import json
import numpy as np
import os
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg') # REMOVE THIS LINE IF TRYING TO INTERACTIVELY SHOW PLOTS  (IN ADDTION TO SAVING)
import matplotlib.pyplot as plt

from chemprop.data.utils import get_task_names

class EvaluationMethod:
    def __init__(self):
        self.name = None

    def evaluate(self, data):
        pass

    def visualize(self, task, data, logger=print, path="./", draw=True):
        evaluation = self.evaluate(data)

        sns.set()

        self._visualize(task, evaluation, logger, path, draw)


class Cutoffs(EvaluationMethod):
    def __init__(self):
        self.name = 'cutoff'

    def evaluate(self, data):
        cutoff = []
        rmse = []
        ideal_rmse = []

        square_error = [set_['error']**2 for set_ in data['sets_by_confidence']]
        ideal_square_error = [set_['error']**2 for set_ in data['sets_by_error']]

        total_square_error = np.sum(square_error)
        ideal_total_square_error = np.sum(ideal_square_error)

        for i in range(len(square_error)):
            cutoff.append(data['sets_by_confidence'][i]['confidence'])

            rmse.append(np.sqrt(total_square_error/len(square_error[i:])))
            total_square_error -= square_error[i]

            ideal_rmse.append(np.sqrt(ideal_total_square_error / len(square_error[i:])))
            ideal_total_square_error -= ideal_square_error[i]

        return {'cutoff': cutoff, 'rmse': rmse, 'ideal_rmse': ideal_rmse}

    def _visualize(self, task, evaluation, logger, path, draw):
        percentiles = np.linspace(0, 100, len(evaluation['rmse']))

        plt.plot(percentiles, evaluation['rmse'])
        plt.plot(percentiles, evaluation['ideal_rmse'])

        plt.xlabel('Percent of Data Discarded')
        plt.ylabel('RMSE')
        plt.legend(['Confidence Discard', 'Ideal Discard'])
        plt.title(task)

        plt.savefig(os.path.join(path, self.name+".pdf"))
        if draw: plt.show()
        plt.close()

        print(evaluation, file=open(os.path.join(path, self.name+".txt"),"w"))


class Scatter(EvaluationMethod):
    def __init__(self):
        self.name = 'scatter'
        self.x_axis_label = 'Confidence'
        self.y_axis_label = 'Error'

    def evaluate(self, data):
        confidence = [self._x_filter(set_['confidence'])
                      for set_ in data['sets_by_confidence']]
        error = [self._y_filter(set_['error'])
                 for set_ in data['sets_by_confidence']]

        slope, intercept, _, _, _ = stats.linregress(confidence, error)

        return {'confidence': confidence,
                'error': error,
                'best_fit_y': slope * np.array(confidence) + intercept}

    def _x_filter(self, x):
        return x

    def _y_filter(self, y):
        return y

    def _visualize(self, task, evaluation, logger, path, draw):
        plt.scatter(evaluation['confidence'], evaluation['error'], s=0.3)
        plt.plot(evaluation['confidence'], evaluation['best_fit_y'])

        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        plt.title(task)

        plt.savefig(os.path.join(path, self.name+".pdf"))
        if draw: plt.show()
        plt.close()

        print(evaluation, file=open(os.path.join(path, self.name+".txt"),"w"))


class AbsScatter(Scatter):
    def __init__(self):
        self.name = 'abs_scatter'
        self.x_axis_label = 'Confidence'
        self.y_axis_label = 'Absolute Value of Error'

    def _y_filter(self, y):
        return np.abs(y)


class LogScatter(Scatter):
    def __init__(self):
        self.name = 'log_scatter'
        self.x_axis_label = 'Log Confidence'
        self.y_axis_label = 'Log Absolute Value of Error'

    def _x_filter(self, x):
        return np.log(x)

    def _y_filter(self, y):
        return np.log(np.abs(y))


class Spearman(EvaluationMethod):
    def __init__(self):
        self.name = 'spearman'

    def evaluate(self, data):
        confidence = [set_['confidence']
                      for set_ in data['sets_by_confidence']]
        error = [set_['error']
                 for set_ in data['sets_by_confidence']]

        rho, p = stats.spearmanr(confidence, np.abs(error))

        return {'rho': rho, 'p': p}

    def _visualize(self, task, evaluation, logger, path, draw):
        print(evaluation, file=open(os.path.join(path, self.name+".txt"),"w"))
        logger("{} - Spearman Rho: {}".format(task, evaluation['rho']))
        logger("{} - Spearman p-value: {}".format(task, evaluation['p']))


class LogLikelihood(EvaluationMethod):
    def __init__(self):
        self.name = 'log_likelihood'

    def evaluate(self, data):
        log_likelihood = 0
        optimal_log_likelihood = 0
        for set_ in data['sets_by_confidence']:
            # Encourage small standard deviations.
            log_likelihood -= np.log(2 * np.pi * max(0.00001, set_['confidence']**2)) / 2
            optimal_log_likelihood -= np.log(2 * np.pi * set_['error']**2) / 2

            # Penalize for large error.
            log_likelihood -= set_['error']**2/(2 * max(0.00001, set_['confidence']**2))
            optimal_log_likelihood -= 1 / 2 # set_['error']**2/(2 * set_['error']**2)

        return {'log_likelihood': log_likelihood,
                'optimal_log_likelihood': optimal_log_likelihood,
                'average_log_likelihood': log_likelihood / len(data['sets_by_confidence']),
                'average_optimal_log_likelihood': optimal_log_likelihood / len(data['sets_by_confidence'])}

    def _visualize(self, task, evaluation, logger, path, draw):
        print(evaluation, file=open(os.path.join(path, self.name+".txt"),"w"))
        logger("{} - Sum of Log Likelihoods: {}".format(task, evaluation['log_likelihood']))


class CalibrationAUC(EvaluationMethod):
    def __init__(self):
        self.name = 'calibration_auc'

    def evaluate(self, data):
        standard_devs = [np.abs(set_['error'])/set_['confidence'] for set_ in data['sets_by_confidence']]
        probabilities = [2 * (stats.norm.cdf(standard_dev) - 0.5) for standard_dev in standard_devs]
        sorted_probabilities = sorted(probabilities)

        fraction_under_thresholds = []
        threshold = 0

        for i in range(len(sorted_probabilities)):
            while sorted_probabilities[i] > threshold:
                fraction_under_thresholds.append(i/len(sorted_probabilities))
                threshold += 0.001

        # Condition used 1.0001 to catch floating point errors.
        while threshold < 1.0001:
            fraction_under_thresholds.append(1)
            threshold += 0.001

        thresholds = np.linspace(0, 1, num=1001)
        miscalibration = [np.abs(fraction_under_thresholds[i] - thresholds[i]) for i in range(len(thresholds))]
        miscalibration_area = 0
        for i in range(1, 1001):
            miscalibration_area += np.average([miscalibration[i-1], miscalibration[i]]) * 0.001


        return {'fraction_under_thresholds': fraction_under_thresholds,
                'thresholds': thresholds,
                'miscalibration_area': miscalibration_area}

    def _visualize(self, task, evaluation, logger, path, draw):
        # Ideal curve.
        plt.plot(evaluation['thresholds'], evaluation['thresholds'])

        # True curve.
        plt.plot(evaluation['thresholds'], evaluation['fraction_under_thresholds'])
        print(task, '-', 'Miscalibration Area', evaluation['miscalibration_area'])

        plt.title(task)

        plt.savefig(os.path.join(path, self.name+".pdf"))
        if draw: plt.show()
        plt.close()

        print(evaluation, file=open(os.path.join(path, self.name+".txt"),"w"))


class Boxplot(EvaluationMethod):
    def __init__(self):
        self.name = 'boxplot'

    def evaluate(self, data):
        errors_by_confidence = [[] for _ in range(10)]

        min_confidence = data['sets_by_confidence'][-1]['confidence']
        max_confidence = data['sets_by_confidence'][0]['confidence']
        confidence_range = max_confidence - min_confidence

        for pair in data['sets_by_confidence']:
            errors_by_confidence[min(
                int((pair['confidence'] - min_confidence)//(confidence_range / 10)),
                9)].append(pair['error'])

        return {'errors_by_confidence': errors_by_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'data': data}

    def _visualize(self, task, evaluation, logger, path, draw):
        errors_by_confidence = evaluation['errors_by_confidence']
        x_vals = list(np.linspace(evaluation['min_confidence'],
                                  evaluation['max_confidence'],
                                  num=len(errors_by_confidence),
                                  endpoint=False) + (evaluation['max_confidence'] - evaluation['min_confidence'])/(len(errors_by_confidence) * 2))
        plt.boxplot(errors_by_confidence, positions=x_vals, widths=(0.02))

        names = (
            f'{len(errors_by_confidence[i])} points' for i in range(10))
        plt.xticks(x_vals, names)
        plt.xlim((evaluation['min_confidence'], evaluation['max_confidence']))
        Scatter().visualize(task, evaluation['data'], logger, path, draw)


class ConfidenceEvaluator:
    methods = [Cutoffs(), AbsScatter(), LogScatter(), Spearman(), LogLikelihood(), Boxplot(), CalibrationAUC()]

    @staticmethod
    def save(val_predictions, val_targets, val_confidence, val_stds, val_smiles,
             test_predictions, test_targets, test_confidence, test_stds, test_smiles,
             val_entropy, test_entropy, args):

        f = open(args.save_confidence, 'w+')
        val_data = ConfidenceEvaluator._log(val_predictions, val_targets,
                                            val_confidence, val_stds, val_smiles,
                                            val_entropy, args)
        test_data = ConfidenceEvaluator._log(test_predictions, test_targets,
                                             test_confidence, test_stds, test_smiles,
                                             test_entropy, args)

        json.dump({'validation': val_data, 'test': test_data}, f)
        f.close()

    @staticmethod
    def _log(predictions, targets, confidence, stds, smiles, entropy, args):
        log = {}

        targets = np.array(targets)

        # Hardcoded
        if args.atomistic:
            task_names = ["U0"]
        else:
            task_names = get_task_names(args.data_path)

        # Loop through all subtasks.
        for task in range(args.num_tasks):
            mask = targets[:, task] != None

            task_predictions = np.extract(mask, predictions[:, task])
            task_targets = np.extract(mask, targets[:, task])
            task_confidence = np.extract(mask, confidence[:, task])
            task_stds = np.extract(mask, stds[:, task])
            task_error = list(task_predictions - task_targets)

            # Extract smiles for this task
            task_smiles = np.extract(mask,smiles)

            task_set_names = ["prediction", "target", "confidence", "stds",
                              "error"]
            task_data = (task_predictions, task_targets,
                         task_confidence, task_stds, task_error, )

            if not args.no_smiles_export:
                # Extract smiles for this task
                task_smiles = np.extract(mask,smiles)
                task_set_names.append("smiles")
                task_data = task_data + (task_smiles,)

            if args.use_entropy:
                task_entropy = np.extract(mask, entropy[:, task])
                task_set_names.append("entropy")
                task_data = task_data + (task_entropy,)

            task_sets = [dict(zip(task_set_names, task_set)) for task_set in zip(*task_data)]

            sets_by_confidence = sorted(task_sets,
                                        key=lambda pair: pair['confidence'],
                                        reverse=True)
            sets_by_error = sorted(task_sets,
                                   key=lambda pair: np.abs(pair['error']),
                                   reverse=True)

            log[task_names[task]] = {
                'sets_by_confidence': sets_by_confidence,
                'sets_by_error': sets_by_error}

            if args.use_entropy:
                sets_by_entropy = sorted(task_sets,
                                       key=lambda pair: np.abs(pair['entropy']),
                                       reverse=True)
                log[task_names[task]]['sets_by_entropy'] = sets_by_entropy

        return log

    @staticmethod
    def visualize(file_path, methods, logger, save_dir, draw):
        f = open(file_path)
        log = json.load(f)['test']

        for task, data in log.items():
            for method in ConfidenceEvaluator.methods:
                if method.name in methods:
                    method.visualize(task, data, logger, save_dir, draw)

        f.close()

    @staticmethod
    def evaluate(file_path, methods):
        f = open(file_path)
        log = json.load(f)['test']

        all_evaluations = {}
        for task, data in log.items():
            task_evaluations = {}
            for method in ConfidenceEvaluator.methods:
                if method.name in methods:
                    task_evaluations[method.name] = method.evaluate(data)
            all_evaluations[task] = task_evaluations

        f.close()

        return all_evaluations

    @staticmethod
    def calibrate(lambdas, beta_init, file_path):
        def objective_function(beta, confidence, errors, lambdas):
            # Construct prediction through lambdas and betas.
            pred_vars = np.zeros(len(confidence))

            for i in range(len(beta)):
                pred_vars += np.abs(beta[i]) * lambdas[i](confidence**2)
            pred_vars = np.clip(pred_vars, 0.001, None)
            costs = np.log(pred_vars) / 2 + errors**2 / (2 * pred_vars)

            return(np.sum(costs))

        def calibrate_sets(sets, sigmas, lambdas):
            calibrated_sets = []
            for set_ in sets:
                calibrated_set = set_.copy()
                calibrated_set['confidence'] = 0

                for i in range(len(sigmas)):
                    calibrated_set['confidence'] += sigmas[i] * lambdas[i](set_['confidence']**2)
                calibrated_sets.append(calibrated_set)
            return calibrated_sets

        f = open(file_path)
        full_log = json.load(f)
        val_log = full_log['validation']
        test_log = full_log['test']

        scaled_val_log = {}
        scaled_test_log = {}

        calibration_coefficients = {}
        for task in val_log:
            # Sample from validation data.
            sampled_data = val_log[task]['sets_by_error']

            # Calibrate based on sampled data.
            confidence = np.array([set_['confidence'] for set_ in sampled_data])
            errors = np.array([set_['error'] for set_ in sampled_data])

            result = minimize(objective_function, beta_init, args=(confidence, errors, lambdas),
                            method='BFGS', options={'maxiter': 500})

            calibration_coefficients[task] = np.abs(result.x)

            scaled_val_data = {}
            scaled_val_data['sets_by_error'] = calibrate_sets(val_log[task]['sets_by_error'], np.abs(result.x), lambdas)
            scaled_val_data['sets_by_confidence'] = calibrate_sets(val_log[task]['sets_by_confidence'], np.abs(result.x), lambdas)
            scaled_val_log[task] = scaled_val_data

            scaled_test_data = {}
            scaled_test_data['sets_by_error'] = calibrate_sets(test_log[task]['sets_by_error'], np.abs(result.x), lambdas)
            scaled_test_data['sets_by_confidence'] = calibrate_sets(test_log[task]['sets_by_confidence'], np.abs(result.x), lambdas)
            scaled_test_log[task] = scaled_test_data

        f.close()

        return {'validation': scaled_val_log, 'test': scaled_test_log}, calibration_coefficients

# OUTDATED VISUALIZATIONS
# def confidence_visualizations(args: Namespace,
#                               predictions: List[Union[float, int]] = [],
#                               targets: List[Union[float, int]] = [],
#                               confidence: List[Union[float, int]] = [],
#                               accuracy_sublog = None):
#     error = list(np.abs(predictions - targets))
#     sets_by_confidence = sorted(
#         list(zip(confidence, error, predictions, targets)), key=lambda pair: pair[0])
#     sets_by_error = sorted(list(zip(confidence, error, predictions, targets)),
#                            key=lambda pair: pair[1])

#     metric_func = get_metric_func(metric=args.metric)

#     if args.c_cutoff_discrete:
#         print(
#             f'----Cutoffs Generated Using '{args.confidence}' Estimation Process----')
#         for i in range(10):
#             c = int((i/10) * len(sets_by_confidence))
#             kept = (len(sets_by_confidence) - c) / len(sets_by_confidence)

#             accuracy = evaluate_predictions(
#                 preds=[[pair[2]] for pair in sets_by_confidence[c:]],
#                 targets=[[pair[3]] for pair in sets_by_confidence[c:]],
#                 num_tasks=1,
#                 metric_func=metric_func,
#                 dataset_type=args.dataset_type
#             )

#             log = f'Cutoff: {sets_by_confidence[c][0]:5.3f} {args.metric}: {accuracy[0]:5.3f} Results Kept: {int(kept*100)}%'
#             print(log)

#             if args.save_confidence:
#                 accuracy_sublog.append({'metric': args.metric,
#                                          'accuracy': accuracy[0],
#                                          'percent_kept': int(kept*100),
#                                          'cutoff': sets_by_confidence[c][0]})

#         print(f'----Cutoffs Generated by Ideal Confidence Estimator----')
#         # Calculate Ideal Cutoffs
#         for i in range(10):
#             c = int((1 - i/10) * len(sets_by_error))
#             kept = c / len(sets_by_error)

#             accuracy = evaluate_predictions(
#                 preds=[[pair[2]] for pair in sets_by_error[:c]],
#                 targets=[[pair[3]] for pair in sets_by_error[:c]],
#                 num_tasks=1,
#                 metric_func=metric_func,
#                 dataset_type=args.dataset_type
#             )

#             print(f'{args.metric}: {accuracy[0]:5.3f}',
#                   f'% Results Kept: {int(kept*100)}%')

#     if args.c_cutoff:
#         cutoff = []
#         rmse = []
#         square_error = [pair[1]*pair[1] for pair in sets_by_confidence]
#         for i in range(len(sets_by_confidence)):
#             cutoff.append(sets_by_confidence[i][0])
#             rmse.append(np.sqrt(np.mean(square_error[i:])))

#         plt.plot(cutoff, rmse)

#     if args.c_bootstrap:
#         # Perform Bootstrapping at 95% confidence.
#         sum_subset = sum([val[0]
#                           for val in sets_by_confidence[:args.bootstrap[0]]])

#         x_confidence = []
#         y_confidence = []

#         for i in range(args.bootstrap[0], len(sets_by_confidence)):
#             x_confidence.append(sum_subset/args.bootstrap[0])

#             ys = [val[1] for val in sets_by_confidence[i-args.bootstrap[0]:i]]
#             y_sum = 0
#             for j in range(args.bootstrap[2]):
#                 y_sum += sorted(np.random.choice(ys,
#                                                  args.bootstrap[1]))[-int(args.bootstrap[1]/20)]

#             y_confidence.append(y_sum / args.bootstrap[2])
#             sum_subset -= sets_by_confidence[i-args.bootstrap[0]][0]
#             sum_subset += sets_by_confidence[i][0]

#         plt.plot(x_confidence, y_confidence)

#     if args.c_cutoff or args.c_bootstrap:
#         plt.show()

#     if args.c_histogram:
#         scale = np.average(error) * 5

#         for i in range(5):
#             errors = []
#             for pair in sets_by_confidence:
#                 if pair[0] < 2 * i or pair[0] > 2 * (i + 1):
#                     continue
#                 errors.append(pair[1])
#             plt.hist(errors, bins=10, range=(0, scale))
#             plt.show()

#     if args.c_joined_histogram:
#         bins = 8
#         scale = np.average(error) * bins / 2

#         errors_by_confidence = [[] for _ in range(10)]
#         for pair in sets_by_confidence:
#             if pair[0] == 10:
#                 continue

#             errors_by_confidence[int(pair[0])].append(pair[1])

#         errors_by_bin = [[] for _ in range(bins)]
#         for interval in errors_by_confidence:
#             error_counts, _ = np.histogram(
#                 np.minimum(interval, scale), bins=bins)
#             for i in range(bins):
#                 if len(interval) == 0:
#                     errors_by_bin[i].append(0)
#                 else:
#                     errors_by_bin[i].append(
#                         error_counts[i] / len(interval) * 100)

#         colors = ['green', 'yellow', 'orange', 'red',
#                   'purple', 'blue', 'brown', 'black']
#         sum_heights = [0 for _ in range(10)]
#         for i in range(bins):
#             label = f'Error of {i * scale / bins} to {(i + 1) * scale / bins}'

#             if i == bins - 1:
#                 label = f'Error {i * scale / bins} +'
#             plt.bar(list(
#                 range(10)), errors_by_bin[i], color=colors[i], bottom=sum_heights, label=label)
#             sum_heights = [sum_heights[j] + errors_by_bin[i][j]
#                            for j in range(10)]

#         names = (
#             f'{i} to {i+1} \n {len(errors_by_confidence[i])} points' for i in range(10))
#         plt.xticks(list(range(10)), names)
#         plt.xlabel('Confidence')
#         plt.ylabel('Percent of Test Points')
#         plt.legend()

#         plt.show()

#     if args.c_joined_boxplot:
#         errors_by_confidence = [[] for _ in range(10)]
#         for pair in sets_by_confidence:
#             if pair[0] == 10:
#                 continue

#             errors_by_confidence[int(pair[0])].append(pair[1])

#         fig, ax = plt.subplots()
#         ax.set_title('Joined Boxplot')
#         ax.boxplot(errors_by_confidence)

#         names = (
#             f'{i} to {i+1} \n {len(errors_by_confidence[i])} points' for i in range(10))
#         plt.xticks(list(range(1, 11)), names)
#         plt.xlabel('Confidence')
#         plt.ylabel('Absolute Value of Error')
#         plt.legend()
#         plt.show()
