from argparse import Namespace
from scipy import stats
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
from scipy.optimize import minimize


class EvaluationMethod:
    def __init__(self):
        self.name = None

    def evaluate(self, data):
        pass

    def visualize(self, task, data):
        evaluation = self.evaluate(data)

        sns.set()

        self._visualize(task, evaluation)


class Cutoffs(EvaluationMethod):
    def __init__(self):
        self.name = "cutoff"

    def evaluate(self, data):
        cutoff = []
        rmse = []
        ideal_rmse = []

        square_error = [set_["error"]**2 for set_ in data["sets_by_confidence"]]
        ideal_square_error = [set_["error"]**2 for set_ in data["sets_by_error"]]

        total_square_error = np.sum(square_error)
        ideal_total_square_error = np.sum(ideal_square_error)

        for i in range(len(square_error)):
            cutoff.append(data["sets_by_confidence"][i]["confidence"])

            rmse.append(np.sqrt(total_square_error/len(square_error[i:])))
            total_square_error -= square_error[i]

            ideal_rmse.append(np.sqrt(ideal_total_square_error / len(square_error[i:])))
            ideal_total_square_error -= ideal_square_error[i]

        return {"cutoff": cutoff, "rmse": rmse, "ideal_rmse": ideal_rmse}

    def _visualize(self, task, evaluation):
        percentiles = np.linspace(0, 100, len(evaluation["rmse"]))

        plt.plot(percentiles, evaluation["rmse"])
        plt.plot(percentiles, evaluation["ideal_rmse"])

        plt.xlabel('Percent of Data Discarded')
        plt.ylabel('RMSE')
        plt.legend(['Confidence Discard', 'Ideal Discard'])
        plt.title(task)

        plt.show()


class Scatter(EvaluationMethod):
    def __init__(self):
        self.name = "scatter"
        self.x_axis_label = 'Confidence'
        self.y_axis_label = 'Error'

    def evaluate(self, data):
        confidence = [self._x_filter(set_["confidence"])
                      for set_ in data["sets_by_confidence"]]
        error = [self._y_filter(set_["error"])
                 for set_ in data["sets_by_confidence"]]

        slope, intercept, _, _, _ = stats.linregress(confidence, error)

        return {"confidence": confidence,
                "error": error,
                "best_fit_y": slope * np.array(confidence) + intercept}

    def _x_filter(self, x):
        return x

    def _y_filter(self, y):
        return y

    def _visualize(self, task, evaluation):
        plt.scatter(evaluation['confidence'], evaluation['error'], s=0.3)
        plt.plot(evaluation['confidence'], evaluation['best_fit_y'])

        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        plt.title(task)

        plt.show()


class AbsScatter(Scatter):
    def __init__(self):
        self.name = "abs_scatter"
        self.x_axis_label = 'Confidence'
        self.y_axis_label = 'Absolute Value of Error'

    def _y_filter(self, y):
        return np.abs(y)


class LogScatter(Scatter):
    def __init__(self):
        self.name = "log_scatter"
        self.x_axis_label = 'Log Confidence'
        self.y_axis_label = 'Log Absolute Value of Error'

    def _x_filter(self, x):
        return np.log(x)

    def _y_filter(self, y):
        return np.log(np.abs(y))


class Spearman(EvaluationMethod):
    def __init__(self):
        self.name = "spearman"

    def evaluate(self, data):
        confidence = [set_["confidence"]
                      for set_ in data["sets_by_confidence"]]
        error = [set_["error"]
                 for set_ in data["sets_by_confidence"]]

        rho, p = stats.spearmanr(confidence, np.abs(error))

        return {"rho": rho, "p": p}

    def _visualize(self, task, evaluation):
        print(task, "-", "Spearman Rho:", evaluation["rho"])
        print(task, "-", "Spearman p-value:", evaluation["p"])


class LogLikelihood(EvaluationMethod):
    def __init__(self):
        self.name = "log_likelihood"

    def evaluate(self, data):
        log_likelihood = 0

        for set_ in data["sets_by_confidence"]:
            # Encourage small standard deviations.
            log_likelihood -= np.log(2 * np.pi * max(0.01, set_["confidence"]**2)) / 2

            # Penalize for large error.
            log_likelihood -= set_["error"]**2/(2 * max(0.01, set_["confidence"]**2))

        return {"log_likelihood": log_likelihood}

    def _visualize(self, task, evaluation):
        print(task, "-", "Sum of Log Likelihoods:", evaluation["log_likelihood"])


class Boxplot(EvaluationMethod):
    def __init__(self):
        self.name = "boxplot"

    def evaluate(self, data):
        errors_by_confidence = [[] for _ in range(10)]

        min_confidence = data["sets_by_confidence"][-1]["confidence"]
        max_confidence = data["sets_by_confidence"][0]["confidence"]
        confidence_range = max_confidence - min_confidence

        for pair in data["sets_by_confidence"]:
            errors_by_confidence[min(
                int((pair["confidence"] - min_confidence)//(confidence_range / 10)),
                9)].append(pair["error"])

        return {"errors_by_confidence": errors_by_confidence,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "data": data}

    def _visualize(self, task, evaluation):
        errors_by_confidence = evaluation["errors_by_confidence"]
        x_vals = list(np.linspace(evaluation["min_confidence"],
                                  evaluation["max_confidence"],
                                  num=len(errors_by_confidence),
                                  endpoint=False) + (evaluation["max_confidence"] - evaluation["min_confidence"])/(len(errors_by_confidence) * 2))
        plt.boxplot(errors_by_confidence, positions=x_vals, widths=(0.02))

        names = (
            f'{len(errors_by_confidence[i])} points' for i in range(10))
        plt.xticks(x_vals, names)
        plt.xlim((evaluation["min_confidence"], evaluation["max_confidence"]))
        Scatter().visualize(task, evaluation["data"])


class ConfidenceEvaluator:
    methods = [Cutoffs(), AbsScatter(), LogScatter(), Spearman(), LogLikelihood(), Boxplot()]

    @staticmethod
    def save(predictions, targets, confidence, args):
        f = open(args.save_confidence, 'w+')
        log = {}

        # Loop through all subtasks.    
        for task in range(args.num_tasks):
            mask = targets[:, task] != None

            task_predictions = np.extract(mask, predictions[:, task])
            task_targets = np.extract(mask, targets[:, task])
            task_confidence = np.extract(mask, confidence[:, task])
            task_error = list(task_predictions - task_targets)

            task_sets = [{"prediction": task_set[0],
                          "target": task_set[1],
                          "confidence": task_set[2],
                          "error": task_set[3]} for task_set in zip(
                                        task_predictions,
                                        task_targets,
                                        task_confidence,
                                        task_error)]

            sets_by_confidence = sorted(task_sets,
                                        key=lambda pair: pair["confidence"],
                                        reverse=True)

            sets_by_error = sorted(task_sets,
                                   key=lambda pair: np.abs(pair["error"]),
                                   reverse=True)

            log[args.task_names[task]] = {
                "sets_by_confidence": sets_by_confidence,
                "sets_by_error": sets_by_error}

        json.dump(log, f)
        f.close()

    @staticmethod
    def visualize(file_path, methods):
        f = open(file_path)
        log = json.load(f)

        for task, data in log.items():
            for method in ConfidenceEvaluator.methods:
                if method.name in methods:
                    method.visualize(task, data)

        f.close()

    @staticmethod
    def evaluate(file_path, methods):
        f = open(file_path)
        log = json.load(f)

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
    def calibrate(file_path):
        def objective_function(beta, confidence, errors):
            pred_vars = np.clip(np.abs(beta[0]) + confidence**2 * np.abs(beta[1]), 0.001, None)
            costs = np.log(pred_vars) / 2 + errors**2 / (2 * pred_vars)

            return(np.sum(costs))
        
        def calibrate_sets(sets, sigmas):
            calibrated_sets = []
            for set_ in sets:
                calibrated_set = set_.copy()
                calibrated_set['confidence'] = sigmas[0] + set_['confidence'] * sigmas[1]
                calibrated_sets.append(calibrated_set)
            return calibrated_sets

        f = open(file_path)
        log = json.load(f)

        cleaned_log = {}
        scaled_log = {}
        for task, data in log.items():
            sampled_data = random.sample(data['sets_by_error'], 30)

            confidence = np.array([set_['confidence'] for set_ in sampled_data])
            errors = np.array([set_['error'] for set_ in sampled_data])

            beta_init = np.array([0, 1])
            result = minimize(objective_function, beta_init, args=(confidence, errors),
                            method='BFGS', options={'maxiter': 500})
            
            # Remove sampled data from test set.
            cleaned_data = {}
            cleaned_data['sets_by_error'] = [set_ for set_ in data['sets_by_error'] if set_ not in sampled_data]
            cleaned_data['sets_by_confidence'] = [set_ for set_ in data['sets_by_confidence'] if set_ not in sampled_data]
            cleaned_log[task] = cleaned_data

            scaled_data = {}
            scaled_data['sets_by_error'] = calibrate_sets(cleaned_data['sets_by_error'], np.abs(result.x))
            scaled_data['sets_by_confidence'] = calibrate_sets(cleaned_data['sets_by_confidence'], np.abs(result.x))
            scaled_log[task] = scaled_data
        
        f.close()

        return cleaned_log, scaled_log

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
#             f'----Cutoffs Generated Using "{args.confidence}" Estimation Process----')
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
#         plt.xlabel("Confidence")
#         plt.ylabel("Percent of Test Points")
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
#         plt.xlabel("Confidence")
#         plt.ylabel("Absolute Value of Error")
#         plt.legend()
#         plt.show()
