import pandas as pd
import argparse
import json
import numpy as np
import os


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", action="store",
                        help="Input json frmo the train run")
    args = parser.parse_args()

    data = json.load(open(args.input_json, "r"))
    # Sort DECREASING by confidence, assume higher confidence is worse?
    test_entries = data['test']
    dataset = list(test_entries.keys())[0]
    data_entries = test_entries[dataset]['sets_by_confidence'][::-1]
    df = pd.DataFrame(data_entries)
    total_entries = len(df)
    # percent_cutoffs = [1, 0.5, 0.25, 0.1, 0.05]
    percent_cutoffs = list(np.logspace(np.log10(0.01), 0, 8))[::-1]
    N = len(percent_cutoffs)

    output = {}
    for cutoff in percent_cutoffs:
        num_items = int(cutoff * total_entries)
        targets = df[:num_items]['target']
        predictions = df[:num_items]['prediction']
        rmse = np.sqrt(np.square(targets - predictions).mean())
        unc = df[:num_items]['confidence'].mean()
        output[cutoff] = rmse

    # Print results to file
    print(output, file=open(os.path.join(os.path.dirname(args.input_json), "cutoffs.txt"), "w"))

    # Scatterplot for curiosity
    import matplotlib.pyplot as plt
    plt.scatter(df["confidence"], np.abs(df["error"]), c=range(len(df["confidence"])))
    plt.xlabel("Uncertainty")
    plt.ylabel("RMSE")
    plt.title(args.input_json)
    plt.show()

    plt.bar(range(N), [output[cutoff] for cutoff in percent_cutoffs])
    plt.xticks(range(N), np.round(percent_cutoffs,2))
    plt.xlabel("Percent cutoff")
    plt.ylabel("RMSE")
    plt.title(args.input_json)
    plt.show()
