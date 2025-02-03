import os
import json
import pandas as pd

inits = ["gaussian", "he", "uniform", "xavier"]
classes = ["ChestCT", "HeadCT", "macro avg", "weighted avg"]
metrics = ["precision", "recall", "f1-score"]

def collect_metrics(directory):
    data = []

    d = {}
    for c in classes:
         for m in metrics:
            d[f"{c}_{m}"] = []

    d["name"] = []
    d["accuracy"] = []
    d["gradients_expectation"] = []
    d["gradients_variance"] = []
    d["weights_expectation"] = []
    d["weights_variance"] = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                file_data = json.load(file)

                d["name"].append(filename)
                d["accuracy"].append(file_data["accuracy"])
                d["gradients_expectation"].append(file_data["gradients_expectation"])
                d["gradients_variance"].append(file_data["gradients_variance"])
                d["weights_expectation"].append(file_data["weights_expectation"])
                d["weights_variance"].append(file_data["weights_variance"])

                for c in classes:
                    for m in metrics:
                        d[f"{c}_{m}"].append(file_data[c][m])

    # Convert the collected data to a DataFrame
    df = pd.DataFrame(d)
    return df

path = "/home/sergei.pnev/confounders/deep_learning_project/drive/MyDrive/deep_learning_project"
df = collect_metrics(path)

output_path = "metrics_summary.csv"
df.to_csv(output_path, index=False)