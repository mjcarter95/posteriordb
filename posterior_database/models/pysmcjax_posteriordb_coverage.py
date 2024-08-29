import os
import importlib.util
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pathlib import Path

from pysmcjax.model.numpyro_bind import NumPyroModel


def load_methods_from_file(file_path, method_names):
    """
    Dynamically load specified methods from a given Python file.
    
    :param file_path: Path to the Python file.
    :param method_names: List of method names to load.
    :return: Dictionary of method names and their corresponding functions.
    """
    methods = {}
    
    module_name = file_path.stem  # Get the file name without the extension
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    for method_name in method_names:
        if hasattr(module, method_name):
            methods[method_name] = getattr(module, method_name)
        else:
            print(f"Method {method_name} not found in {file_path}")
    
    return methods


def evaluate_models():
    numpyro_model_path = Path(os.getcwd(), "numpyro")
    numpyro_models = list(numpyro_model_path.glob("*.numpyro.py"))
    data_path = Path(os.getcwd(), "..", "data", "data")

    with open("pysmcjax_coverage.csv", "w") as f:
        f.write("model_name,status_code,error_message\n")

    for numpyro_model in numpyro_models:
        model_name = numpyro_model.stem.replace(".numpyro", "")

        # Load the model
        try:
            methods = load_methods_from_file(numpyro_model, ["model", "convert_inputs"])
        except ImportError as e:
            print(f"Error loading {model_name}: {e}")
            with open("pysmcjax_coverage.csv", "a") as f:
                f.write(f"{model_name},0,{e}\n")
            continue

        # Access the methods
        model_method = methods.get("model")
        convert_inputs_method = methods.get("convert_inputs")

        # Load the data (catch file not found)
        try:
            model_data_path = Path(data_path, f"{model_name}.json", f"{model_name}.json")

            with open(model_data_path, "r") as f:
                data = json.load(f)

            model_data = convert_inputs_method(data)
        except FileNotFoundError as e:
            print(f"Error loading {model_name} data: {e}")
            with open("pysmcjax_coverage.csv", "a") as f:
                f.write(f"{model_name},1,{e}\n")
            continue

        # Instantiate the model
        try:
            model = NumPyroModel(model_method, model_kwargs=model_data)
        except Exception as e:
            print(f"Error instantiating {model_name}: {e}")
            with open("pysmcjax_coverage.csv", "a") as f:
                f.write(f"{model_name},2,{e}\n")
            continue

        # Sample a random value with dimensionality of the model
        dist = multivariate_normal(mean=np.zeros(model.dim), cov=np.eye(model.dim))
        x = dist.rvs()

        # Try to evaluate log_prob
        try:
            log_prob = model.logpdf(x)
        except Exception as e:
            print(f"Error evaluating log_prob for {model_name}: {e}")
            with open("pysmcjax_coverage.csv", "a") as f:
                f.write(f"{model_name},3,{e}\n")
            continue

        # Try to evaluate gradient
        try:
            grad = model.logpdfgrad(x)
        except Exception as e:
            print(f"Error evaluating gradient for {model_name}: {e}")
            with open("pysmcjax_coverage.csv", "a") as f:
                f.write(f"{model_name},4,{e}\n")
            continue

        with open("pysmcjax_coverage.csv", "a") as f:
            f.write(f"{model_name},5,\n")

        print(f"Successfully evaluated {model_name}")


def summarise_coverage():
    print("\n\nSummarising coverage...")
    coverage = pd.read_csv("pysmcjax_coverage.csv")
    coverage_summary = coverage.groupby("status_code").size().reset_index(name="count")

    print(coverage_summary)

    coverage_summary.to_csv("pysmcjax_coverage_summary.csv", index=False)

    plt.bar(coverage_summary["status_code"], coverage_summary["count"])
    plt.xlabel("Status Code")
    plt.ylabel("Count")
    plt.title("Coverage Summary")
    plt.savefig("pysmcjax_coverage_summary.png")


def main():
    evaluate_models()
    summarise_coverage()


if __name__ == "__main__":
    main()
