import os
import importlib.util
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import multivariate_normal

from pysmcjax.model.numpyro_bind import NumPyroModel
from utils import model_data_mapping


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
    numpyro_model_path = Path(os.getcwd(), "numpyro_processed")
    numpyro_models = list(numpyro_model_path.glob("*.numpyro.py"))
    data_path = Path(os.getcwd(), "..", "data", "data")

    with open("pysmcjax_coverage.tsv", "w") as f:
        f.write("model_name\tstatus_code\terror_message\n")

    for numpyro_model in numpyro_models:
        model_name = numpyro_model.stem.replace(".numpyro", "")

        # Load the model
        try:
            methods = load_methods_from_file(numpyro_model, ["model", "convert_inputs"])
        except ImportError as e:
            print(f"Error loading {model_name}: {e}")
            with open("pysmcjax_coverage.tsv", "a") as f:
                f.write(f"{model_name}\t0:MissingImports\t{e}\n")
            continue

        # Access the methods
        model_method = methods.get("model")
        convert_inputs_method = methods.get("convert_inputs")

        # Load the data (catch file not found)
        model_data_name = model_data_mapping(model_name)

        if model_data_name:
            model_data_path = Path(data_path, f"{model_data_name}.json", f"{model_data_name}.json")
            try:
                with open(model_data_path, "r") as f:
                    data = json.load(f)

                model_data = convert_inputs_method(data)
            except FileNotFoundError as e:
                print(f"Error loading {model_name} data: {e}")
                with open("pysmcjax_coverage.tsv", "a") as f:
                    f.write(f"{model_name}\t1:MissingDataFile\t{e}\n")
                continue
            except KeyError as e:
                print(f"Error converting {model_name} inputs: {e}")
                with open("pysmcjax_coverage.tsv", "a") as f:
                    f.write(f"{model_name}\t1:MissingDataKey\t{e}\n")
                continue
        else:
            model_data = {}

        # Instantiate the model
        try:
            model = NumPyroModel(model_method, model_kwargs=model_data)
        except TypeError as e:
            print(f"Error instantiating {model_name}: {e}")
            with open("pysmcjax_coverage.tsv", "a") as f:
                f.write(f"{model_name}\t2:TypeError\t{e}\n")
            continue
        except NameError as e:
            print(f"Error instantiating {model_name}: {e}")
            with open("pysmcjax_coverage.tsv", "a") as f:
                f.write(f"{model_name}\t2:NameError\t{e}\n")
            continue
        except AssertionError as e:
            print(f"Error instantiating {model_name}: {e}")
            with open("pysmcjax_coverage.tsv", "a") as f:
                f.write(f"{model_name}\t2:AssertionError\t{e}\n")
            continue
        except IndexError as e:
            print(f"Error instantiating {model_name}: {e}")
            with open("pysmcjax_coverage.tsv", "a") as f:
                f.write(f"{model_name}\t2:IndexError\t{e}\n")
            continue

        # Sample a random value with dimensionality of the model
        dist = multivariate_normal(mean=np.zeros(model.dim), cov=np.eye(model.dim))
        x = dist.rvs()

        # Try to evaluate log_prob
        try:
            log_prob = model.logpdf(x)
        except Exception as e:
            print(f"Error evaluating log_prob for {model_name}: {e}")
            with open("pysmcjax_coverage.tsv", "a") as f:
                f.write(f"{model_name}\t3:ProbEval\t{e}\n")
            continue

        # Try to evaluate gradient
        try:
            grad = model.logpdfgrad(x)
        except Exception as e:
            print(f"Error evaluating gradient for {model_name}: {e}")
            with open("pysmcjax_coverage.tsv", "a") as f:
                f.write(f"{model_name}\t4:GradientEval\t{e}\n")
            continue

        with open("pysmcjax_coverage.tsv", "a") as f:
            f.write(f"{model_name}\t5:Success\t\n")

        print(f"Successfully evaluated {model_name}")


def summarise_coverage():
    print("\n\nSummarising coverage...")
    coverage = pd.read_csv("pysmcjax_coverage.tsv", sep="\t")
    coverage_summary = coverage.groupby("status_code").size().reset_index(name="count")

    print(coverage_summary)

    coverage_summary.to_csv("pysmcjax_coverage_summary.csv", index=False)

    plt.figsize = (10, 6)
    plt.bar(coverage_summary["status_code"], coverage_summary["count"])
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.title("Coverage Summary")
    plt.tight_layout()
    plt.savefig("pysmcjax_coverage_summary.png")


def main():
    evaluate_models()
    summarise_coverage()


if __name__ == "__main__":
    main()
