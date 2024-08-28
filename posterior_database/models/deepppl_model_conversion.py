import os
import sys
import subprocess

from pathlib import Path

OUTPUT_MODEL = "numpyro"

def main():
    stan_model_dir = Path(os.getcwd(), "stan")
    stan_model_files = list(stan_model_dir.glob("*.stan"))
    output_model_dir = Path(os.getcwd(), OUTPUT_MODEL)

    if not output_model_dir.exists():
        output_model_dir.mkdir()
    
    for stan_model_file in stan_model_files:
        output_model_file = Path(output_model_dir, stan_model_file.name.replace(".stan", f".{OUTPUT_MODEL}.py"))
        print(f"Converting {stan_model_file} to {output_model_file}")
        subprocess.run(["stanc", f"--{OUTPUT_MODEL}", str(stan_model_file), "--o", str(output_model_file)])


if __name__ == "__main__":
    main()
