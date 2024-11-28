import subprocess
import sys
from typing import List, Dict, Union
import argparse

def run_script_subprocess(script_path: str, args: List[str]) -> None:
    """
    Method 1: Using subprocess.run()
    This is the recommended way as it provides better isolation and control
    """
    try:
        # Construct the command
        command = [sys.executable, script_path] + args
        
        # Run the script and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # This will raise CalledProcessError if the script fails
        )
        
        # Print the output
        print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        print(f"Script output: {e.output}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def run_script_with_params(script_path: str, params: Dict[str, Union[str, bool]]) -> None:
    """
    Helper function to convert dictionary of parameters to command line arguments
    """
    args = []
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    
    run_script_subprocess(script_path, args)

def main():
    # Example usage
    script_path = "net.py"
    
    # Define different parameter sets to run
    common_params = {
        "filename": "results_net2.csv",
        "net_type": "Net2",
    }
    parameter_sets = [
        {
            "conv_type": "valid",
        },
        {
            "conv_type": "sconv",
        },
        {
            "conv_type": "fconv",
        },
        {
            "conv_type": "circular",
        },
        {
            "conv_type": "reflect",
        },
        {
            "conv_type": "replicate",
        }
    ]
    
    # Run the script with each parameter set
    for i, params in enumerate(parameter_sets, 1):
        params.update(common_params)
        print(f"\nRunning iteration {i}:")
        run_script_with_params(script_path, params)

if __name__ == "__main__":
    main()