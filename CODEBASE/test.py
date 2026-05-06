import subprocess
import sys
from pathlib import Path

def run_all_benchmarks(root_directory, mode="CPU1", size=3):
    """
    Recursively finds and executes all .py files in a directory.
    
    Args:
        root_directory (str): The starting path to search for scripts.
        mode (str): The --mode argument to pass to each script.
        size (int): The --size argument to pass to each script.
    """
    root_path = Path(root_directory).resolve()
    
    if not root_path.is_dir():
        print(f"[ERROR] {root_directory} is not a valid directory.")
        return

    # Find all .py files recursively, excluding this runner script itself
    python_files = [
        p for p in root_path.rglob("*.py") 
        if p.name != Path(__file__).name
    ]

    print(f"[INFO] Found {len(python_files)} scripts to execute.\n")

    for script_path in python_files:
        print(f"{'='*80}")
        print(f"EXECUTING: {script_path}")
        print(f"{'='*80}")

        # Construct the command
        # Equivalent to: python path/to/script.py --mode CPU1 --size 3
        command = [
            sys.executable,  # Uses the current python interpreter
            str(script_path),
            "--mode", mode,
            "--size", str(size)
        ]

        try:
            # run() waits for the process to complete
            # text=True and capture_output=False allows you to see the 
            # real-time print statements from the scripts.
            result = subprocess.run(command, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Script failed: {script_path}")
            print(f"Return Code: {e.returncode}")
            # The error details will already be in the terminal output
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred while running {script_path.name}: {e}")
        
        print(f"\n[INFO] Finished attempt for {script_path.name}\n")

if __name__ == "__main__":
    target_dir = "/home/mallik-lab-seed-nx16/anik-lab/fingerprinting/model_hub/CODEBASE" 
    run_all_benchmarks(target_dir, mode="CPU1", size=3)