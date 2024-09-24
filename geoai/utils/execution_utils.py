import subprocess
import tempfile
import os
import sys
import traceback
from utils.config import EXECUTION_TIMEOUT

def execute_code(code_str):
    """
    Executes the generated code securely and returns the output.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            code_file = os.path.join(tmpdirname, 'generated_code.py')
            with open(code_file, 'w') as f:
                f.write(code_str)

            # Execute the code with a timeout
            result = subprocess.run(
                [sys.executable, code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=EXECUTION_TIMEOUT,
                check=True,
                text=True
            )

            return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out."
    except subprocess.CalledProcessError as e:
        return f"Execution error:\n{e.stderr}"
    except Exception as e:
        return f"An error occurred during code execution:\n{traceback.format_exc()}"

