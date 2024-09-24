from utils.execution_utils import execute_code

class ExecutionAgent:
    def __init__(self):
        pass

    def execute(self, code_str):
        """
        Executes the provided code string and returns the output.
        """
        output = execute_code(code_str)
        if "Authentication error" in output or "Unauthorized" in output:
            return "Error: Failed to authenticate with Hugging Face. Please check your token."
        return output

