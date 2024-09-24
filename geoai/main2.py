from workflows.dynamic_workflow import DynamicWorkflow

def main():
    user_query = "Identify all swimming pools in Los Angeles, generate a map showing their locations, and provide a summary report."

    workflow = DynamicWorkflow(query=user_query)
    result = workflow.execute_workflow()

    print("Generated Code:\n", result['code'])
    print("\nExecution Output:\n", result['execution_output'])
    print("\nSummary:\n", result['summary'])

if __name__ == "__main__":
    main()

