from workflows.dynamic_workflow import DynamicWorkflow

def main():
    
    user_query = "Identify all swimming pools in Los Angeles and provide a map and summary report."

    
    workflow = DynamicWorkflow(query=user_query)

    
    result = workflow.execute_workflow()

    
    print(result)

if __name__ == "__main__":
    main()

