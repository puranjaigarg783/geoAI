from workflows.pool_detection_workflow.py import PoolDetectionWorkflow

def main():
    user_query = "Identify all swimming pools in Los Angeles and provide me with a map and a summary report."

    workflow = PoolDetectionWorkflow(query=user_query)
    map_filepath, report_filepath = workflow.execute_workflow()

    print(f"Map saved to: {map_filepath}")
    print(f"Report saved to: {report_filepath}")

if __name__ == "__main__":
    main()

