from agents.main_agent import MainAgent

def main():
    user_query = "Identify all swimming pools in Los Angeles and provide me with a map and summary report."

    agent = MainAgent()

    result = agent.run(user_query)

    print(result)

if __name__ == "__main__":
    main()

