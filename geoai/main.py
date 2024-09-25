from agents.main_agent import MainAgent
import streamlit as st

def main():
    st.title("Foundation Model Analysis Test")
    user_query = st.text_input("Enter your query here", "Identify all swimming pools in Los Angeles and provide me with a map and summary report.")
    # user_query = "Identify all swimming pools in Los Angeles and provide me with a map and summary report."

    if st.button("Run Query"):
        agent = MainAgent()
        result = agent.run(user_query)
        st.write(result)

if __name__ == "__main__":
    main()

