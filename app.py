import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import logging

from data_source_manager import DataSourceManager
from dynamic_prompt_agent import DynamicPromptAgent

# Load environment variables from .env file
load_dotenv()
serp_api_key = os.getenv("SERP_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(page_title="AI Agent Information Retrieval", layout="wide")
    
    if 'data' not in st.session_state:
        st.session_state.data = None

    st.title("AI Agent Information Retriever")
    st.markdown("Choose your data source:")

    data_source = st.radio("Select Data Source", ["Upload CSV", "Google Sheets"], horizontal=True)

    data_manager = DataSourceManager()

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
    elif data_source == "Google Sheets":
        sheet_url = st.text_input("Enter the Google Sheets URL:")
        if sheet_url:
            st.session_state.data = data_manager.load_google_sheet(sheet_url)

    if st.session_state.data is not None:
        df = st.session_state.data
        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        st.markdown("<h3 style='font-size:24px;'>Select a column to view and process:</h3>", unsafe_allow_html=True)
        selected_column = st.radio("", options=st.session_state.data.columns, index=0)
        
        if 'show_contents' not in st.session_state:
            st.session_state.show_contents = False

        if st.button("Show Column Contents"):
            st.session_state.show_contents = True

        if st.session_state.show_contents:
            st.write(f"You selected the column: {selected_column}")
            st.write(st.session_state.data[selected_column])

        template_prompt = st.text_input("Enter the template prompt for information extraction: (DO NOT change the value present within the curly braces)", f"What are the best foods in {{{selected_column}}}")
        results_df =pd.DataFrame(None)
        column_name = "Extracted Data" + f" from {selected_column}"
        
        if 'show_template_info' not in st.session_state:
            st.session_state.show_template_info = False

        if st.button("Start Search"):
            st.session_state.show_template_info = True

        new_column_series = []

        if st.session_state.show_template_info:

            if 'results_df' not in st.session_state:
                try:
                    agent = DynamicPromptAgent(
                        serp_api_key=serp_api_key,
                        groq_api_key=groq_api_key
                    )

                    entities = st.session_state.data[selected_column].dropna().unique()
                    results = []

                    entity_results = {}  # To store the results for each entity

                    for entity_name in entities:
                        if entity_name not in entity_results: 
                            entity = {selected_column: entity_name}
                            result = agent.execute_workflow(template_prompt, entity, selected_column)
                            entity_results[entity_name] = result

                    # Convert the entity results into a DataFrame for display
                    results_df = pd.DataFrame(
                        [(entity, result['data']) for entity, result in entity_results.items()],
                        columns=["entity", "data"]
                    )

                    st.session_state.results_df = results_df
                except Exception as e:
                    st.error(f"Error during information extraction: {e}")
                    st.stop()

                results_df = st.session_state.results_df
                st.subheader("Extracted Information")
                st.dataframe(results_df)

                column_name = f"Extracted Data from {selected_column}"
                df[column_name] = df[selected_column].map(lambda entity: entity_results.get(entity, {}).get('data', None))

                # Save new_column_series from the new column in the DataFrame to use it later on
                st.session_state.new_column_series = df[column_name].tolist()  # Store in session_state = df[column_name].tolist()  # Convert to a list for compatibility with the Google Sheets update

                st.subheader("Data updated in the CSV file")
                st.write(df)

            
            # Choice of google sheet update 
            if data_source == "Google Sheets":
                if 'google_sheet_updated' not in st.session_state:
                    st.session_state.google_sheet_updated = False

                confirm_update = st.checkbox("Confirm Google Sheets Update")
                if confirm_update:
                    if not st.session_state.google_sheet_updated:
                        try:
                            new_column_series =  st.session_state.new_column_series
                            data_manager.add_column_google_sheet(sheet_url,new_column_series, column_name)
                            st.session_state.google_sheet_updated = True
                            st.success("Google Sheets updated successfully.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.info("Google Sheets has already been updated.")

if __name__ == "__main__":
    main()