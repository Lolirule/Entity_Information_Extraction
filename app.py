import streamlit as st
import pandas as pd
from langchain.agents import load_tools, initialize_agent
from groq import Groq
from typing import Optional, List, Dict
import json
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
import gspread
from google.oauth2 import service_account
from typing import Optional
import os
from dotenv import load_dotenv
from gspread_pandas import Spread
from google.oauth2.credentials import Credentials
import base64
from io import StringIO
from typing import List, Dict
import time
import logging

# Load environment variables from .env file
load_dotenv()
serp_api_key = os.getenv("SERP_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Access the environment variables
google_project_id = os.getenv('GOOGLE_PROJECT_ID')
google_private_key = os.getenv('GOOGLE_PRIVATE_KEY')
google_client_email = os.getenv('GOOGLE_CLIENT_EMAIL')
google_client_id = os.getenv('GOOGLE_CLIENT_ID')
google_auth_uri = os.getenv('GOOGLE_AUTH_URI')
google_token_uri = os.getenv('GOOGLE_TOKEN_URI')
google_auth_provider_x509_cert_url = os.getenv('GOOGLE_AUTH_PROVIDER_X509_CERT_URL')
google_client_x509_cert_url = os.getenv('GOOGLE_CLIENT_X509_CERT_URL')
google_private_key_id = os.getenv('GOOGLE_PRIVATE_KEY_ID')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DynamicPromptAgent:
    def __init__(self, serp_api_key: str, groq_api_key: str):
        try:
            logger.info("Initializing DynamicPromptAgent...")
            self.serp_api_key = serp_api_key
            
            self.groq_client = self.initialize_groq_client(groq_api_key)
            
            self.groq_llm = self.initialize_chat_groq(groq_api_key, model_name="llama3-8b-8192")
            
            self.tools = self.load_tools(self.serp_api_key)
            
            self.agent = self.initialize_agent(self.tools, self.groq_llm)
            
            logger.info("DynamicPromptAgent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DynamicPromptAgent: {str(e)}")
            raise

    def initialize_groq_client(self, groq_api_key: str):

        try:
            logger.info("Initializing Groq client...")
            return Groq(api_key=groq_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            raise RuntimeError("Error initializing Groq client.") from e

    def initialize_chat_groq(self, groq_api_key: str, model_name: str):

        try:
            logger.info(f"Initializing ChatGroq LLM with model: {model_name}...")
            return ChatGroq(api_key=groq_api_key, model_name=model_name)
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq LLM: {str(e)}")
            raise RuntimeError("Error initializing ChatGroq LLM.") from e

    def load_tools(self, serp_api_key: str):

        try:
            logger.info("Loading tools...")
            return load_tools(["serpapi"], serpapi_api_key=serp_api_key)
        except Exception as e:
            logger.error(f"Failed to load tools: {str(e)}")
            raise RuntimeError("Error loading tools.") from e

    def initialize_agent(self, tools, groq_llm):

        try:
            logger.info("Initializing the agent...")
            return initialize_agent(
                tools=tools,
                llm=groq_llm,
                agent_type="self-ask-with-search",
                verbose=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise RuntimeError("Error initializing agent.") from e
    def process_with_groq(self, prompt: str) -> str:
        """
        Process text using Groq's LLaMA 3 model.
        """
        structured_prompt = f"""
        You are a general information extraction assistant. Your task is to extract information about the entity mentioned in the prompt and nothing irrelevant must be extracted.
        try to get as specific information as you can but don't narrow down your search.Try searching from multiple websearches to get the most relevant information.
        PROMPT: {prompt}

        Rules:
        1. Focus on retrieving publicly available information related to each entity from the official websites.
        2. Respect website terms of service, and do not attempt to retrieve sensitive or private data.
        3. Perform web searches for each entity using the custom prompt and gather relevant web results.
        """
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a general information extraction assistant. Only provide the specific information requested."
                },
                {
                    "role": "user",
                    "content": structured_prompt
                }
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            stream=False
        )
        return chat_completion.choices[0].message.content
    
    def _create_empty_response(self, entity_name: str) -> Dict:

        return {
            "fields": [],
            "data": {},
            "entity": entity_name
        }
    
    def format_to_table(self, prompt: str, entity_name: str = "") -> Dict:
        try:
            # This is the initial information extraction part where entity related data is extracted.
            initial_response = self.process_with_groq(prompt)
            
            if not initial_response:
                raise ValueError("Response is Empty")

            # Extracting the required infromation asked by the user prompt

            formatting_prompt = f"""
            Convert the following extracted information into a structured JSON format, containing only the most relevant items related to the user query without additional information.

            The goal is to extract only the essential data based on the user's query, excluding any unnecessary details such as descriptions or extra fields.you may add small descriptions if required.

            Information to format:
            {initial_response}

            Rules:
            1. Create a JSON object with "fields" listing only the names explicitly requested in the query, and "data" containing a simple list of relevant items.
            2. Use consistent field names in snake_case and avoid additional fields such as descriptions, ingredients, or supplementary details.
            3. If a field is missing or irrelevant to the query, leave it out.
            4. Extract the field name only from the query given by the template prompt given by user and mention the column_name.
            5. Example format:
            {{
                
                "data": {{
                    "relevant_items": ["item1", "item2", "item3"]
                }}
            }}
            """

            format_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data formatting assistant. Convert information into structured JSON."
                    },
                    {
                        "role": "user",
                        "content": formatting_prompt
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.2,
                max_tokens=400,
                top_p=1,
                stream=False
            )
            
            # Parsing the  JSON response
            response_content = format_completion.choices[0].message.content
            
            # Extracting JSON from the response if it's wrapped in markdown or other text
            try:
                formatted_data = json.loads(response_content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[\s\S]*\}', response_content)
                if json_match:
                    try:
                        formatted_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        return self._create_empty_response(entity_name)
                else:
                    return self._create_empty_response(entity_name)
            
            # Adding entity name to the response
            formatted_data["entity"] = entity_name
            return formatted_data
                
        except Exception:
            return self._create_empty_response(entity_name)

    

    def execute_workflow(self, template_prompt: str, entity: dict, column_name: str) -> dict:
        
        if not isinstance(entity, dict):
            raise ValueError(f"Entity must be a dictionary, got {type(entity)}")
        
        if not column_name:
            raise ValueError("Column name cannot be empty")
            
        if column_name not in entity:
            raise KeyError(f"Column '{column_name}' not found in entity data: {list(entity.keys())}")
        
        entity_name = entity[column_name]
        
        # The search prompt is formatted
        try:
            search_prompt = template_prompt.format(**entity)
            
            formatted_data = self.format_to_table(search_prompt, entity_name)
            
            if not formatted_data.get("fields") and not formatted_data.get("data"):
                return {
                    "entity": entity_name,
                    "error": "No data extracted",
                    "status": "error"
                }
        
            result = {
                "entity": entity_name,
                "data": formatted_data.get("data", {}),
                "status": "success"
            }
            return result
            
        except Exception as e:
            return {
                "entity": entity.get(column_name, "unknown"),
                "error": str(e),
                "status": "error"
            }

class DataSourceManager:
    def __init__(self):
        self.setup_google_credentials()
    
    def setup_google_credentials(self):
        """Setup Google Sheets credentials using environment variables"""
        try:
            # Construct the credentials using the environment variables
            credentials_info = {
                "type": "service_account",
                "project_id": google_project_id,
                "private_key_id": google_private_key_id,
                "private_key": google_private_key.replace("\\n", "\n"),  # Fix newline escape issue
                "client_email": google_client_email,
                "client_id": google_client_id,
                "auth_uri": google_auth_uri,
                "token_uri": google_token_uri,
                "auth_provider_x509_cert_url": google_auth_provider_x509_cert_url,
                "client_x509_cert_url": google_client_x509_cert_url
            }

            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )

            self.gc = gspread.authorize(credentials)
        except Exception as e:
            st.error(f"Failed to setup Google credentials: {str(e)}")
            self.gc = None

    def load_google_sheet(self, sheet_url: str) -> Optional[pd.DataFrame]:
        try:
            sheet_key = sheet_url.split('/')[5]
            sheet = self.gc.open_by_key(sheet_key)
            worksheet = sheet.get_worksheet(0)
            data = worksheet.get_all_records()
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error loading Google Sheet: {str(e)}")
            return None
    
    def add_column_google_sheet(self, sheet_url: str, new_column: pd.Series, column_name: str):
        try:
            # Get the Google Sheet by URL and the first worksheet
            sheet_key = sheet_url.split('/')[5]
            sheet = self.gc.open_by_key(sheet_key)
            worksheet = sheet.get_worksheet(0)

            df = pd.DataFrame(worksheet.get_all_records())

            new_column = new_column.apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))

            df[column_name] = new_column

            worksheet.update([df.columns.values.tolist()] + df.values.tolist())
            st.success(f"Column '{column_name}' added successfully to Google Sheet")
        except Exception as e:
            st.error(f"Error adding column: {str(e)}")
        

def main():
    st.set_page_config(page_title="Data Source Interface", layout="wide")
    
    if 'data' not in st.session_state:
        st.session_state.data = None

    st.title("Data Source Interface")
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


        if st.session_state.show_template_info:
            st.subheader("Information Extraction in Progress...")

            if 'results_df' not in st.session_state:
                try:
                    agent = DynamicPromptAgent(
                        serp_api_key=serp_api_key,
                        groq_api_key=groq_api_key
                    )

                    # Selecting unique entities from the selected column
                    entities = st.session_state.data[selected_column].dropna().unique()
                    results = []

                    entity_results = {}  

                    for entity_name in entities:
                        if entity_name not in entity_results: 
                            entity = {selected_column: entity_name}
                            result = agent.execute_workflow(template_prompt, entity, selected_column)
                            entity_results[entity_name] = result  # Store the result

                    results_df = pd.DataFrame(
                        [(entity, result['data']) for entity, result in entity_results.items()],
                        columns=["Entity", "Extracted Data"]
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

            st.subheader("Data updated in the CSV file")
            st.write(df)

            if data_source == "Google Sheets":
                if 'google_sheet_updated' not in st.session_state:
                    st.session_state.google_sheet_updated = False

                confirm_update = st.checkbox("Confirm Google Sheets Update")
                if confirm_update:
                    if not st.session_state.google_sheet_updated:
                        try:
                            new_column_series = results_df['data']
                            data_manager.add_column_google_sheet(sheet_url, new_column_series, column_name)
                            st.session_state.google_sheet_updated = True
                            st.success("Google Sheets updated successfully.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.info("Google Sheets has already been updated.")

if __name__ == "__main__":
    main()