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
            self.groq_client = Groq(api_key=groq_api_key)
            self.groq_llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
            self.tools = load_tools(["serpapi"], serpapi_api_key=self.serp_api_key)
            
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.groq_llm,
                agent_type="self-ask-with-search",
                verbose=True
            )
            logger.info("DynamicPromptAgent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DynamicPromptAgent: {str(e)}")
            raise

    def process_with_groq(self, prompt: str) -> str:
        """
        Process text using Groq's LLaMA 3 model.
        """
        structured_prompt = f"""
        You are a general information extraction assistant. Your task is to extract information about the entity mentioned in the prompt.

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
        """Helper method to create a consistent empty response"""
        return {
            "fields": [],
            "data": {},
            "entity": entity_name
        }
    
    def format_to_table(self, prompt: str, entity_name: str = "") -> Dict:
        try:
            # Step 1: Get initial response
            initial_response = self.process_with_groq(prompt)
            
            if not initial_response:
                raise ValueError("Response is Empty")

            # Step 2: Create formatting prompt
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
            
            # Step 3: Parsing the  JSON response
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
            
            # Add entity name to the response
            formatted_data["entity"] = entity_name
            return formatted_data
                
        except Exception:
            return self._create_empty_response(entity_name)

    

    def execute_workflow(self, template_prompt: str, entity: dict, column_name: str) -> dict:
        
        # Validate inputs
        if not isinstance(entity, dict):
            raise ValueError(f"Entity must be a dictionary, got {type(entity)}")
        
        if not column_name:
            raise ValueError("Column name cannot be empty")
            
        if column_name not in entity:
            raise KeyError(f"Column '{column_name}' not found in entity data: {list(entity.keys())}")
        
        entity_name = entity[column_name]
        
        # Format the search prompt
        try:
            search_prompt = template_prompt.format(**entity)
            
            # Extract and format information
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
        """Load data from Google Sheet"""
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
        """Add a new column to a Google Sheet"""
        try:
            # Get the Google Sheet by URL and the first worksheet
            sheet_key = sheet_url.split('/')[5]
            sheet = self.gc.open_by_key(sheet_key)
            worksheet = sheet.get_worksheet(0)

            # Convert worksheet to a DataFrame
            df = pd.DataFrame(worksheet.get_all_records())

            new_column = new_column.apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))

            # Add the new column to the DataFrame
            df[column_name] = new_column

            # Update the entire sheet with the modified DataFrame
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

        # selected_column = st.selectbox("Select a column:", options=st.session_state.data.columns)
        st.markdown("<h3 style='font-size:24px;'>Select a column to view:</h3>", unsafe_allow_html=True)
        selected_column = st.radio("", options=st.session_state.data.columns, index=0)
        
        # Initialize session state to track "Show Column Contents" visibility
        if 'show_contents' not in st.session_state:
            st.session_state.show_contents = False

        # Button to toggle content visibility
        if st.button("Show Column Contents"):
            st.session_state.show_contents = True

        # Display selected column contents if show_contents is True
        if st.session_state.show_contents:
            st.write(f"You selected the column: {selected_column}")
            st.write(st.session_state.data[selected_column])

        template_prompt = st.text_input("Enter the template prompt for information extraction: (DO NOT change the value present within the curly braces)", f"What are the best foods in {{{selected_column}}}")
        results_df =pd.DataFrame(None)
        column_name = "Extracted Data" + f" from {selected_column}"
                # Run extraction only after user submits template prompt
        
        if 'show_template_info' not in st.session_state:
            st.session_state.show_template_info = False

        # Button to show/hide template info
        if st.button("Submit Template Prompt"):
            st.session_state.show_template_info = True  # Set to True when button is pressed

        # Display template prompt information if `show_template_info` is True
        if st.session_state.show_template_info:
            
            agent = DynamicPromptAgent(
                serp_api_key=serp_api_key,
                groq_api_key=groq_api_key
            )

            entities = st.session_state.data[selected_column].dropna().unique()
            
            results = []
            for entity_name in entities:
                entity = {selected_column: entity_name}
                result = agent.execute_workflow(template_prompt, entity, selected_column)
                results.append(result)
            # Display results in tabular format
            st.subheader("Extracted Information")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            df[column_name] = results_df['data']
            st.subheader("Data updated in the CSV file\nYou may download the updated csv file")
            st.write(df)

            if data_source == "Google Sheets":
                edit_choice = st.selectbox("Would you like to add the extracted data to your original data source?\n NOTE:Adding this data to Google Sheets will make permanent changes. Do you want to proceed?", ["Select", "Yes", "No"])
                # Add the extracted data as a new column to the original data source
                if edit_choice == "Yes" :
                    if 'data' in results_df.columns:
                        new_column_series = results_df['data']
                        # Add new column to Google Sheet
                        try:
                            data_manager.add_column_google_sheet(sheet_url, new_column_series, column_name)
                            st.success("New column added to Google Sheet successfully.")
                        except:
                            print("Give access level for editing")

if __name__ == "__main__":
    main()