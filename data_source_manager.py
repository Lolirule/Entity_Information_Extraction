import streamlit as st
import pandas as pd

from typing import Optional, List, Dict

import gspread
from google.oauth2 import service_account

import os
from dotenv import load_dotenv
from gspread_pandas import Spread
from google.oauth2.credentials import Credentials

import logging

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
    
    def add_column_google_sheet(self, sheet_url: str, new_column: list, column_name: str):
        try:
            # Get the Google Sheet by URL and the first worksheet
            sheet_key = sheet_url.split('/')[5]
            sheet = self.gc.open_by_key(sheet_key)
            worksheet = sheet.get_worksheet(0)

            # Convert the worksheet data into a DataFrame
            df = pd.DataFrame(worksheet.get_all_records())

            # Process the new column (convert lists to comma-separated strings)
            new_column = [', '.join(map(str, x)) if isinstance(x, list) else str(x) for x in new_column]

            # Add the new column to the DataFrame
            df[column_name] = new_column

            # Update the Google Sheet with the updated DataFrame
            worksheet.update([df.columns.values.tolist()] + df.values.tolist())
            st.success(f"Column '{column_name}' added successfully to Google Sheet")
        except Exception as e:
            st.error(f"Error adding column: {str(e)}")