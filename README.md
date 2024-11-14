# AI Agent for Automated Information Retrieval

## Project Overview
This AI agent application allows users to automate the process of gathering specific information about entities (such as company names) from the web. By uploading a dataset and defining custom search prompts, users can retrieve structured information directly from online sources using APIs. The application includes a user-friendly dashboard, which enables data upload, customizable queries, and exportable results.

## Features
- **Data Upload**: Upload a CSV file or connect a Google Sheet to provide the list of entities.
- **Custom Prompt Input**: Enter specific prompts for retrieving details about each entity.
- **Automated Web Search**: Use a search API to find relevant information online for each entity.
- **Information Extraction with LLM**: An LLM (Large Language Model) extracts the requested information from the search results.
- **Results Display and Export**: View and download results as a CSV file or update the Google Sheet with extracted data.

## Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher
- **API keys**:
  - A search API key (e.g., SerpAPI or ScraperAPI) to perform web searches.
  - An LLM API key (e.g., OpenAI GPT API) for parsing and extracting information.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ai-agent.git
   cd ai-agent
## Set Up Environment Variables

1. **Create a .env file**  
   Create a `.env` file in the root directory of your project.

2. **Add your API keys**  
   Add your API keys to the `.env` file in the following format:
   ```env
   SEARCH_API_KEY=your_search_api_key
   LLM_API_KEY=your_llm_api_key

3 **Add the Google Credentials for direct access to Google Sheets(optional)**
  Add the google credentials to the `.env` file in the following format:
  ```env
  type: service_account,
  project_id: <YOUR_PROJECT_ID>,
  private_key_id: <YOUR_PRIVATE_KEY_ID>,
  private_key: <YOUR_PRIVATE_KEY>,
  client_email: <YOUR_CLIENT_EMAIL>,
  client_id: <YOUR_CLIENT_ID>,
  auth_uri: https://accounts.google.com/o/oauth2/auth,
  token_uri: https://oauth2.googleapis.com/token,
  auth_provider_x509_cert_url: https://www.googleapis.com/oauth2/v1/certs,
  client_x509_cert_url: <YOUR_CLIENT_X509_CERT_URL>


## Step-by-Step Guide to Using the AI Agent

### 1. Upload Data
- **Choose a Data Source**: Upload a CSV file containing the entities for information retrieval or connect a Google Sheet.
- **Data Preview**: Once uploaded, a preview of the data will be displayed, allowing you to verify the contents.

### 2. Select the Main Column
- **Define the Entity Column**: Use the dropdown menu to select the column that contains the entities you are interested in (e.g., "company names"). This will serve as the main focus for the searches.

### 3. Enter a Prompt for Data Retrieval
- **Customizable Prompt**: In the input box, type a prompt that specifies the type of information you want to extract. For instance:
  - `"Retrieve the email address for {company}"`
  - `"Find the headquarters location of {company}"`
- **Placeholders**: Use `{entity}` in the prompt to dynamically replace each entity in your chosen column during searches.

### 4. Start the Web Search Process
- **Automated Search**: Click the "Start Search" button to initiate a search for each entity.
- **API Usage**: The agent will perform web searches using the specified API (e.g., SerpAPI), gathering search results for each entity based on the prompt.

### 5. Information Extraction via LLM
- **Data Parsing**: The application sends search results to the LLM, which then extracts the specified information according to your prompt.
- **Structured Data Output**: Extracted information is displayed in a structured, tabular format on the dashboard for easy viewing.

### 6. View and Export Results
- **Data Preview**: View the extracted results on the dashboard, where each entity and its related information (e.g., email, address) is displayed in the table.
- **Download Options**: Download the results as a CSV file or update the connected Google Sheet with the extracted information.

---

## API Integration Details
- **Search API**: The search API serpAPi is used to retrieve relevant URLs and snippets for each entity.
- **LLM API**: The LLM API processes the search results and extracts specific information as defined by the prompt.

## Additional Integrated Features

1. **Multi-field Extraction**  
   I have customized the prompts to retrieve multiple pieces of information at once. For example, the prompt "Get the email and address for {company}" efficiently extracts both the email and address data.

2. **Direct Google Sheets Output**  
   I have integrated a feature that allows the extracted data to be written directly to Google Sheets. This ensures a seamless process for transferring and updating data in real-time.

3. **Advanced Error Handling**  
   I have implemented advanced error handling mechanisms to manage failed API calls or extraction errors. In case of an error, fallback mechanisms are triggered, and users are notified promptly to ensure smooth operation.
