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
  - A search API key (SerpAPI ) to perform web searches.
  - An LLM API key (OpenAI GPT API or ChatGroq) for parsing and extracting information.
- **Libraries to import**:
  - All The required libraries for my application are listed in the requirements.txt file.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ai-agent.git
   cd ai-agent
## Set Up Environment Variables

1. **Create a .env file**  
   Create a `.env` file in the root directory of your project.
   The .env file has also been attached for better understanding.
2. **Add your API keys**  
   Add your API keys to the `.env` file in the following format:
   ```env
   SEARCH_API_KEY=your_search_api_key
   LLM_API_KEY=your_llm_api_key

3 **Add the Google Credentials for direct access to Google Sheets(optional)**
  Add the google credentials to the `.env` file in the following format:
  - type: service_account,
  - project_id: <YOUR_PROJECT_ID>,
  - private_key_id: <YOUR_PRIVATE_KEY_ID>,
  - private_key: <YOUR_PRIVATE_KEY>,
  - client_email: <YOUR_CLIENT_EMAIL>,
  - client_id: <YOUR_CLIENT_ID>,
  - auth_uri: https://accounts.google.com/o/oauth2/auth,
  - token_uri: https://oauth2.googleapis.com/token,
  - auth_provider_x509_cert_url: https://www.googleapis.com/oauth2/v1/certs,
  - client_x509_cert_url: <YOUR_CLIENT_X509_CERT_URL>

- For more information on setting up access to a Google Sheets link, you may refer to the following Medium post:
- [How to Get Credentials for Google Sheets](https://medium.com/@a.marenkov/how-to-get-credentials-for-google-sheets-456b7e88c430)



## Step-by-Step Guide to Using the AI Agent

### 1. Upload Data
- **Choose a Data Source**: Upload a CSV file containing the entities for information retrieval or connect a Google Sheet.

  ![image](https://github.com/user-attachments/assets/40cdc0df-af45-473b-b7f4-ca4c33506fc9)

- **Data Preview**: Once uploaded, a preview of the data will be displayed, allowing you to verify the contents.

  ![image](https://github.com/user-attachments/assets/0a421240-b8d9-4c39-ba64-190f5b2e8fc7)


### 2. Select the Main Column
- **Define the Entity Column**: Use the dropdown menu to select the column that contains the entities you are interested in (e.g., "company names"). This will serve as the main focus for the searches.

  ![image](https://github.com/user-attachments/assets/e2192a49-937b-4565-8e88-4d5798b3c076)


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

  ![image](https://github.com/user-attachments/assets/e02ceb08-cef1-43ce-980e-0ba4e9e025a6)


### 6. View and Export Results
- **Data Preview**: View the extracted results on the dashboard, where each entity and its related information (e.g., email, address) is displayed in the table.
- **Download Options**: Download the results as a CSV file or update the connected Google Sheet with the extracted information.

![image](https://github.com/user-attachments/assets/c8b4a909-7711-4abd-8a19-4bb7c297b8c2)

---

## API Integration Details
- **Search API**: The search API serpAPi is used to retrieve relevant URLs and snippets for each entity.
- **LLM API**: Chat Groq processes the search results and extracts specific information as defined by the prompt.

## Additional Integrated Features

1. **Multi-field Extraction**  
   The prompts have been customized to retrieve multiple pieces of information at once. For example, the prompt "Get the email and address for {company}" efficiently extracts both the email and address data.


2. **Direct Google Sheets Output**  
   A feature has been integrated that allows the extracted data to be written directly to Google Sheets, ensuring a seamless process for transferring and updating data in real-time.
   
    ![image](https://github.com/user-attachments/assets/8c44abeb-489a-4c55-bf15-1a36a63517cf)


   
  - Below is the Google sheet I have used:
    

     ![image](https://github.com/user-attachments/assets/b141463c-27be-40d9-93b3-1063a8ae126b)



  - Information Extraction:
    
     ![image](https://github.com/user-attachments/assets/2e57b3ae-e870-4572-8955-80cd162dc023)


    
  - Updated Google Sheet:

     ![image](https://github.com/user-attachments/assets/0f106172-c590-480d-b308-e172ca81814c)



  - Google Sheets Updated Message:

     ![image](https://github.com/user-attachments/assets/edcf734d-c429-46d4-bb5a-fe28a1acf637)



3. **Advanced Error Handling**  
   
    Advanced error handling mechanisms have been implemented to manage failed API calls or extraction errors. In case of an error, fallback mechanisms are triggered, and users are notified promptly to ensure smooth operation.



## Project Overview

  - For a breif explanation and walkthrough of the project, please refer to the video below:
   
    https://drive.google.com/file/d/1UgErdfcGiFWJUGvJwrp-s67YxQSlvA9k/view?usp=sharing
