import streamlit as st

from langchain.agents import load_tools, initialize_agent
from groq import Groq

import json
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

import os
from dotenv import load_dotenv

import time
import logging

# Load environment variables from .env file
load_dotenv()
serp_api_key = os.getenv("SERP_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

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
    
    def _create_empty_response(self, entity_name: str) -> dict:

        return {
            "fields": [],
            "data": {},
            "entity": entity_name
        }
    
    def format_to_table(self, prompt: str, entity_name: str = "") -> dict:
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
