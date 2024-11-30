import google.generativeai as genai
import os


class GeminiQueryEngine:
    def __init__(self, api_key):
        """
        Initialize the Gemini query engine with API configuration

        Args:
            api_key (str): Google AI API key
        """
        # Configure the API key
        genai.configure(api_key=api_key)

    def generate_payload(query, system_info=None):
        """
        Generate a comprehensive payload for Gemini query

        Args:
            query (str): User's query or prompt
            system_info (dict, optional): Additional system context

        Returns:
            dict: Configured payload for Gemini API
        """
        # Default system information if not provided
        default_system_info = {
            "role": "system",
            "parts": [
                "You are a helpful AI assistant designed to provide accurate and concise information."
            ]
        }

        # Merge or use default system information
        system_info = system_info or default_system_info

        # Generation configuration
        generation_config = {
            "temperature": 0.7,
            "top_p": 1.0,
            "max_output_tokens": 2048,
            "stop_sequences": []
        }

        # Tools configuration (optional)
        tools = [
            {
                "function_declarations": [
                    {
                        "name": "get_current_weather",
                        "description": "Retrieve current weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ]
            }
        ]

        # Construct full payload
        payload = {
            "contents": [
                system_info,
                {
                    "role": "user",
                    "parts": [{"text": query}]
                }
            ],
            "generation_config": generation_config,
            "tools": tools
        }

        return payload

    def query_gemini(self, query, system_info=None):
        """
        Send query to Gemini 1.5 Flash model

        Args:
            query (str): User's query or prompt
            system_info (dict, optional): Additional system context

        Returns:
            str: Response from Gemini
        """
        try:
            # Prepare payload
            payload = generate_payload(query, system_info)

            # Select the Gemini 1.5 Flash model
            model = genai.GenerativeModel('gemini-1.5-flash-002')

            # Generate response
            response = model.generate_content(payload)

            return response.text

        except Exception as e:
            return f"An error occurred: {str(e)}"

