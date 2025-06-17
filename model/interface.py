import ollama
import json
import re

# This is our persistent system instruction for the OLLAMA.
SYSTEM_PROMPT = """
You are an intelligent agent designed to analyze and harmonize incoming JSON messages from different hospitals.
Your core function is to analyze and harmonize incoming JSON messages related to patients health state, focusing on both their structural and semantic properties.

Input: 
    A message containing a JSON representing the patient information. JSON messages arrives in the form of a stream.
    
Output: 
    A harmonized JSON schema after each streamed JSON message that captures the essential structure and semantics of the incoming messages.
    CRITICAL: Your response MUST be a valid JSON object only. Do not include any explanatory text, markdown formatting, or additional commentary. Just return the harmonized JSON schema.

The following are guidelines to analyze the stream of JSON messages and produce a harmonized JSON schema:
1.  Structural Analysis:
    - Nesting Level: Maintain a consistent nesting depth over the stream of JSON messages. 
    - Keys per Level: Maintain a consistent set of keys at each nesting level over the stream of JSON messages. 

2.  Semantic Analysis:
    - Key Meaning: Understand the semantic meaning of each key. If different keys represent the same underlying concept (e.g., "patient_id" and "subject_id"), merge them to a single, consistent key name (e.g., "id").
    - Value Consistency: Ensure consistency in data types and formats for values associated with harmonized keys. For example, if "id" is sometimes a string and sometimes an integer, harmonize to a string.

Harmonization Process:
 - If it's the first message in the stream, the incoming JSON defines the initial harmonized schema.
 - For subsequent JSON messages composing the stream, you must integrate their schemas into the existing harmonized schema. This involves:
    * Adding new, relevant keys and nested structures.
    * Resolving structural conflicts (nesting, key counts) to maintain a logical and comprehensive schema.
    * Resolving semantic conflicts (key names, value types) by identifying synonyms and standardizing terminology.
    * Retaining important information while eliminating redundancy.

Example:
 - First JSON message: {"subject_id": "12345", "blood_pressure_systolic": 120, "blood_pressure_diastolic": 80} 
        Harmonized JSON schema after the first JSON message: {"subject_id": "string", "blood_pressure": {"systolic": "int", "diastolic": "int"}}
 - Second JSON message: {"patient_id": "102030", "heart_rate": 72, "ward": "ICU", "admition" : {"date": "2023-10-01", "time": "10:00 AM"}}
        Harmonized JSON schema after the second JSON message: {"id": "string", "health_parameters": {"blood_pressure": {"systolic": "int", "diastolic": "int"}, "heart_rate": "int"}, "ward": "string", "admition" : {"date": "2023-10-01", "time": "10:00 AM"}}
 - ...   
 
If you cannot process the JSON or harmonize it, respond with: {"error": "description of the issue"}
"""

class OLLAMA:
    def __init__(self, model_name, system_prompt=SYSTEM_PROMPT, max_retries=3):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        # Initialize messages with the system prompt
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.harmonized_schema = {} # To store the current harmonized JSON
        self.conversation_count = 0

    def predict(self, user_data, retry_count=0):
        
        try:
            # Attempt to parse user_data as JSON
            try:
                user_json = json.loads(user_data)
                # If parsed successfully, use the JSON object for the message
                current_schema_text = json.dumps(self.harmonized_schema, indent=2) if self.harmonized_schema else "No current schema (first message)"
                user_message_content = f"Incoming JSON for harmonization:\n{json.dumps(user_json, indent=2)}\n\nCurrent harmonized schema:\n{current_schema_text}\n\nAnalyze the incoming JSON and provide the updated harmonized schema as a JSON object."
                
            except json.JSONDecodeError:
                # If not a valid JSON, return error
                error_message = {"error": "Invalid JSON input. Please provide valid JSON for harmonization."}
                return json.dumps(error_message, indent=2)

            # Add the new user message to the conversation history
            self.messages.append({'role': 'user', 'content': user_message_content})
            
            # Call ollama.chat with stream=True
            try:
                stream = ollama.chat(
                    model=self.model_name,
                    messages=self.messages,
                    stream=True,
                    format='json',  # Request JSON output from the model
                    options={
                        'temperature': 0.1,  # Lower temperature for more consistent JSON output
                        'top_p': 0.9
                    }
                )
                
                # Process the streamed response
                full_response_content = ""
                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_response_content += chunk['message']['content']
                   
                # Validate and clean the response
                try:
                    model_response_json = json.loads(full_response_content)
                except json.JSONDecodeError:
                    model_response_json = None
                
                if model_response_json is not None:
                    # Update the harmonized schema if the model provided a valid JSON
                    self.harmonized_schema = model_response_json
                    
                    # Add the model's response to the conversation history
                    self.messages.append({'role': 'assistant', 'content': json.dumps(model_response_json, indent=2)})
                    self.conversation_count += 1
                    
                    return json.dumps(model_response_json, indent=2)
                else:
                    # TODO: may add a retrying logic
                    raise ValueError("Model response is not a valid JSON object. Model response: " + full_response_content)

            except Exception as e:
                raise RuntimeError(f"Error during model prediction: {str(e)}")
                
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the input: {str(e)}")


if __name__ == "__main__":
    
    ollama_model = OLLAMA(model_name='mistral')

    print("JSON Harmonization Agent initialized. Provide JSON input.")
    print("Commands: 'exit' to quit, 'reset' to reset conversation, 'stats' to show stats")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou (JSON or command): ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        if not user_input:
            print("Please provide some input.")
            continue
        
        prediction = ollama_model.predict(user_input)
        
        if prediction:
            print(f"\nAgent response:\n{prediction}")
        else:
            print("No response received from the model.")