import ollama
import json

SYSTEM_PROMPT = """
# Identity
    - You are an intelligent assistant designed to analyze incoming JSON messages from different healthcare institutions.
    - Your core function is to analyze incoming JSON messages and focus on the semantic meaning of their keys.
    - You process a stream of JSON messages, and for each processed JSON message you generate a unified, harmonized, JSON message.
    
# Input 
    - A message containing a JSON representing the patient information. 
    - The history of previous JSON messages.
    
# Output 
    - A harmonized JSON message after each streamed JSON message that captures the essential semantics of all JSON messages processed so far.

# Instructions    
    The following are guidelines for the harmonization process to generate the harmonized JSON message given the current and previous JSON messages:
        - The first harmonized JSON message is exactly equal to the first JSON message.
        - For subsequent JSON messages composing the stream, you must integrate them into the existing harmonized JSON message. This involves:
            - Key Meaning: 
                - Extrapolate the semantic meaning of each key in the current message. 
                - If different keys in the same JSON message represent the same underlying concept, merge them to a single, consistent key name that generalize them.
                - Identify semantic similarity also over the keys of previous JSON messages and harmonized JSON messages.
            - Value Consistency: 
                - Ensure consistency in data types and formats for values associated with harmonized keys.
            - Nested Structures: 
                - Identify groups of semantically corrrelated keys and merge them into nested structures.
                - The keys of the nested structure present the semantic meaning of the group.
            - Redundancy Elimination: 
                - Remove redundant keys and values that do not add new information to the harmonized JSON message.
                - Avoid adding redundant keys and values that are already represented in the harmonized JSON message.
            - CRITICAL: 
                - Generate ONLY a JSON object as the output.
                - Generate a valid JSON object that starts with `{' and ends with `}'.
                - DO NOT include any introductory text, explanations, concluding remarks, or fields for metadata.
                - Ensure no comments, explanations, or descriptive text appear inside the JSON structure, even for field values.
                - DO NOT introduce a nesting level in the harmonized JSON message to specfy the data type of the value.
     
# Example
    [{'role': 'user', 'content': 'Incoming JSON for harmonization:
            {
                "date_of_birth": "2023-10-27"
            }
            Current harmonized schema:
            No current schema (first message)    
            Analyze the incoming JSON and provide the updated harmonized schema as a JSON object.'}, 
     {'role': 'assistant', 'content': '
            {
                "date_of_birth": "STRING"
            }, 
     {'role': 'user', 'content': 'Incoming JSON for harmonization:
            {
                "dob": 20231027, 
                "name": "John", 
                "surname": "Doe", 
                "address_street": "123 Main St", 
                "adress_city": "Anytown", 
                "phone_number": 123-456-7890
            }
            Current harmonized schema:
            {
                "date_of_birth": "STRING"
            }
            Analyze the incoming JSON and provide the updated harmonized schema as a JSON object.'},
     {'role': 'assistant', 'content': '
            {
                "date_of_birth": "STRING", 
                "name": {
                    "first": "STRING", 
                    "last": "STRING"
                    }, 
                "address": {
                    "street": "STRING", 
                    "city": "STRING"
                    }, 
                "phone_number": "INTEGER"
            }]

"""

class OLLAMAHarmonizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.system_prompt = SYSTEM_PROMPT
        # Initialize messages with the system prompt
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.harmonized_schema = {} # To store the current harmonized JSON
        self.conversation_count = 0

    def predict(self, user_data):
        """
        Processes an incoming JSON message to update and generate a harmonized schema.

        Parameters:
            user_data (str): The incoming JSON message as a string.

        Returns:
            str: A JSON string representing the harmonized schema, or an error message.
        """
        
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
            
            # Call ollama.chat with stream=False (no need for streaming)
            try:
                # Removed the streaming loop, as stream=False returns the full response directly
                response = ollama.chat(
                    model=self.model_name,
                    messages=self.messages, # Keep the conversation history for harmonization task
                    stream=False, # Changed to False
                    format='json',  # Request JSON output from the model
                    options={
                        'temperature': 0.4,  # Lower temperature for more consistent JSON output
                        'top_p': 0.7 # Use top-p sampling to control diversity
                    }
                )
                
                # The full response content is directly in response['message']['content']
                full_response_content = response['message']['content']
                   
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
    
    ollama_model = OLLAMAHarmonizer(model_name='gemma3') # Changed model name to mistral for consistency

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
