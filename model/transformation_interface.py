import ollama
import json

SYSTEM_PROMPT = """
# Identity
    - You are an intelligent assistant designed to transform incoming JSON messages, into a given harmonized schema.
    - Specifically, JSON messages are produced by healthcare institutions and contain patient information.
    - Your core function is to analyze an incoming JSON message, semantically map its keys to the keys of a provided harmonized schema, and then generate a new JSON message that conforms to the harmonized schema's structure and data types, populating it with values from the incoming JSON where a semantic match is found.

# Input
    - An 'Incoming JSON for transformation' representing **patient information** from a healthcare institution.
    - A 'Current harmonized schema' which is the target schema for transformation, including data types (e.g., "STRING", "INTEGER", "BOOLEAN", "ARRAY", "OBJECT").

# Output
    - A transformed JSON message that strictly adheres to the 'harmonized schema', with values from the 'Incoming JSON for transformation' mapped appropriately.
"""

class OLLAMATransformer:
    def __init__(self, model_name, temperature, top_p):
        self.model_name = model_name
        self.system_prompt = SYSTEM_PROMPT
        # self.messages is now primarily for displaying the conversation history
        # but is not directly passed to ollama.chat for each prediction
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.conversation_count = 0
        
        # Chat options
        self.temperature = temperature
        self.top_p = top_p


    def predict(self, incoming_json: dict, target_harmonized_schema: dict):
        """
        Transforms an incoming JSON message into a target harmonized schema using the OLLAMA model.
        Each prediction is stateless with respect to previous transformations.
        
        Parameters:
            incoming_json (dict): The JSON object to be transformed.
            target_harmonized_schema (dict): The target harmonized schema defining the desired structure and data types.

        Returns:
            str: A JSON string representing the transformed message, or an error message.
        """
        try:
            # Construct the user message content, including both the incoming JSON and the target schema.
            user_message_content = """
            # Instructions
            
                The following are guidelines for the JSON transformation process:
                    - Semantic Mapping: For each key-value pair in the 'Incoming JSON for transformation', identify the key in the 'Current harmonized schema' that has the highest semantic similarity.
                    - Value Assignment:
                        - Assign the value from the 'Incoming JSON for transformation' to the semantically mapped key in the output JSON.
                        - If the harmonized schema specifies a different data type for a field than the incoming value, convert the incoming value to match the harmonized schema's data type. For example:
                            - If schema type is "INTEGER" and incoming is "2023-10-27" (string), try to extract digits (e.g., 20231027). If incoming is "123 Main St", set to "n/a".
                            - If schema type is "STRING" and incoming is 20231027 (integer), convert to "20231027".
                            - If schema type is "BOOLEAN", convert truthy/falsy values accordingly (e.g., "true" to true, "false" to false, 1 to true, 0 to false).
                            - For complex types like "ARRAY" or "OBJECT", ensure the structure matches and fill in nested values recursively.
                        - If a value in the 'Incoming JSON for transformation' cannot be meaningfully converted to the target data type, or if a key in the harmonized schema has no semantically similar counterpart in the incoming JSON, assign "n/a" as its value.
                        - For nested structures in the harmonized schema, populate their sub-keys based on semantic mapping from the incoming JSON's flat or nested keys. If a nested field in the harmonized schema cannot be populated from the incoming JSON, its value should be "n/a".
                    - Structure Adherence: The output JSON must strictly follow the structure (nested objects, key names) of the 'Current harmonized schema'. Do not introduce new keys or alter the structure of the harmonized schema.
                    - CRITICAL:
                        - Generate ONLY a JSON object as the output.
                        - Generate a valid JSON object that starts with `{` and ends with `}`.
                        - DO NOT include any introductory text, explanations, concluding remarks, or fields for metadata.
                        - Ensure no comments, explanations, or descriptive text appear inside the JSON structure, even for field values.
                        - DO NOT modify the harmonized JSON schema, only update its values with those of the input JSON message to be transformed.

            # Examples

                - Example 1:
                    
                    [{'role': 'user', 'content': 'Incoming JSON for transformation:
                            {
                                "dob": "2023-10-27"
                            }
                            Harmonized schema:
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
                            }
                            Analyze the incoming JSON and provide the transformed JSON object.'},
                    {'role': 'assistant', 'content': '
                            {
                                "date_of_birth": "2023-10-27",
                                "name": {
                                    "first": "n/a",
                                    "last": "n/a"
                                    },
                                "address": {
                                    "street": "n/a",
                                    "city": "n/a"
                                    },
                                "phone_number": "n/a"
                            }'}]
                            
                - Example 2:
                
                    [{'role': 'user', 'content': 'Incoming JSON for transformation:
                            {
                                "dob": 20231027,
                                "name": "John",
                                "surname": "Doe",
                                "address_street": "123 Main St",
                                "adress_city": "Anytown",
                                "phone_number": "123-456-7890"
                            }
                            Harmonized schema:
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
                            }
                            Analyze the incoming JSON and provide the transformed JSON object.'},
                    {'role': 'assistant', 'content': '
                            {
                                "date_of_birth": "20231027",
                                "name": {
                                    "first": "John",
                                    "last": "Doe"
                                    },
                                "address": {
                                    "street": "123 Main St",
                                    "city": "Anytown"
                                    },
                                "phone_number": 1234567890
                            }'}]            
            """
            
            user_message_content += (
                f"\n\nIncoming JSON for transformation:\n{json.dumps(incoming_json, indent=2)}\n\n"
                f"Harmonized schema:\n{json.dumps(target_harmonized_schema, indent=2)}\n\n"
                f"Analyze the incoming JSON and provide the transformed JSON object."
            )

            # Create a temporary messages list for the current OLLAMA call
            # This ensures each prediction is independent and only includes the system prompt
            # and the current user request.
            messages_for_ollama_call = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_message_content}
            ]

            # Add the user message to the instance's conversation history (for display/tracking, not for OLLAMA input)
            self.messages.append({'role': 'user', 'content': user_message_content})

            # Call ollama.chat with stream=False (no need for streaming in this batch-like transformation)
            try:
                # Removed the streaming loop, as stream=False returns the full response directly
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages_for_ollama_call, # Pass the temporary, stateless messages list
                    stream=False, # Changed to False for simplicity as full response is awaited
                    format='json',  # Request JSON output from the model
                    options={
                        'temperature': self.temperature,  # Lower temperature for more consistent JSON output
                        'top_p': self.top_p  # Use top-p sampling to control diversity
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
                    # Add the model's response to the instance's conversation history
                    self.messages.append({'role': 'assistant', 'content': json.dumps(model_response_json, indent=2)})
                    self.conversation_count += 1
                    return json.dumps(model_response_json, indent=2)
                else:
                    # TODO: may add a retrying logic
                    raise ValueError(f"Model response is not a valid JSON object. Model response: {full_response_content}")

            except Exception as e:
                raise RuntimeError(f"Error during model prediction: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"An error occurred while processing the input: {str(e)}")

    def reset_conversation(self):
        """Resets the conversation history, keeping only the system prompt."""
        # Note: The `predict` method now constructs messages stateless for OLLAMA calls.
        # This reset is primarily for the internal `self.messages` if you were displaying it.
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.conversation_count = 0
        print("Conversation history reset.")

    def get_stats(self):
        """Prints current conversation statistics."""
        print(f"Total predictions made: {self.conversation_count}")
        print(f"Current conversation length (messages): {len(self.messages)}")


if __name__ == "__main__":

    ollama_model = OLLAMATransformer(model_name='gemma3')

    print("JSON Transformation Agent initialized. Provide JSON input and a target harmonized schema.")
    print("Commands: 'exit' to quit, 'reset' to reset conversation, 'stats' to show stats")
    print("-" * 60)

    # Dummy example demonstrating the transformation
    print("\n--- Running a dummy transformation example ---")

    # Example 1 from the prompt
    incoming_json_1 = {
        "dob": "2023-10-27"
    }
    harmonized_schema_1 = {
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
    }

    print("\nIncoming JSON (Example 1):")
    print(json.dumps(incoming_json_1, indent=2))
    print("\nTarget Harmonized Schema (Example 1):")
    print(json.dumps(harmonized_schema_1, indent=2))

    transformed_json_1 = ollama_model.predict(incoming_json_1, harmonized_schema_1)
    if transformed_json_1:
        print(f"\nTransformed JSON (Example 1):\n{transformed_json_1}")
    else:
        print("No response received for Example 1.")

    # Example 2 from the prompt
    incoming_json_2 = {
        "dob": 20231027,
        "name": "John",
        "surname": "Doe",
        "address_street": "123 Main St",
        "adress_city": "Anytown",
        "phone_number": "123-456-7890"
    }
    harmonized_schema_2 = {
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
    }

    print("\nIncoming JSON (Example 2):")
    print(json.dumps(incoming_json_2, indent=2))
    print("\nTarget Harmonized Schema (Example 2):")
    print(json.dumps(harmonized_schema_2, indent=2))

    transformed_json_2 = ollama_model.predict(incoming_json_2, harmonized_schema_2)
    if transformed_json_2:
        print(f"\nTransformed JSON (Example 2):\n{transformed_json_2}")
    else:
        print("No response received for Example 2.")

    print("\n--- End of dummy transformation example ---\n")

    while True:
        user_input_command = input("\nEnter command ('exit', 'reset', 'stats') or provide JSON for manual test:\n> ").strip()

        if user_input_command.lower() == 'exit':
            break
        elif user_input_command.lower() == 'reset':
            ollama_model.reset_conversation()
        elif user_input_command.lower() == 'stats':
            ollama_model.get_stats()
        else:
            # For manual testing, you'd need to provide both JSON and schema
            print("For manual testing, please provide the incoming JSON and harmonized schema in code.")
            print("Example usage: transformed_data = ollama_model.predict(your_incoming_json, your_harmonized_schema)")
            # As the script is run in a non-interactive shell, this part is for demonstration.
            # In a real interactive session, you might parse this input further.
