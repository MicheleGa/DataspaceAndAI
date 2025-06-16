import ollama
import json
import re

# This is our persistent system instruction for the OLLAMA.
SYSTEM_PROMPT = """
You are an intelligent agent designed to analyze and harmonize incoming JSON messages.
Your core function is to analyze and harmonize incoming JSON messages, focusing on both their structural and semantic properties.

**Harmonization Guidelines:**
1.  **Structural Analysis:**
    * **Nesting Level:** Maintain a consistent and logical nesting depth for related data. If an incoming JSON introduces a new, relevant nesting level, integrate it appropriately. If it presents a shallower or deeper structure for existing data, analyze if it's a simplification, expansion, or conflict, and harmonize towards the most comprehensive and logical structure.
    * **Keys per Level:** Identify the most relevant and comprehensive set of keys at each nesting level. If an incoming JSON adds new keys or omits existing ones, integrate or prune them based on their semantic relevance and frequency across messages.

2.  **Semantic Analysis:**
    * **Key Meaning:** Understand the semantic meaning of each key. If different keys represent the same underlying concept (e.g., "product_id" and "item_id"), harmonize them to a single, consistent key name (e.g., "id").
    * **Value Consistency:** Ensure consistency in data types and formats for values associated with harmonized keys. For example, if "price" is sometimes a string and sometimes a float, harmonize to a float.

**Harmonization Process:**
* If it's the first message in the stream, the incoming JSON defines the initial harmonized schema.
* For subsequent messages, you must integrate the new JSON content into the existing harmonized schema. This involves:
    * Adding new, relevant keys and nested structures.
    * Resolving structural conflicts (nesting, key counts) to maintain a logical and comprehensive schema.
    * Resolving semantic conflicts (key names, value types) by identifying synonyms and standardizing terminology.
    * Retaining important information while eliminating redundancy.

**CRITICAL: Your response MUST be a valid JSON object only. Do not include any explanatory text, markdown formatting, or additional commentary. Just return the harmonized JSON schema.**

If you cannot process the JSON or harmonize it according to these guidelines, respond with: {"error": "description of the issue"}
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

    def _extract_json_from_text(self, text):
        """
        Try to extract JSON from text that might contain markdown or other formatting.
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        
        # Try to find JSON-like structure
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found in matches, try parsing the entire text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    def _validate_and_clean_response(self, response_text):
        """
        Validate and clean the model response to ensure it's valid JSON.
        """
        # Try direct JSON parsing first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from formatted text
        extracted_json = self._extract_json_from_text(response_text)
        if extracted_json is not None:
            return extracted_json
        
        # If all else fails, return None
        return None

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
                model_response_json = self._validate_and_clean_response(full_response_content)
                
                if model_response_json is not None:
                    # Update the harmonized schema if the model provided a valid JSON
                    if 'error' not in model_response_json:
                        self.harmonized_schema = model_response_json
                    
                    # Add the model's response to the conversation history
                    self.messages.append({'role': 'assistant', 'content': json.dumps(model_response_json, indent=2)})
                    self.conversation_count += 1
                    
                    return json.dumps(model_response_json, indent=2)
                else:
                    # Handle invalid JSON response
                    print(f"Warning: Model did not return valid JSON. Raw response: {full_response_content}")
                    
                    # Retry logic
                    if retry_count < self.max_retries:
                        print(f"Retrying... (attempt {retry_count + 1}/{self.max_retries})")
                        # Remove the last user message before retrying
                        self.messages.pop()
                        return self.predict(user_data, retry_count + 1)
                    else:
                        # If all retries failed, return error and reset conversation context
                        error_message = {
                            "error": f"Model failed to return valid JSON after {self.max_retries} attempts. Raw response: {full_response_content}"
                        }
                        # Reset conversation to avoid context pollution
                        self.messages = [{'role': 'system', 'content': self.system_prompt}]
                        return json.dumps(error_message, indent=2)

            except Exception as ollama_error:
                print(f"Ollama API error: {ollama_error}")
                error_message = {"error": f"Ollama API error: {str(ollama_error)}"}
                return json.dumps(error_message, indent=2)

        except Exception as e:
            print(f"Unexpected error: {e}")
            error_message = {"error": f"Unexpected error: {str(e)}"}
            return json.dumps(error_message, indent=2)

    def reset_conversation(self):
        """Reset the conversation and harmonized schema."""
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.harmonized_schema = {}
        self.conversation_count = 0
        print("Conversation reset.")

    def get_conversation_stats(self):
        """Get statistics about the current conversation."""
        return {
            "conversation_count": self.conversation_count,
            "message_count": len(self.messages),
            "current_schema_keys": list(self.harmonized_schema.keys()) if self.harmonized_schema else [],
            "schema_depth": self._get_json_depth(self.harmonized_schema) if self.harmonized_schema else 0
        }
    
    def _get_json_depth(self, obj, depth=0):
        """Calculate the maximum depth of a JSON object."""
        if not isinstance(obj, dict):
            return depth
        return max([self._get_json_depth(v, depth + 1) for v in obj.values()] + [depth])

if __name__ == "__main__":
    
    ollama_model = OLLAMA(model_name='mistral')

    print("JSON Harmonization Agent initialized. Provide JSON input.")
    print("Commands: 'exit' to quit, 'reset' to reset conversation, 'stats' to show stats")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou (JSON or command): ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'reset':
            ollama_model.reset_conversation()
            continue
        elif user_input.lower() == 'stats':
            stats = ollama_model.get_conversation_stats()
            print(f"Conversation Stats: {json.dumps(stats, indent=2)}")
            continue
        
        if not user_input:
            print("Please provide some input.")
            continue
        
        prediction = ollama_model.predict(user_input)
        
        if prediction:
            print(f"\nAgent response:\n{prediction}")
        else:
            print("No response received from the model.")