import ollama
import json

EVALUATION_SYSTEM_PROMPT = """
# Identity
    - You are an expert evaluator of semantic interoperability in JSON schemas.
    - Your purpose is to compare two JSON messages coming from two different data sources.
    - You qualitatively assess their structural and semantic similarity and explain why.

# Input
    - First JSON message.
    - Second JSON message.

# Output
    - A JSON object that contains:
        {
            "structural_similarity": "High | Low",
            "semantic_similarity": "High | Low",
            "verdict": "Qualitative assessment and explanation of whether the harmonization process is successful, highlighting differences between JSONs otherwise."
        }
"""

class OLLAMAEvaluator:
    def __init__(self, model_name="gemma3", temperature=0.2, top_p=0.9):
        self.model_name = model_name
        self.system_prompt = EVALUATION_SYSTEM_PROMPT
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.temperature = temperature
        self.top_p = top_p

    def evaluate(self, original_json_str, harmonized_json_str):
        try:
            # Validate input JSONs
            try:
                original_json = json.loads(original_json_str)
                harmonized_json = json.loads(harmonized_json_str)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON input(s)."}, indent=2)
            
            user_message_content = """
            # Instructions
            
                - To understand the structural similarity between two JSONs you have to examine how data is organized across different nesting levels (level 0 = root, level 1 = first nested level, etc.):
                    - Counts two key components at each level: number of keys and number of nested blocks.
                    - High structural similarity means nearly identical structure: same number of keys and nested elements at every level.
                    - Report where there are differences in structure, such as one JSON having more keys or deeper nesting at any level.
            
                - To understand the semnatic similarity between two JSONs you have to measure the content meaning similarity between the keys of the two JSON messages:
                    - Focus only on the keys and ignore the values as they carry the actual meaning of the JSON content.
                    - Do not care about the JSON structure while defining the semantic similarity, focus purely on what the content means rather than how it's organized.
                    - High semantic similarity means identical semantic meaning between the keys of one of the two JSONs and the keys (or a subset of keys in case of structural differences) of the other JSON.
                    - Report on key differences in meaning, such as one JSON having keys that are absent or semantically different in the other JSON.
                
                - Semantic similarity is the most important criteria to assess the success of the harmonization process.
                - Structural similarity is a secondary criteria, useful to understand if the harmonization process also affected the JSON structure.

            # Examples
            
                - Example 1:
                    
                    [{'role': 'user', 'content': 'Compare the following JSON messages:

                            First JSON:{
                                "date_of_birth": "2023-10-27"
                            }

                            Second JSON:{
                                "date_of_birth": "STRING"
                            }

                            Provide an evaluation according to the Output format.'},
                    {'role': 'assistant', 'content': '
                    
                            "structural_similarity": "High",
                            "semantic_similarity": "High",
                            "verdict": "The two JSONs have high semantic similarity since the keys of the first JSON corrspond to those of the second one and viceversa. The two JSONs have also the same structural similarity since they have the same number of keys per levels. Overall, the harmonization process was successful"
                        }'}] 
                
                - Example 2:
                
                    [{'role': 'user', 'content': 'Compare the following JSON messages:

                            First JSON:{
                                "dob": "2023-10-27"
                            }

                            Second JSON:{
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
                            }

                            Provide an evaluation according to the Output format.'},
                    {'role': 'assistant', 'content': '
                    
                            "structural_similarity": "Low",
                            "semantic_similarity": "High",
                            "verdict": "The two JSONs have high semantic similarity since the meaning of the single key 'dob' of the first JSON is present also in the second one 'date_of_birth'. The two JSONs have different structural similarity since the second JSON has more keys and more nested blocks per levels than the first one. Overall, the harmonization process was successful"
                        }'}] 
                    
                - Example 3:
                
                    [{'role': 'user', 'content': 'Compare the following JSON messages:

                            First JSON:{
                                "name": {
                                    "first": "John",
                                    "last": "Doe"
                                    },
                                "address": {
                                    "street": "n/a",
                                    "city": "n/a"
                                    },
                            }

                            Second JSON:{
                                "first_name": "n/a",
                                "last_name": "n/a",
                                "street": "n/a",
                                "city": "n/a",
                                "date_of_birth": "2023-10-27"
                            }

                            Provide an evaluation according to the Output format.'},
                    {'role': 'assistant', 'content': '
                    
                            "structural_similarity": "Low",
                            "semantic_similarity": "Low",
                            "verdict": "The two JSONs have similar semantic similarity but the second JSON reports 'date_of_birth' which is not present in the other one. The first JSON has has more nested blocks per levels than the second one. Overall, the harmonization process was not successful."
                        }'}] 
                
                - Example 4:
                
                    [{'role': 'user', 'content': 'Compare the following JSON messages:

                            First JSON:{
                                "name": {
                                    "first": "John",
                                    "last": "Doe"
                                    }
                            }

                            Second JSON:{
                                "address": {
                                    "street": "n/a",
                                    "city": "n/a"
                                    }
                            }

                            Provide an evaluation according to the Output format.'},
                    {'role': 'assistant', 'content': '
                    
                            "structural_similarity": "High",
                            "semantic_similarity": "Low",
                            "verdict": "The two JSONs have low semantic similarity because each key in the first JSON has a different meaning than the keys in the second JSON, and vice versa. The structural similarity is high because the two JSONs have the same number of keys and nesting blocks per level. Overall, the harmonization process was not successful."
                        }'}]
            """
            
            user_message_content += (
                    f"\n\nCompare the following JSON messages:"
                    f"\n\nFirst JSON:{json.dumps(original_json, indent=2)}"
                    f"\n\nSecond JSON:{json.dumps(harmonized_json, indent=2)}"
                    f"\n\nProvide an evaluation according to the Output format."
                )

            self.messages.append({'role': 'user', 'content': user_message_content})
            
            response = ollama.chat(
                model=self.model_name,
                messages=self.messages,
                stream=False,
                format='json',
                options={
                    'temperature': self.temperature,
                    'top_p': self.top_p
                }
            )

            full_response_content = response['message']['content']

            try:
                evaluation_json = json.loads(full_response_content)
            except json.JSONDecodeError:
                raise ValueError("Model response is not valid JSON: " + full_response_content)

            return json.dumps(evaluation_json, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)


if __name__ == "__main__":
    evaluator = OLLAMAEvaluator(model_name='gemma3')

    print("JSON Harmonization Evaluator initialized.")
    print("Provide two JSON inputs to compare: original and harmonized.")

    while True:
        original_input = input("\nOriginal JSON (or 'exit'): ").strip()
        if original_input.lower() == 'exit':
            break

        harmonized_input = input("Harmonized JSON: ").strip()

        result = evaluator.evaluate(original_input, harmonized_input)
        print(f"\nEvaluation Result:\n{result}")
