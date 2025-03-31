import ollama


class OLLAMA:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, data):
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': data,
                    },
                ]
            )
            return response['message']['content']
        except ollama.APIError as e:
            print(f"Error calling Ollama API: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

            
if __name__ == "__main__":
    
    ollama_model = OLLAMA(model_name='qwen2.5')
    user_input = "Write a short poem about spring."
    prediction = ollama_model.predict(user_input)
    
    if prediction:
        print(f"Ollama response:\n{prediction}")