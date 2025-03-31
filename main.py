import os
import argparse

from sentence_transformers import SentenceTransformer

from model.interface import OLLAMA
from utils.helpers import load_json, count_keys_per_level, get_json_embedding
from utils.metrics import calculate_structure_similarity, calculate_semantic_similarity


def main(args):
    

    # Load json from the dataset folder
    first_json_message = load_json(path=os.path.join(args.dataset_folder, 'first_json.json'))
    second_json_message = load_json(path=os.path.join(args.dataset_folder, 'second_json.json'))
    third_json_message = load_json(path=os.path.join(args.dataset_folder, 'third_json.json'))
    
    # Structural similarity (based on the number of keys per nesting level)    
    dict1 = count_keys_per_level(first_json_message)
    dict2 = count_keys_per_level(second_json_message)
    dict3 = count_keys_per_level(third_json_message)

    similarity1_2 = calculate_structure_similarity(dict1, dict2)
    similarity1_3 = calculate_structure_similarity(dict1, dict3)
    similarity2_3 = calculate_structure_similarity(dict2, dict3)

    print(f"Key Structure Similarity between JSON 1 and JSON 2: {similarity1_2:.4f}")
    print(f"Key Structure Similarity between JSON 1 and JSON 3: {similarity1_3:.4f}")
    print(f"Key Structure Similarity between JSON 2 and JSON 3: {similarity2_3:.4f}")
    
    # Semantic similarity (based on JSON as a bag of words)
    model = SentenceTransformer(args.embedding_model_name)
    
    embedding1 = get_json_embedding(first_json_message, model)
    embedding2 = get_json_embedding(second_json_message, model)
    embedding3 = get_json_embedding(third_json_message, model)

    similarity1_2 = calculate_semantic_similarity(embedding1, embedding2)
    similarity1_3 = calculate_semantic_similarity(embedding1, embedding3)
    similarity2_3 = calculate_semantic_similarity(embedding2, embedding3)

    print(f"Semantic Similarity between JSON 1 and JSON 2: {similarity1_2:.4f}")
    print(f"Semantic Similarity between JSON 1 and JSON 3: {similarity1_3:.4f}")
    print(f"Semantic Similarity between JSON 2 and JSON 3: {similarity2_3:.4f}")

    # Instantiate the interface that will call the OLLAMA model running in local (some port on local host 127.0.0.1)
    model = OLLAMA(model_name=args.model_name)
    
    # Build the prompt
    prompt = f'Given the following JSON message {first_json_message} and the second one {second_json_message}, can you generate an harmonized JSON schema out of them?'
    
    # Call the AI agent
    response = model.predict(data=prompt)
    
    # Print the response if present
    if response:
        print(response)
    else:
        print(f'No repsonse provided by {args.model_name}')
    

def parseargs():
    parser = argparse.ArgumentParser(description="Dataspace and AI: Schema Harmonization")
    
    parser.add_argument('--dataset_folder', default='./data', type=str, help='path to the dataset folder')
    parser.add_argument('--model_name', default='qwen2.5', type=str, help='LLM to adpot for the schema harmonization')
    parser.add_argument('--embedding_model_name', default='all-MiniLM-L6-v2', type=str, help='model to employ to produce JSON embeddings')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    global args
    args = parseargs()
    
    main(args)