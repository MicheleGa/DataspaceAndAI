import os
import argparse
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

from model.interface import OLLAMA
from utils.helpers import count_keys_per_level, get_json_embedding, generate_run_name
from utils.metrics import calculate_structure_similarity, calculate_semantic_similarity


def main(args):
    
    # Run name generation
    run_name = generate_run_name(args)
    
    # Make fodlers for figs and data
    if not os.path.exists(os.path.join(args.results_folder, run_name)):
        os.makedirs(os.path.join(args.results_folder, run_name))
    
    
    ## ---- JSON Data Loading ----
    # List of JSON file paths to process
    json_file_names = os.listdir(args.dataset_folder)
    original_json_messages = [] # This will store dictionaries

    # Load JSONs from the dataset folder based on the list of file names
    print("Loading JSON messages...")
    for file_name in json_file_names:
        json_path = os.path.join(args.dataset_folder, file_name)
        with open(json_path, 'r') as f:
            message = json.load(f)
        original_json_messages.append(message)
        print(f"Loaded {file_name}")
    
    num_jsons = len(original_json_messages)
    embedding_model = SentenceTransformer(args.embedding_model_name)
    
    
    
    ## ---- Initial Structure/Semantic Similarity Analysis ----
    print("\n## ---- Initial Structure/Semantic Similarity Analysis (Pairwise) ----")
    
    # Pass dictionaries directly to count_keys_per_level
    original_dicts_per_level = [count_keys_per_level(msg) for msg in original_json_messages] # MODIFIED: pass dict directly
    
    # Assuming get_json_embedding can also handle a dictionary by converting it internally
    original_embeddings = [get_json_embedding(msg, embedding_model) for msg in original_json_messages] 

    # Save results to a dict
    initial_similarities = {
        'structure': [],
        'semantic': [],
        'x': [],
        'y': []
    }
    for i in range(num_jsons):
        for j in range(i + 1, num_jsons):
            # Structural similarity
            structure_sim = calculate_structure_similarity(original_dicts_per_level[i], original_dicts_per_level[j])
            print(f"Key Structure Similarity between JSON {i+1} ({json_file_names[i]}) and JSON {j+1} ({json_file_names[j]}): {structure_sim:.4f}")

            # Semantic similarity
            semantic_sim = calculate_semantic_similarity(original_embeddings[i], original_embeddings[j])
            print(f"Semantic Similarity between JSON {i+1} ({json_file_names[i]}) and JSON {j+1} ({json_file_names[j]}): {semantic_sim:.4f}")
            
            # Store the results
            initial_similarities['structure'].append(structure_sim)
            initial_similarities['semantic'].append(semantic_sim)
            initial_similarities['x'].append(json_file_names[i])
            initial_similarities['y'].append(json_file_names[j])

    # Create a DataFrame for results visualization and analysis
    print("\nSaving initial similarity results to a DataFrame...")
    df = pd.DataFrame(initial_similarities)
    df.to_csv(os.path.join(args.results_folder, run_name, f'initial_similarity_results.csv'), index=False)
    
    
    ## ---- AI Agent JSON Harmonization ----
    print("\n## ---- AI Agent JSON Harmonization ----")
    ollama_agent = OLLAMA(model_name=args.model_name)

    harmonized_response_str = None # This will store the final harmonized JSON string received from OLLAMA
    
    # Loop through all JSON messages sequentially for harmonization
    for i, current_json_message_dict in enumerate(original_json_messages): # current_json_message_dict is a dict
        print(f"Processing JSON message {i+1} ({json_file_names[i]})...")
        # Call the AI agent with the current JSON message, which must be a JSON string
        response_from_ollama = ollama_agent.predict(user_data=json.dumps(current_json_message_dict)) # OLLAMA expects a JSON string
        
        # Print the response for each step
        if response_from_ollama:
            harmonized_response_str = response_from_ollama 
            
            # Dump the intermediate harmonized response to a JSON file
            harmonized_json_path = os.path.join(args.results_folder, run_name, f'harmonized_schema_{i+1}.json')

            # COnvert to dict for saving
            try:
                harmonized_response_dict = json.loads(harmonized_response_str)
            except json.JSONDecodeError:
                print(f"Error: Response from {args.model_name} is not a valid JSON. Skipping saving for JSON message {i+1}.")
                exit()

            with open(harmonized_json_path, 'w') as f:
                json.dump(harmonized_response_dict, f, indent=2)
            print(f"Harmonized schema saved to {harmonized_json_path}")
        else:
            print(f'No valid response provided by {args.model_name} for JSON message {i+1}.')
            
            
    
    ## ---- Final Harmonized Schema Structure/Semantic Similarity Analysis ----
    # After the loop, if we have a harmonized_response_str, perform the final similarity analysis
    if harmonized_response_str:
        print("\n## ---- Final Harmonized Schema Structure/Semantic Similarity Analysis ----")
        
        # Parse the final harmonized response string back to a dict for structural analysis
        try:
            final_harmonized_dict = json.loads(harmonized_response_str)
        except json.JSONDecodeError:
            print("Error: Final harmonized response is not a valid JSON. Cannot perform similarity analysis.")
            exit()

        # Save JSON
        final_harmonized_json_path = os.path.join(args.results_folder, run_name, 'final_harmonized_schema.json')
        with open(final_harmonized_json_path, 'w') as f:
            json.dump(final_harmonized_dict, f, indent=2)
            
        # Pass the dictionary directly to count_keys_per_level
        dict_gen = count_keys_per_level(final_harmonized_dict) # MODIFIED: pass dict here
        
        # Pass the dictionary to get_json_embedding as well
        embedding_gen = get_json_embedding(final_harmonized_dict, embedding_model) # MODIFIED: pass dict here
        
        # Save results to a dict
        final_similarities = {
            'structure': [],
            'semantic': [],
            'x': [],
            'y': []
        }
        for i, _ in enumerate(original_json_messages):
            # Structural similarity
            structure_similarity_gen = calculate_structure_similarity(original_dicts_per_level[i], dict_gen)
            print(f"Key Structure Similarity between JSON {i+1} ({json_file_names[i]}) and Final Harmonized JSON: {structure_similarity_gen:.4f}")
            
            # Semantic similarity
            semantic_similarity_gen = calculate_semantic_similarity(original_embeddings[i], embedding_gen)
            print(f"Semantic Similarity between JSON {i+1} ({json_file_names[i]}) and Final Harmonized JSON: {semantic_similarity_gen:.4f}")
            
            # Store the results
            final_similarities['structure'].append(structure_similarity_gen)
            final_similarities['semantic'].append(semantic_similarity_gen)
            final_similarities['x'].append(json_file_names[i])
            final_similarities['y'].append('Final Harmonized JSON')
            
        # Create a DataFrame for final similarity results and analysis
        print("\nSaving final similarity results to a DataFrame...")
        final_df = pd.DataFrame(final_similarities)
        final_df.to_csv(os.path.join(args.results_folder, run_name, f'final_similarity_results.csv'), index=False)
        print("Final similarity results saved to final_similarity_results.csv in the figures folder.")
    
    else:
        print("No valid harmonized response generated for final similarity analysis.")

def parseargs():
    parser = argparse.ArgumentParser(description="Dataspace and AI: Schema Harmonization")
    
    parser.add_argument('--dataset_folder', default='./data/json_data', type=str, help='path to the dataset folder')
    parser.add_argument('--results_folder', default='./results', type=str, help='path to the figures folder')
    parser.add_argument('--experiment_name', default='', type=str, help='name for the experiment')
    parser.add_argument('--model_name', default='llama3.2', type=str, help='LLM to adpot for the schema harmonization')
    parser.add_argument('--embedding_model_name', default='all-MiniLM-L6-v2', type=str, help='model to employ to produce JSON embeddings')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parseargs()
    
    main(args)