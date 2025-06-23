import os
import argparse
import json
import re
import pprint
import pandas as pd
from sentence_transformers import SentenceTransformer
from model.harmonization_interface import OLLAMAHarmonizer
from model.transformation_interface import OLLAMATransformer
from utils.helpers import count_keys_per_level, get_json_embedding, generate_run_name
from utils.metrics import calculate_structure_similarity, calculate_semantic_similarity


def schema_harmonization(run_name, original_json_messages, json_file_names, args):

    num_jsons = len(original_json_messages)
    embedding_model = SentenceTransformer(args.embedding_model_name)


    ## ---- Initial Structure/Semantic Similarity Analysis ----
    print("\n## ---- Initial Structure/Semantic Similarity Analysis (Pairwise) ----")
    
    # Pass dictionaries directly to count_keys_per_level
    original_dicts_per_level = [count_keys_per_level(msg) for msg in original_json_messages]
    
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
            print(f"Key Structure Similarity between JSON {i+1} ({json_file_names[i]}) and JSON {j+1} ({json_file_names[j]}): {structure_sim:.2f}")

            # Semantic similarity
            semantic_sim = calculate_semantic_similarity(original_embeddings[i], original_embeddings[j])
            print(f"Semantic Similarity between JSON {i+1} ({json_file_names[i]}) and JSON {j+1} ({json_file_names[j]}): {semantic_sim:.2f}")
            
            # Store the results
            initial_similarities['structure'].append(structure_sim)
            initial_similarities['semantic'].append(semantic_sim)
            initial_similarities['x'].append(json_file_names[i])
            initial_similarities['y'].append(json_file_names[j])

    # Create a DataFrame for results visualization and analysis
    print(f"\nSaving initial similarity results to {os.path.join(args.results_folder, run_name, f'initial_similarity_results.csv')}...")
    df = pd.DataFrame(initial_similarities)
    df.to_csv(os.path.join(args.results_folder, run_name, f'initial_similarity_results.csv'), index=False)
    
    
    ## ---- AI Agent JSON Harmonization ----
    print("\n## ---- AI Agent JSON Harmonization ----")
    ollama_agent = OLLAMAHarmonizer(model_name=args.model_name, temperature=args.temperature, top_p=args.top_p)

    harmonized_response_str = None # This will store the final harmonized JSON string received from OLLAMAHarmonizer
    
    # Loop through all JSON messages sequentially for harmonization
    for i, current_json_message_dict in enumerate(original_json_messages):
        print(f"Processing JSON message {i+1} ({json_file_names[i]})...")
        # Call the AI agent with the current JSON message, which must be a JSON string
        response_from_ollama = ollama_agent.predict(user_data=json.dumps(current_json_message_dict)) # OLLAMA expects a JSON string
        
        # Print the response for each step
        if response_from_ollama:
            harmonized_response_str = response_from_ollama 
            
            # Dump the intermediate harmonized response to a JSON file
            harmonized_json_path = os.path.join(args.results_folder, run_name, f'harmonized_schema_{i+1}.json')

            # Convert to dict for saving
            try:
                harmonized_response_dict = json.loads(harmonized_response_str)
            except json.JSONDecodeError:
                print(f"Error: Response from {args.model_name} is not a valid JSON. Skipping saving for JSON message {i+1}.")
                # Changed from exit() to continue for robustness
                continue 

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
            # Changed from exit() to return for robustness
            return

        # Save JSON
        final_harmonized_json_path = os.path.join(args.results_folder, run_name, 'final_harmonized_schema.json')
        with open(final_harmonized_json_path, 'w') as f:
            json.dump(final_harmonized_dict, f, indent=2)
            
        # Pass the dictionary directly to count_keys_per_level
        dict_gen = count_keys_per_level(final_harmonized_dict)
        
        # Pass the dictionary to get_json_embedding as well
        embedding_gen = get_json_embedding(final_harmonized_dict, embedding_model)
        
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
            print(f"Key Structure Similarity between JSON {i+1} ({json_file_names[i]}) and Final Harmonized JSON: {structure_similarity_gen:.2f}")
            
            # Semantic similarity
            semantic_similarity_gen = calculate_semantic_similarity(original_embeddings[i], embedding_gen)
            print(f"Semantic Similarity between JSON {i+1} ({json_file_names[i]}) and Final Harmonized JSON: {semantic_similarity_gen:.2f}")
            
            # Store the results
            final_similarities['structure'].append(structure_similarity_gen)
            final_similarities['semantic'].append(semantic_similarity_gen)
            final_similarities['x'].append(json_file_names[i])
            final_similarities['y'].append('final_harmonized_schema')
            
        # Create a DataFrame for final similarity results and analysis
        print("\nSaving final similarity results to a DataFrame...")
        final_df = pd.DataFrame(final_similarities)
        final_df.to_csv(os.path.join(args.results_folder, run_name, f'final_similarity_results.csv'), index=False)
        print(f"Final similarity results saved to {os.path.join(args.results_folder, run_name, f'final_similarity_results.csv')}.")
    
    else:
        print("No valid harmonized response generated for final similarity analysis.")


def messages_transformation(run_name, original_json_messages, json_file_names, args):
    
    # Load harmonized schema
    final_harmonized_schema_path = os.path.join(args.results_folder, run_name, 'final_harmonized_schema.json')

    try:
        with open(final_harmonized_schema_path, 'r') as f:
                final_harmonized_schema = json.load(f)
    except FileNotFoundError:
        print(f"Error: Harmonized schema file not found at {final_harmonized_schema_path}. "
              "Please ensure schema_harmonization step has been run successfully.")
        return # Exit the function if schema isn't found
    except json.JSONDecodeError:
        print(f"Error: Harmonized schema file at {final_harmonized_schema_path} is not a valid JSON.")
        return # Exit the function if schema is invalid


    ## ---- AI Agent JSON Message Transformation ----
    print("\n## ---- AI Agent JSON Message Transformation ----")
    ollama_agent = OLLAMATransformer(model_name=args.model_name, temperature=args.temperature, top_p=args.top_p)

    # No need for transofrmed_json_response_str to store only the last response
    # It's better to process and save each one individually.
    
    # Loop through all JSON messages sequentially for transformation
    for i, current_json_message_dict in enumerate(original_json_messages):
        print(f"Transforming JSON message {i+1} ({json_file_names[i]})...")
        
        # Corrected: Pass dict directly to incoming_json, as OLLAMATransformer expects a dict
        response_from_ollama = ollama_agent.predict(
            incoming_json=current_json_message_dict,
            target_harmonized_schema=final_harmonized_schema
        )
         
        # Process the response
        if response_from_ollama:
            # Dump the transformed JSON to a file
            transformed_json_path = os.path.join(args.results_folder, run_name, f'{json_file_names[i]}_transformed.json') # Renamed file for clarity

            # Convert to dict for saving
            try:
                transformed_response_dict = json.loads(response_from_ollama)
            except json.JSONDecodeError:
                print(f"Error: Response from {args.model_name} is not a valid JSON for {json_file_names[i]}. Skipping saving.")
                continue # Continue to the next message instead of exiting

            with open(transformed_json_path, 'w') as f:
                json.dump(transformed_response_dict, f, indent=2)
            print(f"Transformed JSON saved to {transformed_json_path}")
        else:
            print(f'No valid response provided by {args.model_name} for JSON message {i+1} during transformation.')
            
    
    
def main(args):
    
    # Run name generation
    run_name = generate_run_name(args)
    
    # Make folders for figs and data
    if not os.path.exists(os.path.join(args.results_folder, run_name)):
        os.makedirs(os.path.join(args.results_folder, run_name))
    
    
    ## ---- JSON Data Loading ----
    # List of JSON file paths to process
    json_file_names = os.listdir(args.dataset_folder)
    
    # Sort by the leading number before the underscore (see the readme file in the data folder)
    json_file_names = sorted(
        json_file_names,
        key=lambda x: int(re.match(r'^(\d+)_', x).group(1))
    )
    
    original_json_messages = [] # This will store dictionaries

    # Load JSONs from the dataset folder based on the list of file names
    print("Loading JSON messages...")
    for file_name in json_file_names:
        json_path = os.path.join(args.dataset_folder, file_name)
        with open(json_path, 'r') as f:
            message = json.load(f)
        original_json_messages.append(message)
        print(f"Loaded {file_name}")
    
    ## Step 1 - Schema Harmonization
    schema_harmonization(run_name, original_json_messages, json_file_names, args) # Uncommented to ensure schema exists
    
    ## Step 2 - JSON messages transformation
    messages_transformation(run_name, original_json_messages, json_file_names, args)
    
    
    
def parseargs():
    parser = argparse.ArgumentParser(description="Dataspace and AI: Schema Harmonization")
    
    parser.add_argument('--dataset_folder', default='./data/json_data', type=str, help='path to the dataset folder')
    parser.add_argument('--results_folder', default='./results', type=str, help='path to the figures folder')
    parser.add_argument('--experiment_name', default='', type=str, help='name for the experiment')
    parser.add_argument('--model_name', default='llama3.2', type=str, help='LLM to adpot for the schema harmonization')
    parser.add_argument('--embedding_model_name', default='all-MiniLM-L6-v2', type=str, help='model to employ to produce JSON embeddings')
    parser.add_argument('--temperature', default=0.7, type=float, help='temperature for the OLLAMA model (regulate creativity)')
    parser.add_argument('--top_p', default=0.95, type=float, help='probability that regulates the ratio of tokens the OLLAMA model choose for generating the response')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parseargs()
    
    # Pretty print the parsed arguments
    print("Run Configuration:")
    pprint.pprint(vars(args))
    print()
    
    main(args)
