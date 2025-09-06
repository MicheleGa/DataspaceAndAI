import os
import sys
sys.path.append('../') # to import utils and model folders
import json
import argparse
import pprint
from itertools import combinations
from utils.helpers import generate_run_name
from model.evaluation_interface import OLLAMAEvaluator


def main(args):
    
    # Run name generation
    run_name = generate_run_name(args)
    
    ## ---- JSON Data Loading ----
    print("Loading JSON messages from dataset folder...")
    
    json_file_names = [f for f in os.listdir(args.dataset_folder) if f.endswith(".json")]
    
    if not json_file_names:
        raise ValueError(f"No JSON files found in dataset folder: {args.dataset_folder}")
    
    json_messages = {}
    for file_name in json_file_names:
        json_path = os.path.join(args.dataset_folder, file_name)
        with open(json_path, 'r') as f:
            message = json.load(f)
        json_messages[file_name] = message
        print(f"Loaded {file_name}")
    
    ## ---- Harmonization Evaluator ----
    evaluator = OLLAMAEvaluator(
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    ## ---- Pairwise Comparisons ----
    print("\nRunning pairwise harmonization evaluations...")
    results_folder = os.path.join(args.dataset_folder, 'agent_evaluation', run_name)
    os.makedirs(results_folder, exist_ok=True)
    
    results = {}
    for file1, file2 in combinations(json_file_names, 2):  # unique pairs
        print(f"\nComparing {file1} <-> {file2}")
        
        original_json_str = json.dumps(json_messages[file1], indent=2)
        harmonized_json_str = json.dumps(json_messages[file2], indent=2)
        
        result = evaluator.evaluate(original_json_str, harmonized_json_str)
        
        # Store results under pair key
        pair_key = f"{file1}_VS_{file2}"
        results[pair_key] = json.loads(result)
        
        # Save each pairwise evaluation separately
        pair_result_path = os.path.join(results_folder, f"{pair_key}.json")
        with open(pair_result_path, "w") as f:
            f.write(result)
        print(f"Saved evaluation result to {pair_result_path}")
    
    ## ---- Save summary ----
    summary_path = os.path.join(results_folder, f"{run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary of all evaluations saved to {summary_path}")


def parseargs():
    parser = argparse.ArgumentParser(description="Dataspace and AI: Schema Harmonization Evaluation")
    
    parser.add_argument('--dataset_folder', default='./', type=str, help='path to the folder with the JSONs to evaluate')
    parser.add_argument('--experiment_name', default='', type=str, help='name for the experiment')
    parser.add_argument('--model_name', default='gemma3', type=str, help='LLM to adopt for the schema harmonization evaluation')
    parser.add_argument('--temperature', default=0.7, type=float, help='temperature for the OLLAMA model (regulates creativity)')
    parser.add_argument('--top_p', default=0.95, type=float, help='probability that regulates the ratio of tokens the OLLAMA model chooses for generating the response')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":    
    args = parseargs()
    
    # Pretty print the parsed arguments
    print("Run Configuration:")
    pprint.pprint(vars(args))
    print()
    
    main(args)
