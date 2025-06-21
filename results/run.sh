python data_visualization.py \
    --input_data ./gemma3_20250621_180152/initial_similarity_results.csv \
    --plot_different_llms False \
    --save_path ./ \
    --output_file_name initial_dbs_structural_vs_semantic_similarity.png 

python data_visualization.py \
    --input_data ./gemma3_20250621_180152/final_similarity_results.csv ./gemma3:27b_20250621_180336/final_similarity_results.csv ./qwen3:32b_20250621_190806/final_similarity_results.csv \
    --plot_different_llms True \
    --save_path ./ \
    --output_file_name harmonized_schema_semantic_similarity.png 