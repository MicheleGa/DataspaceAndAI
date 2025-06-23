python data_visualization.py \
    --input_data ./gemma3_20250623_141934/initial_similarity_results.csv \
    --plot_different_llms False \
    --save_path ./ \
    --output_file_name initial_dbs_structural_vs_semantic_similarity.png 

python data_visualization.py \
    --input_data ./gemma3_20250623_141934/initial_similarity_results.csv ./gemma3_20250623_141934/final_similarity_results.csv ./gemma3:27b_20250623_142051/final_similarity_results.csv \
    --plot_different_llms True \
    --plot_llms_heatmaps True \
    --save_path ./ \
    --output_file_name semantic_similarity_heatmap.png 