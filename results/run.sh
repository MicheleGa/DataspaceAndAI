# Plot initial similarity without harmonization

python data_visualization.py \
    --input_data ./gemma3:27b_20250624_182334/initial_similarity_results.csv \
    --plot_different_llms False \
    --save_path ./figs \
    --output_file_name initial_dbs_structure_vs_semantic_similarity.png 

# Plot LLM harmonization results with temperature 0.7

python data_visualization.py \
    --input_data ./gemma3:27b_20250624_182334/initial_similarity_results.csv ./deepseek-r1:32b_20250624_194045/final_similarity_results.csv ./gemma3:27b_20250624_182334/final_similarity_results.csv \
    --plot_different_llms True \
    --plot_llms_heatmaps True \
    --save_path ./figs \
    --output_file_name semantic_similarity.png 

python data_visualization.py \
    --input_data ./gemma3:27b_20250624_182334/initial_similarity_results.csv ./deepseek-r1:32b_20250624_194045/final_similarity_results.csv ./gemma3:27b_20250624_182334/final_similarity_results.csv \
    --plot_different_llms True \
    --plot_llms_heatmaps True \
    --save_path ./figs \
    --output_file_name structure_similarity.png \
    --similarity_type structure

# Plot LLM harmonization results with temperature 0.5

python data_visualization.py \
    --input_data ./gemma3:27b_20250624_182334/initial_similarity_results.csv ./deepseek-r1:32b_temp_0.5_20250624_211754/final_similarity_results.csv ./gemma3:27b_temp_0.5_20250624_200137/final_similarity_results.csv \
    --plot_different_llms True \
    --plot_llms_heatmaps True \
    --save_path ./figs \
    --output_file_name temp_0.5_semantic_similarity.png 

python data_visualization.py \
    --input_data ./gemma3:27b_20250624_182334/initial_similarity_results.csv ./deepseek-r1:32b_temp_0.5_20250624_211754/final_similarity_results.csv ./gemma3:27b_temp_0.5_20250624_200137/final_similarity_results.csv \
    --plot_different_llms True \
    --plot_llms_heatmaps True \
    --save_path ./figs \
    --output_file_name temp_0.5_structure_similarity.png \
    --similarity_type structure