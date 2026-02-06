# N.B.: figs folder (save_path folder, this case `quantitative_results`) must be present in the current directory (or just `mkdir figs` if figs is not present)
# Plot initial similarity without harmonization

python data_visualization.py \
    --input_data ./gemma3:27b_temp_0.0_20260205_160423/initial_similarity_results.csv \
   --plot_different_llms False \
    --save_path ./quantitative_results \
    --output_file_name initial_dbs_structure_vs_semantic_similarity.png 

## Plot LLM harmonization results (N.B.: always check the apth and experiment name)

python data_visualization.py \
    --input_data ./gemma3:27b_temp_0.0_20260205_160423/initial_similarity_results.csv ./gemma3:27b_temp_0.0_20260205_160423/final_similarity_results.csv ./deepseek-r1:32b_temp_0.0_20260205_114606/final_similarity_results.csv \
    --plot_different_llms True \
    --plot_llms_heatmaps True \
    --save_path ./quantitative_results \
    --output_file_name semantic_similarity.png 

python data_visualization.py \
    --input_data ./gemma3:27b_temp_0.0_20260205_160423/initial_similarity_results.csv ./gemma3:27b_temp_0.0_20260205_160423/final_similarity_results.csv ./deepseek-r1:32b_temp_0.0_20260205_114606/final_similarity_results.csv \
    --plot_different_llms True \
    --plot_llms_heatmaps True \
    --save_path ./quantitative_results \
    --output_file_name structure_similarity.png \
    --similarity_type structure

# LLM-based evaluation between 2 JSONs messages

# NOTE: use the same model adopted for the harmonizer/transformer also during evaluationw tih LLMs
# -> available models: gemma3:27b, deepseek-r1:32b
# -> check the temperature first
python agent_evaluator.py \
    --dataset_folder ./initial_jsons_evaluation \
    --experiment_name qualitative_evaluation \
    --model_name gemma3:27b \
    --temperature 0.0

python agent_evaluator.py \
    --dataset_folder ./gemma3:27b_temp_0.0_20260205_160423 \
    --experiment_name qualitative_evaluation \
    --model_name gemma3:27b \
    --temperature 0.0