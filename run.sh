# Test different models

python main.py --model_name gemma3:27b
python main.py --model_name deepseek-r1:32b

# Test different temperature

python main.py --model_name gemma3:27b --experiment_name temp_0.5 --temperature 0.5
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.5 --temperature 0.5
