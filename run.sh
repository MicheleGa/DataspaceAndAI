# gemma3 is very useful for debugging as it has a relatively fast executiion time compared to the working models
#python main.py --model_name gemma3

# Working models -> gemma3:27b & deepseek-r1:32b
#python main.py --model_name gemma3:27b --experiment_name temp_0.0 --temperature 0.0
#python main.py --model_name deepseek-r1:32b --experiment_name temp_0.0 --temperature 0.0


# Ablation on smaller vs larger models and different temperature
python main.py --model_name gemma3 --experiment_name temp_0.0 --temperature 0.0
python main.py --model_name gemma3 --experiment_name temp_0.3 --temperature 0.3
python main.py --model_name gemma3 --experiment_name temp_0.5 --temperature 0.5
python main.py --model_name gemma3 --experiment_name temp_0.7 --temperature 0.7
python main.py --model_name gemma3 --experiment_name temp_1.0 --temperature 1.0

python main.py --model_name gemma3:27b --experiment_name temp_0.0 --temperature 0.0
python main.py --model_name gemma3:27b --experiment_name temp_0.3 --temperature 0.3
python main.py --model_name gemma3:27b --experiment_name temp_0.5 --temperature 0.5
python main.py --model_name gemma3:27b --experiment_name temp_0.7 --temperature 0.7
python main.py --model_name gemma3:27b --experiment_name temp_1.0 --temperature 1.0

python main.py --model_name gemma3:27b --experiment_name temp_0.0_gen_1 --temperature 0.0
python main.py --model_name gemma3:27b --experiment_name temp_0.3_gen_1 --temperature 0.3
python main.py --model_name gemma3:27b --experiment_name temp_0.5_gen_1 --temperature 0.5

python main.py --model_name gemma3:27b --experiment_name temp_0.0_gen_2 --temperature 0.0
python main.py --model_name gemma3:27b --experiment_name temp_0.3_gen_2 --temperature 0.3
python main.py --model_name gemma3:27b --experiment_name temp_0.5_gen_2 --temperature 0.5

python main.py --model_name deepseek-r1 --experiment_name temp_0.0 --temperature 0.0
python main.py --model_name deepseek-r1 --experiment_name temp_0.3 --temperature 0.3
python main.py --model_name deepseek-r1 --experiment_name temp_0.5 --temperature 0.5
python main.py --model_name deepseek-r1 --experiment_name temp_0.7 --temperature 0.7
python main.py --model_name deepseek-r1 --experiment_name temp_1.0 --temperature 1.0

python main.py --model_name deepseek-r1:32b --experiment_name temp_0.0 --temperature 0.0
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.3 --temperature 0.3
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.5 --temperature 0.5
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.7 --temperature 0.7
python main.py --model_name deepseek-r1:32b --experiment_name temp_1.0 --temperature 1.0

python main.py --model_name deepseek-r1:32b --experiment_name temp_0.0_gen_1 --temperature 0.0
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.3_gen_1 --temperature 0.3
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.5_gen_1 --temperature 0.5

python main.py --model_name deepseek-r1:32b --experiment_name temp_0.0_gen_2 --temperature 0.0
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.3_gen_2 --temperature 0.3
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.5_gen_2 --temperature 0.5
