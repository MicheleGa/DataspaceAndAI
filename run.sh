# Test different models

# Gemma 3 is only for debuggin as it has a relatively fast executiion time among the slected LLMs
#python main.py --model_name gemma3 --transform False --use_key_alignment True

# Working models
python main.py --model_name gemma3:27b
python main.py --model_name deepseek-r1:32b --experiment_name temp_0.5 --temperature 0.5
