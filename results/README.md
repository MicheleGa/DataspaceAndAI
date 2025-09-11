# Dataspec & AI

## Results

In this folder you can find the results of system execution.

### Harmonization & Transformation

The JSONs messages (from the *../data/json_data* folder) are processed for harmonization and transformed into the harmonized schema. The results of these two steps of the sytem are saved inside the AI agent corresponding folder. Currently, these two steps were successfull with the following OLLAMA agents:

> - Deepseek: results are in *./deepseek-r1:32b_temp_0.5_20250731_115901*   
> - Gemma 3: results are in *./gemma3:27b_20250731_102743*

### System Evaluation

The system performing the semantic harmonization can be evaluated quantitatively and qualitatively.
The *data_visualization.py* script provides plots of the quantitative assessment of AI agent results. More precisely, the structural and semantic similarities are calculated as quantitative measures for the harmonziation process. The plots are inside of the *key_aligment* folder.

The qualitative assessment of the hamonization process is performed with a third AI agent, in addition to those for JSON messages harmonizationa nd transformation.
The third AI agent qualitative assessment is executed for the intial JSONs messages similarities and for both Deepseek and Gemma 3 agents. You can find results in the folders *agent_evaluation* inside *./deepseek-r1:32b_temp_0.5_20250731_115901*, *./gemma3:27b_20250731_102743*, and *./initial_jsons* folders.