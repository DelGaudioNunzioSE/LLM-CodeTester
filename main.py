import pandas as pd
from pipeline.step1_conversion import DatasetConverter
import os
from pipeline.step2_completion_open_model import TestGenerationManager


df = pd.read_csv("DATASET/Dataset/AIGCodeSet.csv")

n_row = df.shape[0]

print(f"numbers of elements in the dataset: {n_row}.")


dataset_name = "AIGCodeSet.csv"


dataset_PROBLEM_DEF_COLUMN=None
dataset_HUMMAN_CODE_COLUMN=None
dataset_LLM_CODE_COLUMN='code'









CSV_DATASET = os.path.join('DATASET/Dataset/', dataset_name)
JSONL_DATASET = os.path.join('DATASET/DatasetJSONL/', dataset_name.replace('.csv', '.jsonl'))


DatasetConverter_istance = DatasetConverter(problem_def_patterns_to_remove=None,
                                            code_patterns_to_remove=None)






ROLE_KEY, PROBLEM_DEF_KEY, CODE_KEY, LLM_CODE_KEY=DatasetConverter_istance.convert(input_path=CSV_DATASET, 
                                                             output_path=JSONL_DATASET,
                                                             problem_def_column=dataset_PROBLEM_DEF_COLUMN,
                                                             code_column=dataset_HUMMAN_CODE_COLUMN, 
                                                             LLM_code_column=dataset_LLM_CODE_COLUMN,
                                                             radomize=True)

print(f"Conversion completed. Keys used: {ROLE_KEY}, {PROBLEM_DEF_KEY}, {CODE_KEY}, {LLM_CODE_KEY}")





MODEL_CONFIG_PATH="./pipeline/configs/model_configs.json"

Tester=TestGenerationManager(model_config_path=MODEL_CONFIG_PATH, model_nickname="HuggingFaceH4/starchat-alpha",
                             quantization="4bit-nf4", batch_size=1, checkpoint_every=3,
                             debug=True)



# JSONL_DATASET = os.path.join('DATASET/DatasetJSON/', dataset_name) defined above
JSON_WITH_TESTS = os.path.join('DATASET/DatasetTEST/', dataset_name.replace('.csv', '.jsonl'))
CHECKPOINT_JSON_WITH_TESTS = os.path.join('DATASET/DatasetTEST/checkpoints/', dataset_name.replace('.csv', '.jsonl'))



PROMPT_PATH="./pipeline/configs/prompts/gen_code.md"

# Generate tests for LLM written code
Tester.run(prompt_path=PROMPT_PATH,
           input_path=JSONL_DATASET,  
           output_path=JSON_WITH_TESTS,
           checkpoint_path=CHECKPOINT_JSON_WITH_TESTS,
           probelm_def_column=PROBLEM_DEF_KEY,
           # the order is not important
           code_column=LLM_CODE_KEY, 
           code_column2=CODE_KEY)