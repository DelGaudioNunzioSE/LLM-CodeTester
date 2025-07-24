from pipeline.step2_1_completion_open_model import TestGenerationManager

filter_file = [8,9,10,13] # files with other aims
CSV_DATASET = [f'DATASET/Dataset/variant_{i}_full.csv' for i in range(1, 14) if i not in filter_file]
JSON_DTASET = [f'DATASET/DatasetJSON/variant_{i}.jsonl' for i in range(1, 14) if i not in filter_file]


CSV_PROBLEM_DEF_COLUMN='Problem'
CSV_CODE_COLUMN='Python Code'
CSV_LLM_CODE_COLUMN='GPT Answer'

# just in case we jump the conversion part
ROLE_KEY="role"   
PROBLEM_DEF_KEY="problem_def_column"
CODE_KEY="code_column"
LLM_CODE_KEY="LLM_code_column"



PROMPT_PATH="./pipeline/configs/prompts/gen_test.md"
PROMPT_PATH_INPUT="./pipeline/configs/prompts/gen_inputs.md"
MODEL_CONFIG_PATH="./pipeline/configs/model_configs.json"

filter_file = [8,9,10,13] # files with other aims
#HUMAN_CODE_TEST_DATASET = [f"./DatasetTEST/variant_{i}_full_output.jsonl" for i in range(1, 14) if i not in filter_file]
LLM_CODE_TEST_DATASET = [f"./DATASET/DatasetTEST/variant_{i}_LLM_code.jsonl" for i in range(1, 14) if i not in filter_file]
LLM_CODE_TEST_DATASET_INPUT = [f"./DATASET/DatasetTEST/variant_{i}_INPUTS.jsonl" for i in range(1, 14) if i not in filter_file]
#CHECKPOINTS_HUMAN_CODE_TEST_DATASET = [f"./DatasetTEST/checkpoints/variant_{i}_full_output.jsonl" for i in range(1, 14) if i not in filter_file]
CHECKPOINTS_LLM_CODE_TEST_DATASET = [f"./DATASET/DatasetTEST/checkpoints/variant_{i}_LLM_code.jsonl" for i in range(1, 14) if i not in filter_file]
CHECKPOINTS_LLM_CODE_TEST_DATASET_INPUT = [f"./DATASET/DatasetTEST/checkpoints/variant_{i}_INPUTS.jsonl" for i in range(1, 14) if i not in filter_file]






Tester=TestGenerationManager(model_config_path=MODEL_CONFIG_PATH, model_nickname="Qwen/Qwen2.5-14B-Instruct",
                             quantization="4bit-nf4", batch_size=1, checkpoint_every=3,
                             debug=True)



file_index = 1
Tester.run(prompt_path="./pipeline/configs/prompts/gen_inputs.md",
           input_path=JSON_DTASET[file_index],  
           output_path=LLM_CODE_TEST_DATASET[file_index],
           checkpoint_path=CHECKPOINTS_LLM_CODE_TEST_DATASET[file_index],
           probelm_def_column=PROBLEM_DEF_KEY,
           code_column=LLM_CODE_KEY,
           code_column2=CODE_KEY)