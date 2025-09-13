import os
import pandas as pd
from pipeline.step1_conversion import DatasetConverter
from pipeline.step2_completion_open_model import TestGenerationManager
from pipeline.step2_model import Model




dataset_PROBLEM_DEF_COLUMN='prompt'
dataset_CODE_COLUMN='cleared_code'


DatasetConverter_istance = DatasetConverter()

ROLE_KEY, PROBLEM_DEF_KEY, CODE_KEY, LLM_CODE_KEY=DatasetConverter_istance.convert(input_path="DATASET/Dataset/CodeMirage_test(9).csv", 
                                                             output_path='DATASET/DatasetJSONL/CodeMirage_test.jsonl',
                                                             problem_def_column=dataset_PROBLEM_DEF_COLUMN,
                                                             code_column=dataset_CODE_COLUMN, 
                                                             LLM_code_column=None
                                                             )

model = Model()
tester = TestGenerationManager(model = model, 
                               batch_size = 1,
                               num_trials = 2)




# Generate tests for LLM written code
tester.run(prompt_path="./pipeline/configs/prompts/gen_code.md",
           input_path='DATASET/DatasetJSONL/CodeMirage_test.jsonl',  
           #output_path='DATASET/DatasetTEST/CodeMirage_test.jsonl',
           output_path='./../../../localstorage/ndelgaudio/DatasetTEST/CodeMirage_test.jsonl',
           #checkpoint_path='DATASET/DatasetTEST/checkpoints/CodeMirage_test.jsonl',
            checkpoint_path = './../../../localstorage/ndelgaudio/DatasetTEST/checkpoint_path/CodeMirage_test.jsonl',
           probelm_def_column=PROBLEM_DEF_KEY,
           # the order is not important
           code_column=LLM_CODE_KEY
           )