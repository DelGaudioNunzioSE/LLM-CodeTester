from pipeline.step2_2_gen_unit_tests import GenUnitTest
GenTest=GenUnitTest()
HUMAN_CODE_TESTGEN_DATASET = "./DatasetTESTGEN/human"
ROLE_KEY="role"   
PROBLEM_DEF_KEY="problem_def_column"
CODE_KEY="code_column"
LLM_CODE_KEY="LLM_code_column"
GenTest.generate_tests(input_path="./DatasetTEST/variant_1_full_output.jsonl_results.jsonl",
            #input_path=HUMAN_CODE_TEST_DATASET[file_index],
            output_path=HUMAN_CODE_TESTGEN_DATASET,
            role = ROLE_KEY,
            probelm_def_column= PROBLEM_DEF_KEY,
            code = CODE_KEY)