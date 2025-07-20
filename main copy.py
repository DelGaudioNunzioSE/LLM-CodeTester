from pipeline.step2_2_gen_unit_tests import GenUnitTest
import pandas as pd
GenTest=GenUnitTest()
HUMAN_CODE_TESTGEN_DATASET = "./DatasetTESTGEN/human"
HUMAN_CODE_TESTGEN_DATASETOUTPUT = "./DatasetTESTGENOUTPUT/human/final.jsonl"

df =GenTest.process_dataset(input_dir=HUMAN_CODE_TESTGEN_DATASET,output_path= HUMAN_CODE_TESTGEN_DATASETOUTPUT)
print(df.head())
df.to_csv("./DatasetTESTGENOUTPUT/human/final.csv", index=False)
print(df['test_result'].value_counts())
