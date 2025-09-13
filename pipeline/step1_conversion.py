import csv
import json
import random
import re


import itertools

# 
# 
# CSV -> JSON
#
#
#
#
#

class DatasetConverter():
    def __init__(self,
                 problem_def_patterns_to_remove=None,
                 code_patterns_to_remove=None,
                 role_key="role",   
                 problem_def_key="prompt",
                 code_key="code",
                 LLM_code_key=None
                 ):
        """
        This class is useful for converting a CSV into a format that 
        the generation framework can easily read. 
        It also allows for the automatic removal of patterns.
        """

        self.problem_def_patterns_to_remove = problem_def_patterns_to_remove if problem_def_patterns_to_remove is not None else []
        self.code_patterns_to_remove = code_patterns_to_remove if code_patterns_to_remove is not None else []
        self.role_key = role_key
        self.problem_def_key = problem_def_key
        self.code_key = code_key
        self.LLM_code_key = LLM_code_key
        self.len_max = len_max









    def _clear_texts(self, text, patterns):
        """
        Remove specific patterns from the text.
        """
        for pattern in patterns:
            pattern = re.escape(pattern)  # Escape special characters in the pattern
            text = re.sub(pattern, '', text, flags=re.DOTALL)  # DOTALL per multilinea
        return text.strip()
    



    def convert(self, input_path, output_path, problem_def_column, code_column, LLM_code_column):
        """
        Converts a CSV file to a JSONL file with specified columns and patterns removed.

        Args:
            input_path (str): Path to the input CSV file.
            output_path (str): Path to the output JSONL file.
            problem_def_column (str): Column name for problem definitions.
            code_column (str): Column name for human code.
            LLM_code_column (str): Column name for LLM-generated code.

        Returns:
            tuple: The keys used in the JSON structure (role, problem definition, code, and LLM code).
        """
        
        # read csv file like a dict
        with open(input_path, mode='r', encoding='utf-8') as file_csv:
            reader = csv.DictReader(file_csv) 
            output_data = []


            # for all the dataset ...           
            for row in reader:

                #  -> Extract the problem definition and code columns
                text_problem_def_column = row.get(problem_def_column, "")
                text_code_column = row.get(code_column, "")
                text_LLM_code_column = row.get(LLM_code_column, "") if LLM_code_column is not None else None


                # remove patterns that we do not wish to keep
                text_problem_def_column = self._clear_texts(text=text_problem_def_column, patterns=self.problem_def_patterns_to_remove)
                text_code_column = self._clear_texts(text=text_code_column, patterns=self.code_patterns_to_remove)
                text_LLM_code_column = self._clear_texts(text=text_LLM_code_column, patterns=self.code_patterns_to_remove) if LLM_code_column is not None else None

                # metadat are all columns except the problem definition and codes columns
                # messages and metadata are dict both
                metadata = {k: v for k, v in row.items() if k != problem_def_column and k != code_column and k != LLM_code_column}

                item = {
                    "messages": [
                        {
                            self.role_key: "user",
                            self.problem_def_key: text_problem_def_column, 
                            self.code_key: text_code_column,
                            self.LLM_code_key: text_LLM_code_column if self.LLM_code_key is not None else None
                                
                        }],
                    "metadata": metadata
                }
                output_data.append(item)


        # save the new jsonl file  (multiple json)  
        with open(output_path, mode='w', encoding='utf-8') as file_json:
            for item in output_data:
                json.dump(item, file_json, ensure_ascii=False)
                file_json.write('\n')

        return self.role_key, self.problem_def_key, self.code_key, self.LLM_code_key
    
