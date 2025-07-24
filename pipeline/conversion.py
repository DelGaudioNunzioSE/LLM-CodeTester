import csv
import json
import random
import re


class DatasetConverter():
    def __init__(self,
                 problem_def_patterns_to_remove=None,
                 code_patterns_to_remove=None,
                 role_key="role",   
                 problem_def_key="problem_def_column",
                 code_key="code_column",
                 LLM_code_key="LLM_code_column"):
        """
        problem_def_patterns_to_remove: param problem_def_patterns_to_remove: List of patterns to remove from problem definitions.
        code_patterns_to_remove: param code_patterns_to_remove: List of patterns to remove from code (both LLM and human code).
        """

        self.problem_def_patterns_to_remove = problem_def_patterns_to_remove if problem_def_patterns_to_remove is not None else []
        self.code_patterns_to_remove = code_patterns_to_remove if code_patterns_to_remove is not None else []
        self.role_key = role_key
        self.problem_def_key = problem_def_key
        self.code_key = code_key
        self.LLM_code_key = LLM_code_key

    def _clear_texts(self, text, patterns):
        """
        Remove specific patterns from the text.
        """
        for pattern in patterns:
            pattern = re.escape(pattern)  # Escape special characters in the pattern
            text = re.sub(pattern, '', text, flags=re.DOTALL)  # DOTALL per multilinea
        return text.strip()
    

    def convert(self, input_path, output_path, problem_def_column,code_column,LLM_code_column, radomize=True):
        """
        Convert a CSV file to a JSONL file with specific columns and patterns removed.
        input_file: input_file: Path to the input CSV file.
        output_file: output_file: Path to the output JSONL file.
        problem_def_column: problem_def_column: Name of the column containing the problem definition.
        code_column: Name of the column containing the code.
        LLM_code_column: Name of the column containing the LLM code.
        """
        
        with open(input_path, mode='r', encoding='utf-8') as file_csv:
            reader = csv.DictReader(file_csv)
            output_data = []


            
            for row in reader:

                # Extract the problem definition and code columns
                text_problem_def_column = row.get(problem_def_column, "")
                text_code_column = row.get(code_column, "")
                text_LLM_code_column = row.get(LLM_code_column, "")


                # ✂️
                text_problem_def_column = self._clear_texts(text=text_problem_def_column, patterns=self.problem_def_patterns_to_remove)
                text_code_column = self._clear_texts(text=text_code_column, patterns=self.code_patterns_to_remove)
                text_LLM_code_column = self._clear_texts(text=text_LLM_code_column, patterns=self.code_patterns_to_remove)

                # metadat are all columns except the problem definition and code columns
                metadata = {k: v for k, v in row.items() if k != problem_def_column and k != code_column and k != text_LLM_code_column}

                elemento = {
                    "messages": [
                        {
                            self.role_key: "user",
                            self.problem_def_key: text_problem_def_column, 
                            self.code_key: text_code_column,
                            self.LLM_code_key: text_LLM_code_column
                                
                        }
                    ],
                    "metadata": metadata
                }
                output_data.append(elemento)

        if radomize: 
            random.shuffle(output_data)
            
        with open(output_path, mode='w', encoding='utf-8') as file_json:
            for item in output_data:
                json.dump(item, file_json, ensure_ascii=False)
                file_json.write('\n')

        return self.role_key, self.problem_def_key, self.code_key, self.LLM_code_key
    
