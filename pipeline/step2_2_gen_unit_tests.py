import os
import json
from tqdm import tqdm
import re
import pandas as pd
import shutil
import time
from pathlib import Path


################
# Configurations
################
TEST_ASSERT = "assert"



class GenUnitTest():
    def __init__(self):
        self.no_test_indexes = []



    def generate_tests(self, input_path, output_path, role, probelm_def_column, code ,bat_path = "./pipeline/configs/run_test.bat", sh_path= "./pipeline/configs/run_test.sh"):


        input_path = input_path
        oritigal_output_path = output_path
        self.no_test_indexes = []
        decreaser_index = 1

        
    
        # Process each input file
        with open(input_path, 'r', encoding='utf-8') as f:
            TOTAL = sum(1 for _ in f)

        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f,total=TOTAL)): # loading bar


                code_data = json.loads(line)
                # Get the assistant message content as the instruction
                user_message = next((msg for msg in code_data["messages"] if msg[role] == "user"), None)
                PROBLEM = user_message[probelm_def_column]
                CODE = user_message[code]
                assistant_message = next((msg for msg in code_data["messages"] if msg[role] == "assistant"), None)
                
                if user_message is None:
                    raise ValueError("No user message found in seed data")        
                
                if assistant_message is None:
                    decreaser_index -= 1
                    print("No assistant message found in seed data")
                    continue


                    # Extract test code from assistant message
                assistant_message = assistant_message["content"]
                TEST = re.search(r'```python(.*?)```', assistant_message, re.DOTALL)
                TEST = TEST.group(1) if TEST else None
                TEST = assistant_message
                count = TEST.count(TEST_ASSERT) if TEST else 0

                if count <3:
                    decreaser_index -= 1
                    self.no_test_indexes.append(idx+decreaser_index)
                    continue





                # generate foulder 
                output_path=os.path.join(oritigal_output_path, str(idx+decreaser_index))
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)  # elimina cartella e contenuto
                os.makedirs(output_path, exist_ok=True)

                TEST_CODE = TEST.strip().replace('```python\n', '').replace('```', '').strip()


                #with open(os.path.join(output_path, "solution.py"), "w", encoding="utf-8") as f:
                #    f.write(CODE)

                # Write test_solution.py  
                with open(os.path.join(output_path, "test_solution.py"), "w", encoding="utf-8") as f:
                    f.write(TEST_CODE)

                # Copy run_test.bat
                with open(bat_path, "r") as src:
                    with open(os.path.join(output_path, "run_test.bat"), "w", encoding="utf-8") as dst:
                        dst.write(src.read())

                                
                # Make run_test.sh executable
                #os.chmod(os.path.join(output_path, "run_test.bat"), 0o755)

                # Save processed data as JSON
                processed_data = {
                    "metadata": code_data["metadata"],
                    "instruction": PROBLEM,
                    "solution_code": CODE,
                    "test_code": TEST_CODE,
                    "file_source": input_path
                }
                if "gen_response_configs" in code_data:
                    processed_data["gen_response_configs"] = code_data["gen_response_configs"]
                with open(os.path.join(output_path, "data.json"), "w") as f:
                    json.dump(processed_data, f, indent=2)



                # double controll 
                Ppath = os.path.join(oritigal_output_path, str(idx+decreaser_index))
                if os.path.exists(Ppath) and not os.listdir(Ppath):
                    os.rmdir(Ppath)

        print(f"number of indexes without test code: {len(self.no_test_indexes)}")




    


    def copy_bat(self, dst_folder_path, src_bat_path="./pipeline/configs/run_all.bat"):

        src_bat = Path(src_bat_path)
        dst_folder = Path(dst_folder_path)

        if not src_bat.is_file():
            print(f"Error: source file not found: {src_bat}")
            return

        if not dst_folder.is_dir():
            print(f"Error: destination folder is not valid: {dst_folder}")
            return

        dst_bat = os.path.join(dst_folder, src_bat.name)

        try:
            # Copy the .bat file to the destination folder (overwrites if it exists)
            shutil.copy2(src_bat, dst_bat)
            print(f"Copied {src_bat} to {dst_bat}")

        except Exception as e:
            print(f"Error during copy or execution: {e}")



    def _load_jsonl_to_dataframe(self, jsonl_path):
        # Usa pandas.read_json con lines=True
        df = pd.read_json(jsonl_path, lines=True)
        metadata_df = pd.json_normalize(df['metadata'])
        df = pd.concat([metadata_df, df.drop(columns=['metadata'])], axis=1)
        return df
    


    def process_dataset(self, input_dir, output_path):
        # Lista per contenere tutti i json arricchiti
        enriched_data = []

        # Itera sulle sottocartelle
        for folder_name in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder_name)
            if os.path.isdir(folder_path):
                try:
                    # Percorsi dei file
                    data_path = os.path.join(folder_path, "data.json")
                    test_result_path = os.path.join(folder_path, "test_result.txt")
                    log_path = os.path.join(folder_path, "test_log.log")

                    # Leggi il JSON originale
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Leggi contenuti dei file test_result e log
                    with open(test_result_path, 'r', encoding='utf-8') as f:
                        test_result = f.read()

                    with open(log_path, 'r', encoding='utf-8') as f:
                        log = f.read()

                    # Aggiungi i nuovi campi
                    data["test_result"] = test_result
                    data["log"] = log

                    # Aggiungi alla lista dei json arricchiti
                    enriched_data.append(data)
                except Exception as e:
                    print(f"Errore nella cartella {folder_name}: {e}")

        # Scrivi tutto in un unico file JSONL
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for item in enriched_data:
                json_line = json.dumps(item, ensure_ascii=False)
                out_file.write(json_line + '\n')

        print(f"Processamento completato. File salvato in: {output_path}")

        return self._load_jsonl_to_dataframe(output_path)

