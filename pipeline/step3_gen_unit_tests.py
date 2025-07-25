import os
import json
from tqdm import tqdm
import re
import pandas as pd
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


################
# Configurations
################
TEST_ASSERT = "assert"



class UnitTest():
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
                    self.no_test_indexes.append(idx+decreaser_index)
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

                #with open(bat_path, "r") as src:
                #    with open(os.path.join(output_path, "run_test.bat"), "w", encoding="utf-8") as dst:
                #        dst.write(src.read())

                                
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




    



    def _load_jsonl_to_dataframe(self, jsonl_path):
        # Usa pandas.read_json con lines=True
        df = pd.read_json(jsonl_path, lines=True)
        metadata_df = pd.json_normalize(df['metadata'])
        df = pd.concat([metadata_df, df.drop(columns=['metadata'])], axis=1)
        return df
    




#########

        # Scrivi tutto in un unico file JSONL
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for item in enriched_data:
                json_line = json.dumps(item, ensure_ascii=False)
                out_file.write(json_line + '\n')

        print(f"File saved in: {output_path}")

        return self._load_jsonl_to_dataframe(output_path)
    





    ###################


    def _remove_cache(self,folder_path):
        """Rimuove cache di pytest e __pycache__ se presenti"""
        
        for cache_dir in [".pytest_cache", "__pycache__"]:
            path = os.path.join(folder_path, cache_dir)
            if os.path.isdir(path):
                shutil.rmtree(path)



    def run_test(self, folder_path, timeout_seconds=10):
        """Esegue pytest in una cartella e ritorna il risultato"""

        test_file = os.path.join(folder_path, "test_solution.py")
        result_file = os.path.join(folder_path, "test_result.txt")
        folder_name = os.path.basename(folder_path)

        if not os.path.isfile(test_file):
            return f"[!] Nessun test_solution.py trovato in {folder_name}, salto."

        self._remove_cache(folder_path)

        try:
            with open(result_file, "w") as f:
                process = subprocess.run(
                    ["pytest", "test_solution.py", "-v"],
                    cwd=folder_path,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds
                )
            if process.returncode == 0:
                return f"[✓] in {folder_name}"
            else:
                return f"[✗] in {folder_name}"
        except subprocess.TimeoutExpired:
            with open(result_file, "w") as f:
                f.write(f"[TIMEOUT] of {timeout_seconds} seconds.\n")
            return f"TIMEOUT in {folder_name}"



    def run_all_tests_parallel(self, base_dir, timeout_seconds=10, max_workers=8):
        folders = sorted([
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit()
        ], key=lambda x: int(os.path.basename(x)))

        print(f"Start tests in {len(folders)} folders with {max_workers} thread\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_test, folder, timeout_seconds): folder
                for folder in folders
            }
            for future in tqdm(as_completed(futures), total=len(folders), desc="Running tests"):
                print(future.result())


#

    def _parse_test_output(self, text):
        """Estrae info da test_result.txt"""


        if "[TIMEOUT]" in text:
            return {
                "test_reliability": "NO_TEST_PASSED",
                "passed": 0,
                "failed": 0,
                "errors": ["TIMEOUT"]
            }

        passed = failed = 0
        errors = []

        # Trova linee tipo "== 2 failed, 1 passed in 0.05s =="
        summary_match = re.search(r"==+.*?(\d+) passed.*?(\d+) failed.*?==+", text)
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2))
        else:
            # fallback alternativo: cerca 'passed' e 'failed' anche separatamente
            passed_match = re.search(r"(\d+) passed", text)
            failed_match = re.search(r"(\d+) failed", text)
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0

        # Cerca messaggi di errore comuni
        error_lines = []
        for line in text.splitlines():
            if "AssertionError" in line or "Error" in line or "Exception" in line:
                error_lines.append(line.strip())

        status = "OK" if failed > 0  and passed > 1 else "NO_TEST_PASSED"

        return {
            "test_reliability": status,
            "passed": passed,
            "failed": failed,
            "errors": error_lines
        }


    def update_all_json_with_test_results(self, base_dir):
        folders = [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, f))
        ]

        print(f"==> Starting update data.json in {len(folders)} folders...\n")

        for folder in folders:
            data_path = os.path.join(folder, "data.json")
            result_path = os.path.join(folder, "test_result.txt")

            if not (os.path.isfile(data_path) and os.path.isfile(result_path)):
                continue

            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result_text = f.read()

                result_data = self._parse_test_output(result_text)

                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                data["test_result"] = result_data


                # adding folder name to metadata
                folder_name = os.path.basename(folder)
                data["metadata"]["test_folder_name"] = folder_name


                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"[✓] updated data.json in {os.path.basename(folder)}")

            except Exception as e:
                print(f"[!] Error in {os.path.basename(folder)}: {e}")





    def collect_all_data_jsons(self, input_dir: str, output_path: str):
        all_data = []

        for folder_name in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder_name)
            data_path = os.path.join(folder_path, "data.json")

            if os.path.isdir(folder_path) and os.path.isfile(data_path):
                try:
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.append(data)
                except Exception as e:
                    print(f"[!] Can't read {data_path}: {e}")

        # ✅ Scrittura nel file JSONL
        try:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for item in all_data:
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"\n[✓] File JSONL created: {output_path} ({len(all_data)} elements)")
        except Exception as e:
            print(f"[!] Error in {output_path}: {e}")



    def jsonl_to_dataframe(self, jsonl_path: str, csv_path: str = None) -> pd.DataFrame:
        data = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            print(f"[Errore] Impossibile leggere il file JSONL: {e}")
            return pd.DataFrame() 

        #  Appiattisce tutte le strutture annidate
        df = pd.json_normalize(data)

        if csv_path:
            try:
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"[✓] CSV saved in: {csv_path}")
            except Exception as e:
                print(f"[!][Error] CSV: {e}")

        return df

