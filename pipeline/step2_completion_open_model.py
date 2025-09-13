
import warnings
import torch
import os
import re
import copy
from time import sleep, time
from tqdm import tqdm
from pipeline.utils import load_dataset_from_file, save_dataset, make_api_request_with_retry

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

################################################

from pipeline.step2_model import Model




#       MODEL PARAMETER ____________________________________________________________
#       model_nickname:         Nickname of the model to use
#       quantization:           what quantization to use (default: None)
#       dtype:                  Data type for the model (e.g., "bfloat16", "float16")
#       tensor_parallel_size:   Number of GPUs to use for tensor parallelism
#       gpu_memory_utilization: GPU memory utilization (0.0 to 1.0)
#       max_tokens:             Maximum number of tokens to generate
#       max_model_len:          Maximum model length
#       temperature:            Temperature for generation
#       top_p:                  Top-p sampling for generation
#       repetition_penalty:     Repetition penalty for generation
#       model_config_path:      Path to config file
#       stop_tokens:            List of textual stop tokens
#       stop_token_ids:         List of token ID stop tokens
#
#       get_model:              self.model (the model from hugging face)
#       get_tokenizer:          self.tokenizer (the tokenizer of the model)

        


################################################################################
class TestGenerationManager:
    def __init__(self,
                 model : Model,
                 batch_size=32,
                 checkpoint_every=5, 
                 num_trials: int = 4
                 ):
        '''
        batch_size: Number of samples per batch (default: 32)
        checkpoint_every: Save checkpoint every n batches
        model_nickname: model class reference

        '''

        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every

        self.num_trials = num_trials

        self.model= model

        return 
            



    def prompt_elaboration(self, 
                           problem_def : str, 
                           code1: str, 
                           prompt_path : str = "./pipeline/configs/prompts/get_code.md") -> str:
        
        '''Generate the prompt for the LLM'''

        with open(prompt_path, encoding="utf-8") as f:
            prompt_template = f.read()

            prompt_elaborated = prompt_template.replace("{description}", problem_def).replace("{code1}", code1)

        return  prompt_elaborated










    # For the code tester branch ##
    #def code_elaboration(self, code):
    #    try:
    #        tree = ast.parse(code)
    #    except SyntaxError as e:
    #        print(f"[ERROR] Syntax error while parsing: {e}")
    #        return code  # fallback

    #    import_nodes = []
    #    other_nodes = []

    #    for node in tree.body:
    #        if isinstance(node, (ast.Import, ast.ImportFrom)):
    #            import_nodes.append(node)
    #        else:
    #            other_nodes.append(node)

    #    # Convert AST back to code (Python 3.9+)
    #    try:
    #        import_code = "\n".join([ast.unparse(n) for n in import_nodes])
    #        other_code = "\n".join([ast.unparse(n) for n in other_nodes])
    #    except Exception as e:
    #        print(f"[ERROR] Could not unparse AST: {e}")
    #        return code  # fallback

    #    return import_code.strip() + "\n\n" + other_code.strip()









    # For the code tester branch ##
    #def _suggest_imports(self, codes: list[str]) -> list[str]:
    #    updated_codes = []
    #    for code in codes:
    #        updated_codes.append( f"from typing import *\nfrom collections import *\nfrom math import *\n{code}")
    #    return updated_codes






    # For the code tester branch ##
    #def _rename_class(self, codes: list[str]) -> list[str]:
    #    updated_codes = []
    #    class_name_template = "{}_class"
    #    class_prefixes = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]

    #    for code in codes:
    #        class_counter = 0

    #        def replacer(match):
    #            nonlocal class_counter
    #            prefix = class_prefixes[class_counter] if class_counter < len(class_prefixes) else f"Class{class_counter+1}"
    #            class_counter += 1
    #            return f"class {class_name_template.format(prefix)}"

    #        # Regex per trovare definizioni di classi
    #        code = re.sub(r'class\s+(\w+)\s*:', replacer, code)
    #       updated_codes.append(code)

    #    return updated_codes
    




    #import ast


    # For the code tester branch ##
    #def _keep_only_functions_and_classes(self, codes: list[str]) -> list[str]:
    #    updated_codes = []
    #    for code in codes:
    #        try:
    #            tree = ast.parse(code)

    #            filtered_nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]

    #            new_module = ast.Module(body=filtered_nodes, type_ignores=[])

    #            updated_codes.append( ast.unparse(new_module) )

    #        except Exception as e:
    #            return f"# Errore durante la rimozione del codice fuori da funzioni/classi:\n# {e}"
    #    return updated_codes
            



#
#
# Generated by step1_conversion
#    item = {
#         "messages":
#              {
#                self.role_key: "user",
#                elf.problem_def_key: text_problem_def_column, 
#                self.code_key: text_code_column,
#                self.LLM_code_key: text_LLM_code_column
#                                
#                },
#             "metadata": metadata
#          }
#
#
#

    def process_batch(self, 
                      batch_of_item: list, 
                      llm_model : AutoModelForCausalLM, 
                      tokenizer : AutoTokenizer,
                      code_column: str = "code", 
                      prompt_path : str = "./pipeline/configs/prompts/gen_code.md",
                      
                      ):
        
        # obtain problem definition
        local_prompts = []
        # item is a dict of 2 dicts andhe the first one has inside a list within a dict
        codes = [item['messages'][0]["code"] for item in batch_of_item]
        
        for code1 in codes:
            if code1 is None: 
                print("Code could not be parsed. Please check the code format.")
                return []
            


            # elaboration for an Instruct model
            PROMPT = self.prompt_elaboration(problem_def=" ", code1=code1, prompt_path=prompt_path)
            chat = [
                    {"role": "system", "content": "You are a LLM code generator. For every code snippet provided, first understand what it does, then rewrite it in a single markdown code block."},
                    {"role": "user", "content": PROMPT},
                ]

            template = tokenizer.apply_chat_template(chat, tokenize=False)

            local_prompts.append(template)





        ### check the max_model_len 
        # 1. Tokenizzo senza padding/troncamento per misurare la vera lunghezza
        #tokenized = tokenizer(local_prompts, padding=False, truncation=False, return_tensors=None, add_special_tokens=False)

        # 2. Calcolo la lunghezza di ogni prompt in token
        #input_lengths = [len(x) for x in tokenized["input_ids"]]

        # 3. Trovo gli indici validi
        #valid_indices = [i for i, l in enumerate(input_lengths) if l <= self.model.max_model_len]
        #invalid_indices = [i for i in range(len(local_prompts)) if i not in valid_indices]
        #if invalid_indices:
        #    print(f"[INFO] Discarded {len(invalid_indices)} prompt too much long. batch idex: {invalid_indices}")

        # 4. Filtro i local_prompts in base a questi indici
        #filtered_prompts = [local_prompts[i] for i in valid_indices]

        #if filtered_prompts == []:
        #    warnings.warn("No valid prompts to process in this batch. Skipping...")
        #    return []
        ######
        
        # 5. Tokenizzo solo i prompt validi
        inputs = tokenizer(local_prompts, 
                           return_tensors="pt", 
                           padding=True, 
                           truncation=True).to(torch.cuda.current_device()) 



        gen_do_sample = False if self.model.temperature == 0 else True

        # Generation
        outputs = llm_model.generate(**inputs,
                tokenizer=tokenizer, 
                do_sample=gen_do_sample, 
                temperature=self.model.temperature if gen_do_sample else None, # To avoid temperature` (=0) has to be a strictly positive float
                top_p=self.model.top_p,
                repetition_penalty=self.model.repetition_penalty, 
                max_length=self.model.max_tokens
                )
        
        # collect generation token -> str whitout input
        outputs = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))

        # Setting stop tokens seems not working for some LLM, so we manually truncate the outputs
        #for i, completion in enumerate(outputs):
        #    for stop_token in self.model.stop_tokens:
        #        if stop_token in completion:
        #            outputs[i] = completion[:completion.index(stop_token)]

        #batch_of_item_output = [batch_of_item[i] for i in valid_indices]


        # generrate/modificate field messages in batch
        for i, item in enumerate(batch_of_item):
            # obtaining the oringial message
            message = item["messages"]

            print(outputs[i] )
            if outputs[i] is not None:
                outputs[i] = re.search(r'```(.*?)```', outputs[i].strip(), re.DOTALL)
                if outputs[i] is not None:
                    outputs[i]  = outputs[i] .group(1)
                else:
                    outputs[i] = None 
                
                # NEW VALUE FOR ITEM
                item['messages'] =  message+ [ # cange the messages field of the batch
                        {   
                            "role": "assistant",
                            "content": outputs[i] 
                        }
                    ]
                
            
        return batch_of_item














    # Process a batch of data using local vllm engine
    #def process_batch(self, batch, llm, params, 
    #                  problem_def_column, code_column, code_column2, 
    #                  prompt_path,
    #                  tokenizer=None):
    #    '''
    #    Process a batch of data using local vllm engine or Hugging Face engine.
    #    batch: List of messages to process (instructions and code)
    #    llm: LLM used for generation
    #    params: Sampling parameters for generation (optional, only needed for vllm engine)
    #    tokenizer: Tokenizer object for encoding/decoding (optional, only needed for Hugging Face engine)
    #    IMPORTANT:
    #     - problem_def_column: The name of the column containing problem definitions (default: "Problem")
    #     - code_column: The name of the column containing code solution (default: "Python Code")
    #    '''

    #    if 'gen_code' in prompt_path: return self.procesprocess_batch_alternatives(batch, llm, params, code_column, prompt_path, tokenizer=tokenizer)
        
        
    #    # obtain problem definition
    #   problems_definitions = [item['messages'][0][problem_def_column] for item in batch] 
    #    local_prompts = []
    #    codes1 = [item['messages'][0][code_column] for item in batch]
    #    codes1 = self._rename_class(codes=codes1)  # Rename class Solution to First_class
    #    codes2 = [item['messages'][0][code_column2] for item in batch]
    #    codes1 = self._keep_only_functions_and_classes(codes1)
    #    codes2 = self._keep_only_functions_and_classes(codes2)
    #    #codes1=self._suggest_imports(codes=codes1)  # Suggest imports for code1
    #    #codes2=self._suggest_imports(codes=codes2)  # Suggest imports for code2
        
    #    for problem_def, code1, code2 in zip(problems_definitions, codes1, codes2):
    #        #code1 = self.code_elaboration(code1)
    #        #code2 = self.code_elaboration(code2)
    #        def1= 'def' in code1
    #        def2= 'def' in code2
    #        if code1 is None or code2 is None or def1 is False or def2 is False: 
    #            print("Code could not be parsed. Please check the code format.")
    #            return batch
    #        PROMPT = self.prompt_elaboration(problem_def=problem_def, code1=code1, code2=code2, prompt_path=prompt_path)
    #        chat = [{"role": "user", "content": PROMPT}] 
    #        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    #        local_prompts.append(template)

    #        if self.debug:
    #            print(f" input:\n {template}")
    #            print(f"\n-------------------------------------------------------\n\n")



    #    # 1. Tokenizzo senza padding/troncamento per misurare la vera lunghezza
    #    tokenized = tokenizer(local_prompts, padding=False, truncation=False, return_tensors=None)

    #    # 2. Calcolo la lunghezza di ogni prompt in token
    #    input_lengths = [len(x) for x in tokenized["input_ids"]]

    #    # 3. Trovo gli indici validi
    #    valid_indices = [i for i, l in enumerate(input_lengths) if l <= self.model.max_model_len]
    #    invalid_indices = [i for i in range(len(local_prompts)) if i not in valid_indices]
    #    if invalid_indices:
    #        print(f"[INFO] Discarded {len(invalid_indices)} prompt too much long. batch idex: {invalid_indices}")

    #    # 4. Filtro i local_prompts in base a questi indici
    #    filtered_prompts = [local_prompts[i] for i in valid_indices]

    #    if filtered_prompts == []:
    #        print("[INFO] No valid prompts to process in this batch. Skipping...")
    #        return batch
        
    #    # 5. Tokenizzo solo i prompt validi
    #    inputs = tokenizer(filtered_prompts, return_tensors="pt", padding=True, truncation=True).to(torch.cuda.current_device()) 

    #    gen_do_sample = False if self.model.temperature == 0 else True
    #    outputs = llm.generate(**inputs,
    #            tokenizer=tokenizer, 
    #            do_sample=gen_do_sample, 
    #            temperature=self.model.temperature if gen_do_sample else None, # To avoid temperature` (=0) has to be a strictly positive float
    #            top_p=self.model.top_p,
    #            repetition_penalty=self.model.repetition_penalty, 
    #            max_length=self.model.max_tokens
    #            )
    #    outputs = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))
    #    # Setting stop tokens seems not working for Gemma, so we manually truncate the outputs
    #    for i, completion in enumerate(outputs):
    #        for stop_token in self.model.stop_tokens:
    #            if stop_token in completion:
    #                outputs[i] = completion[:completion.index(stop_token)]

    #   batch = [batch[i] for i in valid_indices]


    #    # generrate/modificate field messages in batch
    #    for i, item in enumerate(batch):
    #        message = item["messages"]
    #        #response = codes1[i] + codes2[i] + outputs[i].strip()
    #        if outputs[i] is not None:
    #            outputs[i] = re.search(r'```python(.*?)```', outputs[i].strip(), re.DOTALL).group(1)

    #            codes1=self._suggest_imports(codes=codes1)  # Suggest imports for code1
    #            
    #            response = "\n\n".join([
    #                textwrap.dedent(codes1[i]).strip(),
    #                textwrap.dedent(codes2[i]).strip(),
    #                textwrap.dedent(outputs[i]).strip()
    #            ])

    #            response = self.code_elaboration(response)   
    #            
    #            item['messages'] =  message+ [ # cange the messages field of the batch
    #                    {   
    #                        "role": "assistant",
    #                        "content": response
    #                    }
    #                ]
    #            
    #            if self.debug:
    #                print(f"Response for item {i}:\n{response}")
    #                print(f"\n-------------------------------------------------------\n\n")
    #        
    #    return batch












    def number_of_batches(self, dataset,dataset_done=0):
        '''
        Calculate the number of batches needed for the dataset.
        dataset: List of dictionaries containing code and instructions
        '''
        return (len(dataset) - dataset_done + self.batch_size  - 1) // self.batch_size
    












    # Generate outputs, update dataset in batches, and overwrite checkpoint
    def generate_and_update(self, 
                            llm : AutoModelForCausalLM, 
                            tokenizer : AutoTokenizer,
                            dataset : list, 
                            checkpoint_path : str, 
                            problem_def_column : str,
                            code_column : str = "code",
                            prompt_path: str = "./pipeline/configs/prompts/gen_code.md") -> str:
        '''
        Generate outputs for the dataset in batches and update the dataset.
        dataset: List of dictionaries containing code and instructions
        checkpoint_path: File to save the checkpoint
        llm: LLM used for generation
        params: Sampling parameters for generation
        tokenizer: Tokenizer object for encoding/decoding (optional, only needed for Hugging Face engine)
        '''
        processed_dataset = copy.deepcopy(dataset)

        # Initialize tokenizer
        if tokenizer is not None:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            if "gemma-2" in self.model.model_nickname.lower():
                # Gemma-2 richiede padding a destra
                tokenizer.padding_side = "right"
            else:
                # Tutti gli altri decoder-only: padding a sinistra
                tokenizer.padding_side = "left"




        # Intialize the dataset with the checkpoint file (if it exists)
        if os.path.exists(checkpoint_path):
            last_checkpoint_idx = len(load_dataset_from_file(checkpoint_path))
            print(f"Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
            processed_dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_path)
            # Calculate total number of batches
            num_batches = self.number_of_batches(dataset=processed_dataset, dataset_done=last_checkpoint_idx)

            print(f"Remaining number of batches: {num_batches}")


        else:
            last_checkpoint_idx = 0
            # Calculate total number of batches
            num_batches = self.number_of_batches(dataset=processed_dataset)
            print(f"Total number of batches: {num_batches}")



        for i in tqdm(range(num_batches)):
            start_idx = i * self.batch_size + last_checkpoint_idx
            end_idx = min((i + 1) * self.batch_size + last_checkpoint_idx, len(processed_dataset))
            batch = processed_dataset[start_idx:end_idx]
            batch = self.process_batch(batch_of_item=batch, 
                                    llm_model=llm, 
                                    prompt_path=prompt_path, 
                                    tokenizer=tokenizer,
                                    code_column=code_column)
            
            processed_dataset[start_idx:end_idx] = batch
            # Overwrite the same checkpoint file after serveral batches
            if i % self.checkpoint_every == 0:
                save_dataset(processed_dataset[:end_idx], checkpoint_path, convert_to_jsonl=True, append= False)
                print(f"Dataset checkpoint saved after batch {i + 1}.")

        return processed_dataset







    # Main function to control workflow
    def run(self,input_path,output_path,checkpoint_path,prompt_path,
            probelm_def_column, code_column):

        # Setting ingput and output paths
                # Create different output folders for different trials
        if self.num_trials > 1:
            checkpoint_files = [f"{checkpoint_path}_results_checkpoint{i}.jsonl" for i in range(self.num_trials)]
            saved_files = [f"{output_path}_results{i}.jsonl" for i in range(self.num_trials)]
        else:
            checkpoint_file = f"{checkpoint_path}"
            saved_file = f"{output_path}"


        # Load instructions from the input file
        dataset = load_dataset_from_file(input_path)
        
        # HF-LLM engine
        print("Start Hugging Face engine...")


        # Load the model and tokenizer
        llm = self.model.get_llm()

        torch.cuda.empty_cache()
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        tokenizer = self.model.get_tokenizer()
        

        if self.num_trials == 1:
            updated_dataset = self.generate_and_update(dataset=dataset, checkpoint_path=checkpoint_file, 
                                                       llm=llm, prompt_path=prompt_path, tokenizer=tokenizer,
                                                       problem_def_column=probelm_def_column, code_column=code_column)
            save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)

            # Optionally remove the checkpoint file after completion
            os.remove(checkpoint_file)
            print("Final dataset saved. Checkpoint removed.")
        else:
            for i in range(self.num_trials):
                updated_dataset = self.generate_and_update(dataset=dataset, checkpoint_path=checkpoint_files[i], 
                                                           llm=llm, prompt_path= prompt_path, tokenizer=tokenizer,
                                                           problem_def_column=probelm_def_column, code_column=code_column)
                save_dataset(updated_dataset, saved_files[i], convert_to_jsonl=True)

                # Optionally remove the checkpoint file after completion
                os.remove(checkpoint_files[i])
                print(f"Dataset for trial {i} saved. Checkpoint {i} removed.")#
