import torch
import os
import re
import argparse
import copy
import json
import requests
import concurrent.futures
from time import sleep, time
from tqdm import tqdm
from pipeline.utils import load_dataset_from_file, save_dataset, make_api_request_with_retry
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import glob

################
# Configurations
################

class Model:
    """
    Class to hold model configuration parameters.
    """
    def __init__(self, model_nickname, quantization, dtype,tensor_parallel_size, gpu_memory_utilization, max_tokens, max_model_len, 
                 temperature, top_p, repetition_penalty, model_config_path,
                 stop_tokens, stop_token_ids):
        """
        model_nickname: Nickname of the model to use
        quantization: what quantization to use (default: None)
        dtype: Data type for the model (e.g., "bfloat16", "float16")
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        gpu_memory_utilization: GPU memory utilization (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        max_model_len: Maximum model length
        temperature: Temperature for generation
        top_p: Top-p sampling for generation
        repetition_penalty: Repetition penalty for generation
        model_config_path: Path to config file
        stop_tokens: List of textual stop tokens
        stop_token_ids: List of token ID stop tokens
        """
        self.model_nickname = model_nickname

        if quantization is not None and quantization.lower() in ["4bit-nf4", "4bit-fp4", "8bit", "dont"]:
            if quantization == "4bit-nf4":
                self.quantization= BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True
                    )
                    
            elif quantization == "4bit-fp4":
                self.quantization= BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="fp4",
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True
                    )
                    
            elif quantization == "8bit":
                self.quantization= BitsAndBytesConfig(
                load_in_8bit=True
                )
                    
            elif quantization == "dont":
                self.quantization= None

            else:
                raise ValueError(f"Invalid quantization type: {quantization}. Must be one of: ['4bit-nf4', '4bit-fp4', '8bit', 'dont']")

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.model_config_path = model_config_path
        self.stop_tokens = stop_tokens
        self.stop_token_ids = stop_token_ids




    def print_config(self):
        """
        Print the model configuration.
        """
        print(f"Model Nickname: {self.model_nickname}")
        print(f"Quantization: {self.quantization}")
        print(f"Dtype: {self.dtype}")
        print(f"Tensor Parallel Size: {self.tensor_parallel_size}")
        print(f"GPU Memory Utilization: {self.gpu_memory_utilization}")
        print(f"Max Tokens: {self.max_tokens}")
        print(f"Max Model Length: {self.max_model_len}")
        print(f"Temperature: {self.temperature}")
        print(f"Top P: {self.top_p}")
        print(f"Repetition Penalty: {self.repetition_penalty}")
        print(f"Model Config Path: {self.model_config_path}")

        



class TestGenerationManager:
    def __init__(self,batch_size=128,checkpoint_every=20, 
                 model_nickname="Qwen/Qwen2.5-14B-Instruct",
                 quantization="8bit", api_url=None, api_key=None,
                 device="0",dtype="float16", tensor_parallel_size=1, gpu_memory_utilization=0.95,  max_tokens=8192,
                 max_model_len=4000, temperature=1.0, top_p=1.0, repetition_penalty=1.0, 
                 num_trials=1, model_config_path="configs/model_configs.json", debug=False):
        '''
        batch_size: Number of samples per batch (default: 128)
        checkpoint_every: Save checkpoint every n batches
        model_nickname: model name
        quantization: what quantization to use (default: None)
        checkpoint_every: Save checkpoint every n batches (default: 20)
        api_url: API URL for the model (if using API) (default: None)
        api_key: API key for the model (if using API) (default: None)
        device: Device to use for generation (default: "0")
        dtype: Data type for the model (default: "bfloat16")
        tensor_parallel_size: Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.
        gpu_memory_utilization: GPU memory utilization (default: 0.95) 
        max_tokens: Maximum number of tokens to generate (default: 8192)
        max_model_len: Maximum model length (default: 4000)
        temperature: Temperature for generation (default: 1.0)
        top_p: Top-p sampling for generation (default: 1.0)
        repetition_penalty: Repetition penalty for generation (default: 1.0)
        num_trials: Number of trials to run (default: 1)
        model_config_path: Path to the model configuration file (default: "configs/model_configs.json")
        '''
        #model_nickname <-saved in model object
        # self.quantization = quantization <-saved in model object
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.api_url = api_url
        self.api_key = api_key
        self.device = device
        # self.dtype = dtype <-saved in model object
        # self.tensor_parallel_size = tensor_parallel_size <-saved in model object
        # self.gpu_memory_utilization = gpu_memory_utilization <-saved in model object
        #max_tokens <-saved in model object
        #max_model_len <-saved in model object
        #temperature <-saved in model object
        #top_p <-saved in model object
        #repetition_penalty <-saved in model object
        self.num_trials = num_trials
        #model_config_path <-saved in model object
        self.debug = debug

        
        valid_dtypes = ["float16", "bfloat16"]
        if dtype not in valid_dtypes:
            raise ValueError(f"Invalid dtype '{dtype}'. Must be one of: {valid_dtypes}")

        if not (0.0 < gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be between 0.0 and 1.0")




        
        # model configurations
        with open(model_config_path, "r",encoding="utf-8") as f:
            self.model_configs = json.load(f)
            self.model_config = self.model_configs[model_nickname]
            temp_stop_tokens = self.model_config["stop_tokens"]
            temp_stop_token_ids = self.model_config["stop_token_ids"]
        
        # model configuration object
        self.model= Model(
            model_nickname=model_nickname,
            quantization=quantization,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            model_config_path=model_config_path,
            stop_tokens=temp_stop_tokens,
            stop_token_ids=temp_stop_token_ids
        )

        if self.debug:
            self.model.print_config()



    def prompt_elaboration(self,problem_def ,code,prompt_path):
        with open(prompt_path, encoding="utf-8") as f:
            prompt_template = f.read()

        return prompt_template.replace("{description}", problem_def).replace("{code}", code)

        

    # Process a batch of data using local vllm engine
    def process_batch(self, batch, llm, params, 
                      problem_def_column, code_column,
                      prompt_path,
                      tokenizer=None):
        '''
        Process a batch of data using local vllm engine or Hugging Face engine.
        batch: List of messages to process (instructions and code)
        llm: LLM used for generation
        params: Sampling parameters for generation (optional, only needed for vllm engine)
        tokenizer: Tokenizer object for encoding/decoding (optional, only needed for Hugging Face engine)
        IMPORTANT:
         - problem_def_column: The name of the column containing problem definitions (default: "Problem")
         - code_column: The name of the column containing code solution (default: "Python Code")
        '''
        # obtain problem definition
        problems_definitions = [item['messages'][0][problem_def_column] for item in batch] 
        codes_raw = [item['messages'][0][code_column] for item in batch]
        codes = [
            re.search(r"def\s+[^\n]+", code).group(0)
            if re.search(r"def\s+[^\n]+", code)
            else code.splitlines()[0]
            for code in codes_raw
        ]
        local_prompts = []
        for problem_def,code in zip(problems_definitions, codes):
            PROMPT = self.prompt_elaboration(problem_def=problem_def, code=code, prompt_path=prompt_path)
            chat = [{"role": "user", "content": PROMPT}] 
            template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            local_prompts.append(template)

            if self.debug:
                print(f" template: {template}")
                print(f"\n-------------------------------------------------------\n\n")



        # 1. Tokenizzo senza padding/troncamento per misurare la vera lunghezza
        tokenized = tokenizer(local_prompts, padding=False, truncation=False, return_tensors=None)

        # 2. Calcolo la lunghezza di ogni prompt in token
        input_lengths = [len(x) for x in tokenized["input_ids"]]

        # 3. Trovo gli indici validi
        valid_indices = [i for i, l in enumerate(input_lengths) if l <= self.model.max_model_len]
        invalid_indices = [i for i in range(len(local_prompts)) if i not in valid_indices]
        if invalid_indices:
            print(f"[INFO] Discarded {len(invalid_indices)} prompt too much long. batch idex: {invalid_indices}")

        # 4. Filtro i local_prompts in base a questi indici
        filtered_prompts = [local_prompts[i] for i in valid_indices]

        # 5. Tokenizzo solo i prompt validi
        inputs = tokenizer(filtered_prompts, return_tensors="pt", padding=True, truncation=True).to(torch.cuda.current_device())

        gen_do_sample = False if self.model.temperature == 0 else True
        outputs = llm.generate(**inputs,
                tokenizer=tokenizer, 
                do_sample=gen_do_sample, 
                temperature=self.model.temperature if gen_do_sample else None, # To avoid temperature` (=0) has to be a strictly positive float
                top_p=self.model.top_p,
                repetition_penalty=self.model.repetition_penalty, 
                max_length=self.model.max_tokens
                )
        outputs = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))
        # Setting stop tokens seems not working for Gemma, so we manually truncate the outputs
        for i, completion in enumerate(outputs):
            for stop_token in self.model.stop_tokens:
                if stop_token in completion:
                    outputs[i] = completion[:completion.index(stop_token)]

        batch = [batch[i] for i in valid_indices]
        # generrate/modificate field messages in batch
        for i, item in enumerate(batch):
            message = item["messages"]
            response = outputs[i].strip()
            item['messages'] =  message+ [
                    {   
                        "role": "assistant",
                        "content": response
                    }
                ]
            
        return batch


    def number_of_batches(self, dataset,dataset_done=0):
        '''
        Calculate the number of batches needed for the dataset.
        dataset: List of dictionaries containing code and instructions
        '''
        return (len(dataset) - dataset_done + self.batch_size  - 1) // self.batch_size
    



    # Generate outputs, update dataset in batches, and overwrite checkpoint
    def generate_and_update(self, dataset, checkpoint_path, problem_def_column,
                            code_column, prompt_path, llm=None, params=None, tokenizer=None):
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
            batch = self.process_batch(batch=batch, llm=llm, params=params, prompt_path=prompt_path, tokenizer=tokenizer,
                                       problem_def_column=problem_def_column, code_column=code_column)
            
            processed_dataset[start_idx:end_idx] = batch
            # Overwrite the same checkpoint file after serveral batches
            if i % self.checkpoint_every == 0:
                save_dataset(processed_dataset[:end_idx], checkpoint_path, convert_to_jsonl=True)
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
            checkpoint_file = f"{checkpoint_path}_results_checkpoint.jsonl"
            saved_file = f"{output_path}_results.jsonl"


        # Load instructions from the input file
        dataset = load_dataset_from_file(input_path)
        
        # HF-LLM engine
        print("Start Hugging Face engine...")
        params = None
        #

        # Load the model and tokenizer
        llm = AutoModelForCausalLM.from_pretrained(
            self.model.model_nickname,
            device_map={'':torch.cuda.current_device()},
            torch_dtype=self.model.dtype,
            quantization_config=self.model.quantization, # qunatization
        )
        torch.cuda.empty_cache()
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model.model_nickname)
        

        if self.num_trials == 1:
            updated_dataset = self.generate_and_update(dataset=dataset, checkpoint_path=checkpoint_file, 
                                                       llm=llm, params=params, prompt_path=prompt_path, tokenizer=tokenizer,
                                                       problem_def_column=probelm_def_column, code_column=code_column)
            save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)

            # Optionally remove the checkpoint file after completion
            os.remove(checkpoint_file)
            print("Final dataset saved. Checkpoint removed.")
        else:
            for i in range(self.num_trials):
                updated_dataset = self.generate_and_update(dataset=dataset, checkpoint_path=checkpoint_files[i], 
                                                           llm=llm, params=params, prompt_path= prompt_path, tokenizer=tokenizer,
                                                           problem_def_column=probelm_def_column, code_column=code_column)
                save_dataset(updated_dataset, saved_files[i], convert_to_jsonl=True)

                # Optionally remove the checkpoint file after completion
                os.remove(checkpoint_files[i])
                print(f"Dataset for trial {i} saved. Checkpoint {i} removed.")
