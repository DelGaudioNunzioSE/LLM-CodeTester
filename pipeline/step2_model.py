import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


class Model:
    """
    Holds configuration parameters for the model (e.g., quantization settings, dtype, parallelism settings).
    """
    def __init__(self, 
                 model_nickname : str = "meta-llama/CodeLlama-13b-Instruct-hf",
                 quantization : str = "4bit-nf4", 
                 dtype : torch.dtype = torch.bfloat16, 
                 gpu_memory_utilization : int = 1, 
                 max_tokens : int = None, 
                 max_model_len : int = None, 
                 temperature :float = 1, 
                 top_p :float = 1, 
                 repetition_penalty : float = 1, 
                 stop_tokens : str = None, 
                 stop_token_ids : int = None):
        """
        model_nickname:             Nickname of the model to use
        quantization:               what quantization to use (default: None)
        dtype:                      Data type for the model (e.g. "float16")
        gpu_memory_utilization:     GPU memory utilization (0.0 to 1.0)
        max_tokens:                 Maximum number of tokens to generate
        max_model_len:              Maximum model length
        temperature:                Temperature for generation ---------------------> to increase to increase the casuality
        top_p:                      Top-p sampling for generation ------------------> considerating only the top_p token
        repetition_penalty:         Repetition penalty for generation
        model_config_path:          Path to config file
        stop_tokens:                List of textual stop tokens  --------------------> generation stop token
        stop_token_ids:             List of token ID stop tokens --------------------> the id of the stop token
        """


        # MODEL NAME
        self.model_nickname = model_nickname



        # dtype
        valid_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        if dtype not in valid_dtypes:
            raise ValueError(f"Invalid dtype '{dtype}'. Must be one of: {valid_dtypes}")


        # gpu_memory_utilization
        if not (0.0 < gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be between 0.0 and 1.0")
        
        if top_p > 1:
            top_p = 1
            warnings.warn("top_p setted to 1")



        self.quantization = None
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
                    
            elif quantization == "dont" or quantization is None:
                self.quantization= None

            else:
                raise ValueError(f"Invalid quantization type: {quantization}. Must be one of: ['4bit-nf4', '4bit-fp4', '8bit', 'dont']")

        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty




        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_nickname,
            device_map="auto",
            dtype=self.dtype,
            quantization_config=self.quantization, # qunatization
        )


        if self.max_model_len  is None: 
            self.max_model_len = self.llm.config.max_position_embeddings
        
        if self.max_tokens is None:
            self.max_tokens = self.llm.config.max_position_embeddings


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_nickname)

        if stop_token_ids is None or stop_tokens is None:
            self.stop_tokens = self.tokenizer.eos_token
            self.stop_token_ids = self.tokenizer.eos_token_id
        else:
            self.stop_tokens = stop_tokens
            self.stop_token_ids = stop_token_ids

        return 





    def get_llm(self):
        return self.llm
    
    def get_tokenizer(self):
        return self.tokenizer