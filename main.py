import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from DAQ import quantizer
from ABMP import allocator

MODEL_PATH = "/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                            trust_remote_code=True,
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            device_map="auto",
                                            max_memory={
                                                0: "13GiB",
                                                "cpu": "80GiB",
                                            },
                                            offload_folder="/tmp/llada_offload")

