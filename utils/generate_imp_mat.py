import torch
import torch.nn as nn
import sys
from MCS import load_dataset as mcs
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from safetensors import safe_open
from pathlib import Path
import json

MODEL_PATH = "/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base"

def load_weight_map(model_path=MODEL_PATH):
    index_path = model_path + "/model.safetensors.index.json"
    with open(index_path, "r") as f:
        return json.load(f)["weight_map"]

def load_weight_from_checkpoint(weight_name, weight_map, model_path=MODEL_PATH):
    if weight_name not in weight_map:
        raise KeyError(f"{weight_name} not found in checkpoint index.")

    shard_path = model_path + '/' + weight_map[weight_name]

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        weight = f.get_tensor(weight_name)

    return weight.to(dtype=torch.float32)

def init_imp_mat(model,device="cpu"):
    mat={}
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear):
            mat[name+".weight"]=torch.zeros(
                module.in_features,
                module.in_features,
                dtype=torch.float32,
                device=device
            )
    return mat

def register_mat_hooks(model,mcs_mat_dict,device="cpu"):
    handles=[]
    def make_hook(weight_name):
        def simulated_mcs_hook(module,input,output):
            X=input[0].reshape(-1,input[0].shape[-1])
            X=X.detach().to(torch.float32)
            mcs_mat_dict[weight_name]+=(X.T@X).cpu()
        return simulated_mcs_hook
    
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear):
            weight_name=name+".weight"
            handle=module.register_forward_hook(make_hook(weight_name))
            handles.append(handle)
    return handles

def get_mcs_mat(model):
    sample_cnt=1
    input_device = model.model.device
    
    args,trainloader,calibset=mcs.get_calib()
    calibset=torch.concat(calibset,dim=0).to(input_device)
    calibset=calibset[:1].to(input_device)

    mcs_mats=init_imp_mat(model)

    handles=register_mat_hooks(model,mcs_mats)

    for batchset in calibset:
        model.eval()
        batchset=batchset.unsqueeze(0)
        print(f"Start Calculating {sample_cnt}th sample")
        sample_cnt+=1
        with torch.no_grad():
            model.model(
                input_ids=batchset,
                use_cache=False,
                last_logits_only=True,
            )

    for handle in handles:
        handle.remove()

    for name in mcs_mats.keys():
        mcs_mats[name]/=(args.num_steps+1)

    return mcs_mats

def get_imp_mat(model,gamma=0.01):
    mcs_mats=get_mcs_mat(model)
    imp_mats=dict.fromkeys(mcs_mats.keys())
    weight_map=load_weight_map()
    for weight_name in imp_mats.keys():
        d_mat=torch.inverse(mcs_mats[weight_name]+gamma*torch.eye(mcs_mats[weight_name].shape[0]))
        d_inv=1/torch.diag(d_mat)
        target_weight=load_weight_from_checkpoint(weight_name,weight_map)
        imp_mats[weight_name]=(target_weight.to("cpu")*d_inv.unsqueeze(0))**2
    return imp_mats


# for name, module in model.named_modules():
#     print("name=",name)
#     if isinstance(module, nn.Linear):
#         print(
#             "in_features=",module.in_features,
#             "out_features=",module.out_features,
#             "weight_shape=",tuple(module.weight.shape)
#         )
# print(model.named_modules())
# print(model.model.transformer.blocks[0].attn_norm.weight.shape)
# print(model.model.transformer.blocks[0].attn_norm.__dict__)