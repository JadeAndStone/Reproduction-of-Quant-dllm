from utils.generate_imp_mat import get_imp_mat

def ABMP(model,model_path,nsamples,
        seed,seqlen,
        num_steps,gamma,
        group_size,schedule,
        max_docs,source,allocate_ratio,damping,imp_mats)->dict[str,int]:

    num_k=int(len(imp_mats)*allocate_ratio)
    
    imp_add_mat=dict.fromkeys(imp_mats)
    
    for weight_name in imp_mats:
        imp_add_mat[weight_name]=imp_mats[weight_name].sum().item()
        
    imp_add_mat=dict(sorted(imp_add_mat.items(),key=lambda x:x[1]))
    
    precisions=num_k*[1]+(len(imp_mats)-num_k*2)*[2]+num_k*[3]
    
    precision_mat=dict(zip(list(imp_add_mat.keys()),precisions))
    
    return precision_mat
