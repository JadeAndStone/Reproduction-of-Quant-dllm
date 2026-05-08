from utils.generate_imp_mat import get_imp_mat

def ABMP(model):
    imp_mat=get_imp_mat(model)
    num_k=int(len(imp_mat)*0.05)
    
    imp_add_mat=dict.fromkeys(imp_mat)
    
    for weight_name in imp_mat:
        imp_add_mat[weight_name]=imp_mat[weight_name].sum()
        
    imp_add_mat=dict(sorted(imp_add_mat.items(),lambda x:x[1]))
    
    precisions=num_k*[1]+(len(imp_mat)-num_k*2)*[2]+num_k*[3]
    
    precision_mat=dict(zip(list(imp_add_mat.keys()),precisions))
    
    return precision_mat
    
    