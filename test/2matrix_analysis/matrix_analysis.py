import yaml
import sys
import os
import pickle
import torch 
import numpy as np
from utils.model_utils import load_model
# import umap
import umap.umap_ as umap
from test.step_analysis.step_analysis import load_data,step_info, get_umap
from collections import defaultdict

def get_matrix_info(matrix):
    det = np.linalg.det(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    rank = np.linalg.matrix_rank(matrix)
    U, S, Vt = np.linalg.svd(matrix)
    is_projection = np.allclose(np.dot(matrix, matrix), matrix)
    return det, eigenvalues, rank,  S,  is_projection
    

if __name__ == '__main__':
    device = 'cpu' # 'cuda:0'
    save_path = os.path.join(sys.path[0], './results/')
    if not os.path.exists(save_path): os.makedirs(save_path)

    ## instruct
    models = [ 'meta-llama/Llama-2-7b-chat-hf','meta-llama/Llama-2-7b-hf',
                # 'Qwen/Qwen2-7B' , 'Qwen/Qwen2-7B-Instruct',
                # 'Qwen/Qwen2-1.5B', 'Qwen/Qwen2-1.5B-Instruct',
                # 'Qwen/Qwen2-1.5B',
                # 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf'
                ]
    # save_path_tmp = save_path
    # for model_name in models:
    #     p = model_name.split('/')[1]
    #     path = os.path.join(save_path_tmp, 'param', f'{p}.dat')
    #     if not os.path.exists(path): 
    #         if not os.path.exists(os.path.dirname(path)): 
    #             os.makedirs(os.path.dirname(path))
    #         param_info = defaultdict(list)
    #         cache_dir = os.path.join(f"./model_cache/{p}")
    #         model, tokenizer, model_config = load_model(model_name, device, cache_dir=cache_dir)
    #         model_modules = dict(model.named_modules())
    #         for layer_k in model_config['k_names']:
    #             if layer_k not in param_info:
    #                 param = model_modules[layer_k].weight.clone().detach().cpu().numpy()
    #                 det, eigenvalues, rank,  S,  is_projection = get_matrix_info(param)
    #                 # print(det, eigenvalues, rank, S, is_projection)
    #                 param_info[layer_k] = [det, eigenvalues, rank,  S,  is_projection]
    #                 with open(path, 'wb') as f:
    #                     pickle.dump(param_info, f)
    #     else:
    #         with open(path, 'rb') as f:
    #             param_info = pickle.load(f)
    

    # ## pythia
    # revisions = [0] + [int(2**i) for i in range(0, 10)]  +  list(range(1000, 5001, 1000)) +list(range(5000, 143000, 5000))
    # model_size = '2.8b' # '70m', '160m','410m','1b','1.4b' ,'6.9b'
    # save_path_tmp = os.path.join(save_path, 'param' , 'pythia', f'{model_size}.dat')
    # if not os.path.exists(os.path.dirname(save_path_tmp)): 
    #     os.makedirs(os.path.dirname(save_path_tmp))
    #     last_param_change_dist = defaultdict(list)
    # else:
    #     with open(save_path_tmp, 'rb') as f:
    #         last_param_change_dist = pickle.load(f)
    # model_name = f'EleutherAI/pythia-{model_size}'
    # last_param_list = []
    # for r_i, revision in enumerate(revisions):
    #     if len(last_param_change_dist.values()) == 0 or r_i >= len(list(last_param_change_dist.values())[0]) :
    #         cache_dir = os.path.join(f"./model_cache/{model_name}/{revision}")
    #         model, tokenizer, model_config = load_model(model_name, device, cache_dir=cache_dir, revision=revision)
    #         model_modules = dict(model.named_modules())
    #         for i, layer_k in enumerate(model_config['k_names']):
    #             param = model_modules[layer_k]
    #             if len(last_param_list) < len(model_config['k_names']):
    #                 last_param_list.append(param.weight.clone().detach().cpu().numpy())
    #             else:
    #                 param = param.weight.clone().detach().cpu().numpy()
    #                 change = np.linalg.norm(param - last_param_list[i])
    #                 last_param_list[i] = param
    #                 last_param_change_dist[layer_k].append(change)
    #         print(last_param_change_dist)
    #         with open(save_path_tmp, 'wb') as f:
    #             pickle.dump(last_param_change_dist, f)

    
                
            