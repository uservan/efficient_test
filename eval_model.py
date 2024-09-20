import lm_eval
import yaml
import sys
import os
import pickle
import torch 
import numpy as np
from utils.model_utils import load_model
# import umap
import umap.umap_ as umap
from collections import defaultdict

def lm_evaluate(model, tasks=["mmlu"], batch_size=1, num_fewshot=0):
    lm = lm_eval.api.registry.get_model('hf').create_from_arg_obj({'pretrained': model})
    task_manager = lm_eval.tasks.TaskManager()
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        device=model.device,
        task_manager=task_manager,
    )
    return results

def change_model(model, weight_names):
    model_modules = dict(model.named_modules())
    for layer_k in weight_names:
        param = model_modules[layer_k].weight
        U, S, Vh = torch.linalg.svd(param, full_matrices=False)
        S[S < 1] = 0
        S_matrix = torch.diag(S)
        param_reconstructed = torch.mm(U, torch.mm(S_matrix, Vh)).to(model.device)
        model_modules[layer_k].weight = torch.nn.Parameter(param_reconstructed)

if __name__ == '__main__':
    device = 'cuda:0'
    save_path = os.path.join(sys.path[0], 'results')
    if not os.path.exists(save_path): os.makedirs(save_path)

    ## instruct
    models = ['Qwen/Qwen2-1.5B', 'meta-llama/Llama-2-7b-chat-hf', # 'meta-llama/Llama-2-7b-hf',
                # 'Qwen/Qwen2-7B' , 'Qwen/Qwen2-7B-Instruct',
                #  'Qwen/Qwen2-1.5B-Instruct',
                # 
                # 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf'
                ]
    save_path_tmp = save_path
    for model_name in models:
        p = model_name.split('/')[1]
        cache_dir = os.path.join(f"./model_cache/{p}")
        model, tokenizer, model_config = load_model(model_name, device, cache_dir=cache_dir)
        model = change_model(model, model_config['k_names']+model_config['q_names'])
        result = lm_evaluate(model=model, batch_size=128)
        path = os.path.join(save_path_tmp, 'eval', f'{p}_kq.dat')
        if not os.path.exists(os.path.dirname(path)): 
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(result, f)

    # models = ['meta-llama/Llama-2-7b-chat-hf','meta-llama/Llama-2-7b-hf',
    #                 # 'Qwen/Qwen2-7B' , 'Qwen/Qwen2-7B-Instruct',
    #                 # 'Qwen/Qwen2-1.5B', 'Qwen/Qwen2-1.5B-Instruct',
    #                 # 'Qwen/Qwen2-1.5B',
    #                 # 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf'
    #                 ]
    # save_path_tmp = save_path
    # for model_name in models:
    #     p = model_name.split('/')[1]
    #     path = os.path.join(save_path_tmp, 'eval', f'{p}.dat')
    #     with open(path, 'rb') as f:
    #         result = pickle.load(f)
    #         print(result)
