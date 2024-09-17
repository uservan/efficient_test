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
from step_analysis import load_data,step_info, get_umap
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

if __name__ == '__main__':
    device = 'cpu' #'cuda:0'
    save_path = os.path.join(sys.path[0], './results/')
    if not os.path.exists(save_path): os.makedirs(save_path)

    ## instruct
    models = [ 'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-7b-hf',
                # 'Qwen/Qwen2-7B' , 'Qwen/Qwen2-7B-Instruct',
                # 'Qwen/Qwen2-1.5B', 'Qwen/Qwen2-1.5B-Instruct',
                # 'Qwen/Qwen2-1.5B',
                # 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf'
                ]
    save_path_tmp = save_path
    for model_name in models:
        p = model_name.split('/')[1]
        path = os.path.join(save_path_tmp, 'eval', f'{p}.dat')
        if not os.path.exists(os.path.dirname(path)): 
            os.makedirs(os.path.dirname(path))
        cache_dir = os.path.join(f"./model_cache/{p}")
        model, tokenizer, model_config = load_model(model_name, device, cache_dir=cache_dir)
        result = lm_evaluate(model=model, batch_size=64)
        with open(path, 'rb') as f:
            pickle.dump(result, f)
