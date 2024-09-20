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

class step_info:
    def __init__(self,hidden_states,key,values,ground_attentions) -> None:
        self.hidden_states=hidden_states
        self.key=key
        self.values=values
        self.ground_attentions=ground_attentions

def load_data(save_path):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
        return data
    
save_path = os.path.join('./results/')
if not os.path.exists(save_path): os.makedirs(save_path)

def get_diff(whole_diff, diff, alpha=0.2):
    if whole_diff is not None:
        whole_diff = whole_diff + np.where(diff<alpha, 1-diff, 0)
    else: 
        whole_diff = np.where(diff<alpha, diff, 0)
    return whole_diff
## instruct
models = ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-7b-hf', 
        # 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf',
        # 'Qwen/Qwen2-7B' , 'Qwen/Qwen2-7B-Instruct',
        # 'Qwen/Qwen2-1.5B', 'Qwen/Qwen2-1.5B-Instruct'
        ]
save_path_tmp = save_path
for model_name in models:
    p = model_name.split('/')[1]
    if os.path.exists(os.path.join(save_path_tmp, f'{p}.dat')):
        data = load_data(os.path.join(save_path_tmp, f'{p}.dat'))
        attn = data['long'].ground_attentions
        for layer_id in range(attn.shape[0]):
            head_attn = attn[layer_id]
            layer_diff = None
            num_tokens = head_attn.shape[1]
            for token_id in range(1, num_tokens): 
                last_attn = head_attn[:, token_id-1, :token_id]
                diff = np.sum(np.abs(last_attn[:,np.newaxis,:] - last_attn[np.newaxis,:,:]),axis=-1)
                layer_diff = get_diff(layer_diff,diff)
            for head_id in range(layer_diff.shape[0]):
                value = layer_diff[head_id] / (num_tokens-2)
                print(f'{p}, layer:{layer_id}', f'head:{head_id}', np.where(value>0.9))