# from datasets import load_dataset

# mmlu_cs_dataset = load_dataset("cais/mmlu", 'all', split="test", cache_dir='./model_cache/mmlu')

# print(mmlu_cs_dataset)

import yaml
import sys
import os
import pickle
import torch 
import numpy as np
from utils.model_utils import load_model

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

# pythia
save_path_tmp = os.path.join(save_path, 'pythia')
if not os.path.exists(save_path_tmp): os.makedirs(save_path_tmp)
model_size_list = ['70m','1.4b', '160m','410m','1b','2.8b' ]  # ,'6.9b'
revisions = [0] + [int(2**i) for i in range(0, 10)]  + list(range(1000, 143000, 5000))
for model_size in model_size_list:
    model_name = f'EleutherAI/pythia-{model_size}'
    for revision in revisions:
        if os.path.exists(os.path.join(save_path_tmp,model_size, f'{revision}.dat')):
            data = load_data(os.path.join(save_path_tmp,model_size, f'{revision}.dat'))
            print(data['short'].hidden_states.shape)
            attn_sink = data['short'].ground_attentions[:,:,-1]
            attn_sink_mean = np.mean(attn_sink.reshape(-1, attn_sink.shape[-1]), axis=0)
            attn_sink_percent = attn_sink_mean[0] / np.sum(attn_sink_mean)
            print(model_name, revision, attn_sink_percent)

## instruct
models = ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-7b-chat-hf',
          'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf',
          'Qwen/Qwen2-7B' , 'Qwen/Qwen2-7B-Instruct',
          'Qwen/Qwen2-1.5B', 'Qwen/Qwen2-1.5B-Instruct']
save_path_tmp = save_path
for model_name in models:
    p = model_name.split('/')[1]
    if os.path.exists(os.path.join(save_path_tmp, f'{p}.dat')):
        data = load_data(os.path.join(save_path_tmp, f'{p}.dat'))
        print(data['short'].hidden_states.shape)

