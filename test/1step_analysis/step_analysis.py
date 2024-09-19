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
# import umap
import umap.umap_ as umap

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

def cosine_similarity(a, b):
    dot_product =np.diagonal(np.dot(a, b.T)).reshape(-1,1)
    norm_a = np.linalg.norm(a, axis=-1).reshape(-1,1)
    norm_b = np.linalg.norm(b, axis=-1).reshape(-1,1)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def BI_score(a, b):
    similarity = cosine_similarity(a, b)
    return 1-similarity

def cosine_similarity_matrix(a, b):
    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a, axis=-1).reshape(-1,1)
    norm_b = np.linalg.norm(b, axis=-1).reshape(-1,1)
    similarity = dot_product / (np.dot(norm_a, norm_b.T))
    return similarity

def get_umap(activations, dim=2):
    old_shape = activations.shape
    embedding = umap.UMAP(n_components = dim).fit_transform(activations.reshape(-1, old_shape[-1]))
    embedding = embedding.reshape(old_shape[:-1]+(dim,))
    return embedding

if __name__ == '__main__':
    save_path = os.path.join('./results/')
    if not os.path.exists(save_path): os.makedirs(save_path)

    # # pythia 
    # save_path_tmp = os.path.join(save_path, 'pythia')
    # if not os.path.exists(save_path_tmp): os.makedirs(save_path_tmp)
    # model_size_list = ['1b', '70m','410m', '1.4b', '160m','2.8b' ]  # ,'6.9b'
    # revisions = [0] + [int(2**i) for i in range(0, 10)]  + list(range(1000, 143000, 5000))
    # for model_size in model_size_list:
    #     model_name = f'EleutherAI/pythia-{model_size}'
    #     for revision in revisions:
    #         if os.path.exists(os.path.join(save_path_tmp,model_size, f'{revision}.dat')):
    #             data = load_data(os.path.join(save_path_tmp,model_size, f'{revision}.dat'))
    #             keys = data['long'].key
    #             key_list = []
    #             for layer in range(keys.shape[0]):
    #                 key_simi_list = []
    #                 for head in range(keys.shape[1]):
    #                     key_simi = cosine_similarity_matrix(keys[layer, head], keys[layer, head])
    #                     key_simi_list.append(key_simi.reshape(-1, key_simi.shape[0], key_simi.shape[1]))
    #                 key_simi_list = np.concatenate(key_simi_list, axis=0)
    #                 key_list.append(key_simi_list.reshape([-1]+list(key_simi_list.shape)))
    #             key_list = np.concatenate(key_list, axis=0)
    #             print(key_list)
    #         else:
    #             print(f'No data for {model_name}, {revision}')

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
            attn = data['long'].ground_attentions
            for layer_id in range(attn.shape[0]):
                head_attn = attn[layer_id]
                for head_id in range(head_attn.shape[0]-1):
                    print(f'layer:{layer_id}', f'head:{head_id}', np.linalg.norm(head_attn[head_id, :] - head_attn[head_id+1, :]))

