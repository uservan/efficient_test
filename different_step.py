import yaml
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))

from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
from datasets import load_dataset
import torch 
import numpy as np
from utils.trace_utils import TraceDict2
from utils.model_utils import try_hook_ground


def exploit_data(model,tokenizer,model_config):
    with torch.inference_mode():
        # 'attn sink collection'
        prompt = 'Beats Music is owned by'
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
        ground_attentions = torch.cat(output_and_cache.attentions, dim=0).detach().cpu().numpy()
        # k,q,v,hidden states collection
        prompt = 'The concept of artificial intelligence (AI) has captured the imagination of scientists, engineers, and enthusiasts alike for decades. AI refers to the simulation of human intelligence in machines, enabling them to perform tasks that typically require human cognition, such as learning, problem-solving, and decision-making. Over the years, AI has evolved from simple rule-based systems to complex neural networks capable of remarkable feats, including natural language processing, image recognition, and autonomous driving. One of the most significant advancements in AI has been the development of deep learning, a subfield of machine learning that uses artificial neural networks with many layers to learn from vast amounts of data. Deep learning algorithms have demonstrated unprecedented accuracy in various domains, revolutionizing industries such as healthcare, finance, and transportation. However, despite its tremendous potential, AI also raises ethical concerns and challenges. Issues related to bias in algorithms, data privacy, job displacement, and the existential risk posed by superintelligent'
        info = try_hook_ground(model, tokenizer, model_config, prompt, device)
        return {'info':info, 'attn_sink':ground_attentions}


def save_data(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

device = 'cuda:0'
save_path = os.path.join(sys.path[0], './results/')
if not os.path.exists(save_path): os.makedirs(save_path)

## pythia
save_path_tmp = os.path.join(save_path, 'pythia')
if not os.path.exists(save_path_tmp): os.makedirs(save_path_tmp)
model_size_list = ['70m', '160m','410m','1b','1.4b', '2.8b','6.9b'] 
revisions = [0] + [int(2**i) for i in range(0, 10)]  + list(range(1000, 143000, 1000))
for model_size in model_size_list:
    model_name = f'EleutherAI/pythia-{model_size}'
    for revision in revisions:
        if not os.path.exists(os.path.join(save_path_tmp, f'{revision}.dat')):
            cache_dir = os.path.join(f"./model_cache/{model_name}/{revision}")
            model = AutoModelForCausalLM.from_pretrained(model_name,revision=f"step{revision}",cache_dir=cache_dir).to(device)
            model_config = model.config
            tokenizer = AutoTokenizer.from_pretrained(model_name,revision=f"step{revision}",cache_dir=cache_dir)
            result = exploit_data(model, tokenizer, model.config)
            save_data(result, os.path.join(save_path_tmp, revision))

## instruct
models = ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-7b-chat-hf',
          'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf',
          'Qwen/Qwen2-7B' , 'Qwen/Qwen2-7B-Instruct',
          'Qwen/Qwen-14B', 'Qwen/Qwen-14B-Chat']
save_path_tmp = save_path
for model_name in models:
    p = model_name.split('/')[1]
    if not os.path.exists(os.path.join(save_path_tmp, f'{p}.dat')):
        cache_dir = os.path.join(f"./model_cache/{model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name,revision=f"step{revision}",cache_dir=cache_dir).to(device)
        model_config = model.config
        tokenizer = AutoTokenizer.from_pretrained(model_name,revision=f"step{revision}",cache_dir=cache_dir)
        result = exploit_data(model, tokenizer, model.config)
        save_data(result, os.path.join(save_path_tmp, revision))
