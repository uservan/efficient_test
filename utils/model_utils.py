# from utils.trace_utils import TraceDict2
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str, device='cuda', cache_dir='', low_cpu_mem_usage=False, show_params=True):

    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage, 
                                                 token='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',
                                                 cache_dir=cache_dir).to(device)
    device, dtype = model.device, model.dtype

    model_modules = dict(model.named_modules())

    tokenizer = AutoTokenizer.from_pretrained(model_name,token='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')
    tokenizer.pad_token = tokenizer.eos_token
  
    MODEL_CONFIG = model.config

    if model_name.find('qiwen') !=-1:
        MODEL_CONFIG = {    
                        'name':'qiwen',
                        "n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        'result_dir':'qiwen',
                        "blocks": [f'model.layers.{layer}' for layer in
                                            range(model.config.num_hidden_layers)],
                         'wte': ['lm_head']
                        }

    if model_name.find('pythia') != -1:
        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        'result_dir':'pythia',
                        "k_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,          
                        "v_names": [f'model.layers.{layer}.self_attn.v_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "q_names": [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "k_q_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "out_proj": [f'model.layers.{layer}.self_attn.dense' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_in": [f'model.layers.{layer}.mlp.fc1' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_out": [f'model.layers.{layer}.mlp.fc2' for layer in
                                            range(model.config.num_hidden_layers)],
                        "blocks": [f'model.layers.{layer}' for layer in
                                            range(model.config.num_hidden_layers)],
                        }

    if model_name.find('phi-2') != -1:
        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        'result_dir':'phi_2',
                        "k_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,          
                        "v_names": [f'model.layers.{layer}.self_attn.v_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "q_names": [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "k_q_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "out_proj": [f'model.layers.{layer}.self_attn.dense' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_in": [f'model.layers.{layer}.mlp.fc1' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_out": [f'model.layers.{layer}.mlp.fc2' for layer in
                                            range(model.config.num_hidden_layers)],
                        "blocks": [f'model.layers.{layer}' for layer in
                                            range(model.config.num_hidden_layers)],
                        }


    if model_name.find('gpt-neo') != -1:
        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        'result_dir':'gpt_neo',
                        "attn_hook_names": [f'transformer.h.{layer}.attn.attention.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        "k_names": [f'transformer.h.{layer}.attn.attention.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,          
                        "v_names": [f'transformer.h.{layer}.attn.attention.v_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "q_names": [f'transformer.h.{layer}.attn.attention.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "k_q_names": [f'transformer.h.{layer}.attn.attention.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'transformer.h.{layer}.attn.attention.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "out_proj": [f'transformer.h.{layer}.attn.attention.out_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_in": [f'transformer.h.{layer}.mlp.c_fc' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_out": [f'transformer.h.{layer}.mlp.c_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "blocks": [f'transformer.h.{layer}' for layer in
                                            range(model.config.num_hidden_layers)],
                        }

    if model_name.find('llama') != -1:
        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        'result_dir':'llama2_7b',
                        "k_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,          
                        "v_names": [f'model.layers.{layer}.self_attn.v_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "q_names": [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "k_q_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "out_proj": [f'model.layers.{layer}.self_attn.o_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_in": [f'model.layers.{layer}.mlp.up_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_out": [f'model.layers.{layer}.mlp.down_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "blocks": [f'model.layers.{layer}' for layer in
                                            range(model.config.num_hidden_layers)],
                        }
        if model_name.find('7b') != -1: MODEL_CONFIG['result_dir'] = 'llama2_7b'
        if model_name.find('8b') != -1: MODEL_CONFIG['result_dir'] = 'llama3_8b'
        if model_name.find('13b-chat') != -1: MODEL_CONFIG['result_dir'] = 'llama2_13b_chat'
    
    if model_name.find('gpt-j') !=-1:
        MODEL_CONFIG = {    
                        'name':'gptj',
                        "n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        'result_dir':'gpt_j',
                        "attn_hook_names": [f'transformer.h.{layer}.attn.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        "k_names": [f'transformer.h.{layer}.attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,          
                        "v_names": [f'transformer.h.{layer}.attn.v_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "q_names": [f'transformer.h.{layer}.attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "k_q_names": [f'transformer.h.{layer}.attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'transformer.h.{layer}.attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "out_proj": [f'transformer.h.{layer}.attn.out_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_out": [f'transformer.h.{layer}.mlp.fc_out' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_in": [f'transformer.h.{layer}.mlp.fc_in' for layer in
                                            range(model.config.num_hidden_layers)],
                        "blocks": [f'transformer.h.{layer}' for layer in
                                            range(model.config.num_hidden_layers)],
                        }
        
    if model_name.find('opt') !=-1:
        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        'result_dir':'llama2_7b',
                        "k_names": [f'model.decoder.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,          
                        "v_names": [f'model.decoder.layers.{layer}.self_attn.v_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "q_names": [f'model.decoder.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "k_q_names": [f'model.decoder.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'model.decoder.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "out_proj": [f'model.decoder.layers.{layer}.self_attn.out_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_in": [f'model.decoder.layers.{layer}.fc1' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_out": [f'model.decoder.layers.{layer}.fc2' for layer in
                                            range(model.config.num_hidden_layers)],
                        "blocks": [f'model.decoder.layers.{layer}' for layer in
                                            range(model.config.num_hidden_layers)],
                        }
        if model_name.find('opt-6.7b') != -1: MODEL_CONFIG['result_dir'] = 'opt_6b'
        if model_name.find('opt-1.3b') != -1: MODEL_CONFIG['result_dir'] = 'opt_1b'
        if model_name.find('opt-2.7b') != -1: MODEL_CONFIG['result_dir'] = 'opt_2b'
        

    if show_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # print(torch.isnan(param.grad).any())
                # print('name:{} param grad:{} param requires_grad:{},params:{}'.format(name, param.grad, param.requires_grad,param))
                print('name:{} param requires_grad:{}, detype:{}, device:{}'.format(name, param.requires_grad, param.dtype, param.device))

    return model, tokenizer, MODEL_CONFIG


class Info:
    def __init__(self, attn_input ,attn_output,mlp_input,mlp_output, hidden_state, ground_attentions, past_key, past_values, past_qs) -> None:
        self.attn_input ,self.attn_output,self.mlp_input,self.mlp_output, self.hidden_state,\
              self.ground_attentions, self.past_key, self.past_values, self.past_qs = attn_input \
                ,attn_output,mlp_input,mlp_output, hidden_state, ground_attentions, past_key, past_values, past_qs
    def get_info(self):
        return self.attn_input ,self.attn_output,self.mlp_input,self.mlp_output, self.hidden_state,\
              self.ground_attentions, self.past_key, self.past_values, self.past_qs

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# def try_hook_ground(model, tokenizer, model_config, prompt, device, edit_input=None, edit_output=None):
#     model.eval()
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with TraceDict2(model, layers=model_config['q_names']+model_config['out_proj']+model_config['fc_out']+model_config['fc_in'],
#                     retain_output=True, retain_input=True,edit_input=edit_input, edit_output=edit_output) as ret:
#         output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
#         attn_output_list = [ret[q].output for q in model_config['out_proj']]
#         attn_output = torch.cat(attn_output_list, dim=0).detach().cpu().numpy()
#         attn_input_list = [ret[q].input for q in model_config['out_proj']]
#         attn_input = torch.cat(attn_input_list, dim=0).detach().cpu().numpy()
#         if model_config['result_dir'].find('opt') != -1:
#             mlp_output_list = [torch.unsqueeze(ret[q].output, dim=0) for q in model_config['fc_out']]
#             mlp_output = torch.cat(mlp_output_list, dim=0).detach().cpu().numpy()
#             mlp_input_list = [torch.unsqueeze(ret[q].output, dim=0) for q in model_config['fc_in']]
#             mlp_input = torch.cat(mlp_input_list, dim=0).detach().cpu().numpy()
#         else:
#             mlp_output_list = [ret[q].output for q in model_config['fc_out']]
#             mlp_output = torch.cat(mlp_output_list, dim=0).detach().cpu().numpy()
#             mlp_input_list = [ret[q].output for q in model_config['fc_in']]
#             mlp_input = torch.cat(mlp_input_list, dim=0).detach().cpu().numpy()
#         q_list = [ret[q].output for q in model_config['q_names']]
#         past_qs = torch.cat(q_list, dim=0).detach().cpu().numpy()
#         past_qs = np.transpose(np.reshape(past_qs,newshape=past_qs.shape[:-1]+(model_config['n_heads'], -1)),(0,2,1,3))
#     if model_config['result_dir'].find('llama3') != -1:
#         past_key = torch.cat([repeat_kv(key_values[0], model_config['n_heads']//key_values[0].shape[1]) for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
#         past_values = torch.cat([repeat_kv(key_values[1], model_config['n_heads']//key_values[0].shape[1]) for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
#     else:
#         past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
#         past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
#     ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
#     hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
#     return Info(attn_input ,attn_output,mlp_input,mlp_output, hidden_state, ground_attentions.detach().numpy(), past_key, past_values, past_qs)
