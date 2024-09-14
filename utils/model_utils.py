from utils.trace_utils import TraceDict2
import torch
import numpy as np

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


def try_hook_ground(model, tokenizer, model_config, prompt, device, edit_input=None, edit_output=None):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with TraceDict2(model, layers=model_config['q_names']+model_config['out_proj']+model_config['fc_out']+model_config['fc_in'],
                    retain_output=True, retain_input=True,edit_input=edit_input, edit_output=edit_output) as ret:
        output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
        attn_output_list = [ret[q].output for q in model_config['out_proj']]
        attn_output = torch.cat(attn_output_list, dim=0).detach().cpu().numpy()
        attn_input_list = [ret[q].input for q in model_config['out_proj']]
        attn_input = torch.cat(attn_input_list, dim=0).detach().cpu().numpy()
        if model_config['result_dir'].find('opt') != -1:
            mlp_output_list = [torch.unsqueeze(ret[q].output, dim=0) for q in model_config['fc_out']]
            mlp_output = torch.cat(mlp_output_list, dim=0).detach().cpu().numpy()
            mlp_input_list = [torch.unsqueeze(ret[q].output, dim=0) for q in model_config['fc_in']]
            mlp_input = torch.cat(mlp_input_list, dim=0).detach().cpu().numpy()
        else:
            mlp_output_list = [ret[q].output for q in model_config['fc_out']]
            mlp_output = torch.cat(mlp_output_list, dim=0).detach().cpu().numpy()
            mlp_input_list = [ret[q].output for q in model_config['fc_in']]
            mlp_input = torch.cat(mlp_input_list, dim=0).detach().cpu().numpy()
        q_list = [ret[q].output for q in model_config['q_names']]
        past_qs = torch.cat(q_list, dim=0).detach().cpu().numpy()
        past_qs = np.transpose(np.reshape(past_qs,newshape=past_qs.shape[:-1]+(model_config['n_heads'], -1)),(0,2,1,3))
    if model_config['result_dir'].find('llama3') != -1:
        past_key = torch.cat([repeat_kv(key_values[0], model_config['n_heads']//key_values[0].shape[1]) for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
        past_values = torch.cat([repeat_kv(key_values[1], model_config['n_heads']//key_values[0].shape[1]) for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    else:
        past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
        past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
    hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
    return Info(attn_input ,attn_output,mlp_input,mlp_output, hidden_state, ground_attentions.detach().numpy(), past_key, past_values, past_qs)
