import torch
from torchgen.packaged.autograd import context

from LLM.Attention_mechanism import W_query

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]
    ]
)

query = inputs[1]

attn_score_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_score_2[i] = torch.dot(query, x_i)

attn_weight_2 = torch.softmax(attn_score_2, dim=0)

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2[i] * x_i

x_i = inputs[1]
dim_in = inputs.shape[-1]
dim_out = 8

W_query = torch.nn.Parameter(torch.randn(dim_in, dim_out))
W_value = torch.nn.Parameter(torch.randn(dim_in, dim_out))
W_key = torch.nn.Parameter(torch.randn(dim_in, dim_out))

query_2 = x_i @ W_query
key_2 = x_i @ W_key
value_2 = x_i @ W_value

keys = inputs @ W_key
values = inputs @ W_value

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)

attn_score_2 = query_2 @ keys.T

dim_keys = keys.shape[-1]
attn_weight_2 = torch.softmax(attn_score_2 / dim_keys**0.5, dim=-1)

context_vec_2 = attn_weight_2 @ values
print(context_vec_2)

