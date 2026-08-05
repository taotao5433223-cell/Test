# 文本——>分词——>数据处理（滑动窗口）——>嵌入向量——>位置嵌入——>最终输入向量

import torch
import torch.nn as nn
from modelscope import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


with open('eaasy.txt', 'r', encoding='utf-8') as f:
    text = f.read()


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.inputs = []
        self.targets = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            self.inputs.append(torch.tensor(token_ids[i: i + max_length]))
            self.targets.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def data_loader(text, batch_size, max_length, stride, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


batch_size = 8
max_length = 8
stride = 4
vocab_size = 151643
dim_output = 256
token_embedding_layer = nn.Embedding(vocab_size, dim_output)
dataloader = data_loader(text, batch_size, max_length, stride)

iter_text = iter(dataloader)
inputs, targets = next(iter_text)
token_embeddings = token_embedding_layer(inputs)

pos_embedding_layer = nn.Embedding(max_length, dim_output)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
inputs_embeddings = token_embeddings + pos_embeddings
print(inputs_embeddings.shape)

