import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WordAttNet(nn.Module):
    def __init__(self, dict,hidden_size=100):
        super(WordAttNet, self).__init__()
        dict = torch.from_numpy(dict.astype(np.float))
        self.lookup = nn.Embedding(num_embeddings=4000, embedding_dim=50).from_pretrained(dict)
        self.conv1 = nn.Conv1d(in_channels=50,out_channels=100,kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear( 100,100)
        self.fc2 = nn.Linear( 100 , 1,bias =False)


    def forward(self, input):
        output = self.lookup(input)
        output = self.dropout(output)
        output = output.permute(1,2,0)
        f_output = self.conv1(output.float()) # shape : batch * hidden_size * seq_len
        f_output = f_output.permute(2,0,1)   # shape : seq_len * batch * hidden_size

        weight = torch.tanh(self.fc1(f_output))
        weight = self.fc2(weight)
        weight = F.softmax(weight,0)
        weight = weight * f_output
        output = weight.sum(0).unsqueeze(0)  # 1 * batch * hidden_size
        return output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
