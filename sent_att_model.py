import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.utils import matrix_mul, element_wise_mul

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=100, word_hidden_size=100, num_classes=14):
        super(SentAttNet, self).__init__()
        self.LSTM = nn.LSTM(word_hidden_size, sent_hidden_size)
        self.fc = nn.Linear( sent_hidden_size, num_classes)
        self.fc1 = nn.Linear( sent_hidden_size,sent_hidden_size)
        self.fc2 = nn.Linear( sent_hidden_size , 1,bias =False)

    def forward(self, input):
        f_output, _ = self.LSTM(input)
        weight = F.tanh(self.fc1(f_output))
        weight = self.fc2(weight)
        weight = F.softmax(weight,0)
        weight = weight * f_output
        output = weight.sum(0)
        output = torch.sigmoid(self.fc(output))
        return output


if __name__ == "__main__":
    abc = SentAttNet()
