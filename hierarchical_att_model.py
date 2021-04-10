import torch
import torch.nn as nn
from sent_att_model import SentAttNet
from word_att_model import WordAttNet


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, embed_table,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(embed_table,word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):

        output_list = torch.empty(0,).cuda()
        input = input.permute(1, 0, 2)
        for i in input:
            output = self.word_att_net(i.permute(1, 0))
            output_list = torch.cat((output_list,output))
        output= self.sent_att_net(output_list)
        return output
