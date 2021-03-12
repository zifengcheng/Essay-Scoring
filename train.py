import os
import sys
import argparse
import random
import time
import numpy as np
from utils import *
from metrics import *
from utils import rescale_tointscore
from utils import domain_specific_rescale
import data_prepare
from hierarchical_att_model import HierAttNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as Data
from reader import *

logger = get_logger("Train sentence sequences Recurrent Convolutional model (LSTM stack over CNN)")
np.random.seed(100)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
    parser.add_argument('--embedding', type=str, default='word2vec', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dict', type=str, default=None, help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Only useful when embedding is randomly initialised')

    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")

    parser.add_argument('--oov', choices=['random', 'embedding'], help="Embedding for oov word", required=True)

    # parser.add_argument('--project_hiddensize', type=int, default=100, help='num of units in projection layer')
    parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')

    parser.add_argument('--datapath',type =str,default='data/fold_')  # "data/word-level/*.train"
    parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')


    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    count = []
    
    for epoch in range(5):
        datapaths = [args.datapath+str(epoch)+'/train.tsv', args.datapath+str(epoch)+'/dev.tsv', args.datapath+str(epoch)+'/test.tsv']

        embedding_path = args.embedding_dict
        oov = args.oov
        embedding = args.embedding
        embedd_dim = args.embedding_dim
        prompt_id = args.prompt_id

        vocab = create_vocab(datapaths[0],prompt_id,0,True,True)
        (X_train, Y_train, mask_train,train_pmt), (X_dev, Y_dev, mask_dev,dev_pmt), (X_test, Y_test, mask_test,test_pmt), \
                     embed_table, overal_maxlen, overal_maxnum, init_mean_value = prepare_sentence_data(datapaths, vocab,\
                    embedding_path, embedding, embedd_dim, prompt_id, tokenize_text=True, \
                    to_lower=True, sort_by_len=False,  score_index=6)        
        max_sentnum = overal_maxnum
        max_sentlen = overal_maxlen

        Y_train= torch.tensor(Y_train)
        Y_dev = torch.tensor(Y_dev)
        Y_test= torch.tensor(Y_test)

        X_train= torch.LongTensor(X_train)
        X_dev = torch.LongTensor(X_dev)
        X_test= torch.LongTensor(X_test)


        train_data = Data.TensorDataset(X_train, Y_train)
        dev_data = Data.TensorDataset(X_dev, Y_dev)
        test_data = Data.TensorDataset(X_test,Y_test)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)
        dev_loader = Data.DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True, num_workers=3)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=3)

        model = HierAttNet(100,100,10,embed_table,max_sentnum,max_sentlen)
        model.word_att_net.lookup.weight.requires_grad = True

        if torch.cuda.is_available():
            model.cuda()
        print(model)
        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, alpha=0.9)
        best_loss = 1e5
        best_epoch = 0
        model.train()
        p = 0
        num_iter_per_epoch = len(train_loader)
        for epoch in range(args.num_epochs):
            print("begin train")
            for iter, (feature, label) in enumerate(train_loader):
                #print(len_train,label)
                if torch.cuda.is_available():
                    feature = feature.cuda()
                    label = label.cuda()
                optimizer.zero_grad()
                model._init_hidden_state()
                predictions= model(feature)
                loss = criterion(predictions, label)
                loss.backward()
                optimizer.step()
            print("loss:", loss)
            if epoch >= 0:
                model.eval()
                loss_ls = []
                te_label_ls = []
                te_pred_ls = []
                for te_feature, te_label in dev_loader:
                    num_sample = len(te_label)
                    if torch.cuda.is_available():
                        te_feature = te_feature.cuda()
                        te_label = te_label.cuda()
                    with torch.no_grad():
                        model._init_hidden_state(num_sample)
                        te_predictions = model(te_feature)
                    te_loss = criterion(te_predictions, te_label)
                    loss_ls.append(te_loss * num_sample)
                    te_label_ls.extend(te_label.clone().cpu())
                    te_pred_ls.extend(te_predictions.clone().cpu())
            te_label = np.array(te_label_ls)
            predictions = convert_to_dataset_friendly_scores(np.array(te_pred_ls), prompt_id)
            q1 = quadratic_weighted_kappa(predictions, te_label)
            p1 = pearson(predictions, te_label)
            s1 = spearman(predictions, te_label)

            print(
            "dev  Epoch: {}/{}, Iteration: {}/{}, loss : {},quadratic_weighted_kappa: {}, pearson: {}, spearman: {}".format(
                epoch + 1,
                20,
                iter + 1,
                num_iter_per_epoch, sum(loss_ls),
                q1, p1, s1))

            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_loader:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions= model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.extend(te_predictions.clone().cpu())

            te_label = np.array(te_label_ls)
            predictions = convert_to_dataset_friendly_scores(np.array(te_pred_ls), prompt_id)
            q2 = quadratic_weighted_kappa(predictions, te_label)
            p2 = pearson(predictions, te_label)
            s2 = spearman(predictions, te_label)

            print(
            "test  Epoch: {}/{}, Iteration: {}/{}, loss: {},quadratic_weighted_kappa: {}, pearson: {}, spearman: {}".format(
                epoch + 1,20,iter + 1,num_iter_per_epoch, sum(loss_ls),q2, s2, p2))

            if q1 > p:
                p = q1
                q3 = q2
                p3 = p2
                s3 = s2
                torch.save(model, 'net.pkl')
                print("best result Epoch : {},quadratic_weighted_kappa: {}, pearson: {}, spearman: {}".format(epoch + 1, q2, p2,s2))
            model.train()
        print("best result Epoch : {},quadratic_weighted_kappa: {}, pearson: {}, spearman: {}".format(epoch + 1, q3, p3,s3))
        count.append(q3)
    cc = 0
    for i in count:
        cc += i
    print('mean qwk is ',cc/len(count))


if __name__ == '__main__':
    main()

