# CNN-LSTM-ATT model for essay scoring
This is a Pytorch implementations for paper Attention-based Recurrent Convolutional Neural Network for Automatic Essay Scoring.
[[pdf](https://www.aclweb.org/anthology/K17-1017.pdf)]


# Training
python train.py --oov embedding --embedding glove --embedding_dict glove.6B.50d.txt --embedding_dim 50 --datapath data/fold_ --prompt_id 1
