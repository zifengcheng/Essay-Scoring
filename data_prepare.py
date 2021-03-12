import reader
import utils
import numpy as np

logger = utils.get_logger("Prepare data ...")


def prepare_sentence_data(datapaths, embedding_path=None, embedding='word2vec', embedd_dim=100, prompt_id=1, vocab_size=0, tokenize_text=True, \
                         to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):

    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, train_y, train_prompts), (dev_x, dev_y, dev_prompts), (test_x, test_y, test_prompts), vocab, overal_maxlen, overal_maxnum = \
        reader.get_data(datapaths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overal_maxnum, overal_maxlen, post_padding=True)

    if prompt_id:
        train_pmt = np.array(train_prompts, dtype='int32')
        dev_pmt = np.array(dev_prompts, dtype='int32')
        test_pmt = np.array(test_prompts, dtype='int32')

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)


    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    Y_train = reader.get_model_friendly_scores(y_train, train_prompts)
    Y_dev = np.array(y_dev)
    Y_test = np.array(y_test)
    #Y_dev = reader.get_model_friendly_scores(y_dev, dev_prompts)
    #Y_test = reader.get_model_friendly_scores(y_test, test_prompts)
    scaled_train_mean = Y_train.mean(axis=0)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))

    if embedding_path:
        embedd_dict, embedd_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = utils.build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    return (X_train, Y_train, mask_train,train_prompts), (X_dev, Y_dev, mask_dev,dev_prompts), (X_test, Y_test, mask_test,test_prompts), \
            vocab, len(vocab), embedd_matrix, overal_maxlen, overal_maxnum, scaled_train_mean