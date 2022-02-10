import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import argparse
import random
import numpy as np
import time
import logging
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from util import custom_f1, fasta_data_loader

def cnn_model():

    model = ks.Sequential()

    model.add(ks.layers.Conv1D(
                input_shape = (X_train.shape[1], X_train.shape[2]),
                filters = 90,
                kernel_size = 19,
                padding = 'same',
                activation = ks.activations.relu))

    model.add(ks.layers.Dropout(0.5, input_shape = (X_train.shape[1], 30)))
    model.add(ks.layers.MaxPool1D(
                pool_size = 10,
                strides= 10))

    model.add(ks.layers.Conv1D(
                filters = 128,
                kernel_size= 5,
                padding= 'same',
                activation= ks.activations.relu))

    model.add(ks.layers.Dropout(0.5, input_shape = (X_train.shape[1], 128)))

    model.add(ks.layers.MaxPool1D(
                pool_size= 10,
                strides= 10))

    model.add(ks.layers.Flatten())

    model.add(ks.layers.Dense(
                units= 512,
                activation=ks.activations.relu))

    model.add(ks.layers.Dropout(0.5, input_shape = (512,)))
    
    model.add(ks.layers.Dense(
                units= 1,
                activation=ks.activations.sigmoid,
                name = 'visualized_layer'))

    model.compile(
                optimizer=ks.optimizers.Adam(learning_rate= 0.0003,decay=1e-6),
                loss = ks.losses.BinaryCrossentropy(),
                metrics= [custom_f1, 'accuracy'])

    return model


def cv_run(X_train, y_train, X_test, y_test, n_splits, seed, epochs, batch_size):

    tf.random.set_seed(seed)
    np.random.seed(seed)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_f1_scores = []
    val_f1_scores = []
    test_f1_scores = []
    test_confusions = []

    t1= time.time()
    for k, (tr_inds, val_inds) in enumerate(kfold.split(X_train, y_train)):

        print('\n---- Fold {} ----'.format(k+1))
        print('{} training, {} validation'.format(len(tr_inds), len(val_inds)))

        model = cnn_model()
        X_tr, y_tr = X_train[tr_inds], y_train[tr_inds]
        X_val, y_val = X_train[val_inds], y_train[val_inds]

        model.fit(x=X_tr, y=y_tr, batch_size=batch_size, epochs=epochs, verbose=0)

        y_tr_pred = model.predict(X_tr)
        y_tr_pred_cat = (np.asarray(y_tr_pred)).round()

        y_val_pred = model.predict(X_val)
        y_val_pred_cat = (np.asarray(y_val_pred)).round()

        ### Get performance metrics after each fold
        tr_f1 = f1_score(y_tr, y_tr_pred_cat)
        print("Train F1 = {:.5f}".format(tr_f1))
        tr_f1_scores.append(tr_f1)

        val_f1 = f1_score(y_val, y_val_pred_cat)
        print("Valid F1 = {:.5f}".format(val_f1))
        val_f1_scores.append(val_f1)

        ### Run testing after each fold
        y_test_pred = model.predict(X_test, batch_size=batch_size).round()
        confusion = confusion_matrix(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_f1_scores.append(test_f1)
        test_confusions.append(confusion.tolist())
        print('Test F1 = {:.5f}'.format(test_f1))
        print('Test confusion = {}'.format(confusion.tolist()))

    t2= time.time()
    total_exec_time = t2 - t1
    median_tr_f1 = np.median(tr_f1_scores)
    median_val_f1 = np.median(val_f1_scores)
    median_test_f1 = np.median(test_f1_scores)
    median_test_conf = np.median(test_confusions, axis=0)

    print('{}\nTotal Execution Time = {:.5f} seconds'.format('-'*20, total_exec_time))
    print('Median Train F1 score = {:.5f}'.format(median_tr_f1))
    print('Median Val F1 score = {:.5f}'.format(median_val_f1))
    print('Median Test F1 score = {:.5f}'.format(median_test_f1))
    print('Median Test Confusion = {}\n{}\n'.format(np.median(test_confusions, axis=0).tolist(), '-'*20))

    stats = {
        'time' : total_exec_time,
        'tr_f1': median_tr_f1,
        'val_f1': median_val_f1,
        'test_f1': median_test_f1,
        'test_conf': median_test_conf,
        'test_conf_all': test_confusions
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert4epi')
    parser.add_argument('--cell_line', default='GM12878', type=str) # GM12878, HUVEC, HeLa-S3, K562, combined
    parser.add_argument('--cross_cell_line', default=None, type=str) # GM12878, HUVEC, HeLa-S3, K562
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    random.seed(args.seed)

    if args.cell_line == 'US_UU':
        train, test = fasta_data_loader(pro_fa='data/US_UU/K562_US.fa',
                                        enh_fa='data/US_UU/K562_UU.fa', seed=args.seed)
    elif args.cell_line == 'CAGE':
        train, test = fasta_data_loader(pro_fa='data/CAGE/fantom_promoters_600.fa',
                                        enh_fa='data/CAGE/fantom_enhancers_600.fa', seed=args.seed)
    else:
        train, test = fasta_data_loader(pro_fa='data/{}/enhancers.fa'.format(args.cell_line),
                                        enh_fa='data/{}/promoters.fa'.format(args.cell_line), seed=args.seed)

        if args.cross_cell_line == None or (args.cell_line == args.cross_cell_line):
            print('\nTESTING ON SAME CELL-LINE ({})'.format(args.cell_line))
        else:
            # Overwrite test data as 20% of cross cell-line
            print('\nTESTING ON CROSS CELL-LINE ({})'.format(args.cross_cell_line))
            _, test = fasta_data_loader(pro_fa='data/{}/enhancers.fa'.format(args.cross_cell_line),
                                        enh_fa='data/{}/promoters.fa'.format(args.cross_cell_line), seed=args.seed)

    # Reshape the data to (n_samples, n_seqs, n_channels)
    X_train, y_train = train[0].transpose([0,2,1]), train[1]
    X_test, y_test = test[0].transpose([0,2,1]), test[1]

    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("y_test shape: " + str(y_test.shape))

    EPOCHS = 40
    BATCH_SIZE = 30
    N_SPLITS = 5

    val_size = int(X_train.shape[0]*(1/N_SPLITS))
    train_size = X_train.shape[0] - val_size
    test_size = X_test.shape[0]

    stats = cv_run(X_train, y_train, X_test, y_test, n_splits=N_SPLITS,
                   seed=args.seed, epochs=EPOCHS, batch_size=BATCH_SIZE)

    ### LOGS

    log_dir = "results/{}".format(args.seed)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if args.cross_cell_line == None:
        log_name = '{}/{}'.format(log_dir, args.cell_line)
    else:
        log_name = '{}/{}'.format(log_dir, args.cell_line + '_' + args.cross_cell_line)

    log_file = "{}.txt".format(log_name)
    open(log_file, 'w').close() # clear file content
    logging.basicConfig(format='%(message)s', filename=log_file,level=logging.DEBUG)
    logging.info("Cell-line                  = {}".format(args.cell_line))
    logging.info("Cross Cell-line            = {}".format(args.cross_cell_line))
    logging.info("Random seed                = {}".format(args.seed))
    logging.info("Total size                 = {}".format(train_size + val_size + test_size))
    logging.info("Training size              = {}".format(train_size))
    logging.info("Validation size            = {}".format(val_size))
    logging.info("Test size                  = {}".format(test_size))
    logging.info("Train epochs               = {}".format(EPOCHS))
    logging.info("Train batch size           = {}".format(BATCH_SIZE))
    logging.info("Total Execution Time       = {:.5f}".format(stats['time']))
    logging.info("Median Train F1            = {:.5f}".format(stats['tr_f1']))
    logging.info("Median Validation F1       = {:.5f}".format(stats['val_f1']))
    logging.info("Median Test F1             = {:.5f}".format(stats['test_f1']))
    test_cm = stats['test_conf_all']
    for i, cm in enumerate(test_cm):
        logging.info("Test Confusions (Fold {})   = {}".format(i+1, cm))
    logging.info("Median Test Confusion      = {}".format(stats['test_conf'].tolist()))
