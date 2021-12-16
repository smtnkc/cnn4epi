import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from util import fasta_data_loader

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
                metrics= [ks.metrics.AUC()])

    return model


def cross_validation(seed):

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    total_auc = []
    model = cnn_model()

    for train, test in kfold.split(X_train, Y_train):
        history = model.fit(x=X_train[train],
                            y=Y_train[train],
                            validation_data = (X_train[test], Y_train[test]),
                            batch_size= 30,
                            epochs=40,
                            verbose=0)

        for key in history.history.keys():
            if key.startswith('auc_'):
                train_auc = history.history[key][-1]
            if key.startswith('val_auc_'):
                valid_auc = history.history[key][-1]

        _, test_auc = model.evaluate(X_test, Y_test, verbose = 0)
        mean_auc = (train_auc + valid_auc + test_auc)/3
        total_auc.append(mean_auc)

    return model, total_auc


def best_model(seed):

    model = cnn_model()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    i = 0
    tprs = []
    auc_total = []
    tprs_pred_total = []
    tprs_test_total = []

    mean_fpr = np.linspace(0, 1, 100)

    for train, test in tqdm(kfold.split(X_train, Y_train), total=kfold.get_n_splits(), desc="k-fold"):
        model.fit(
                x=X_train[train],
                y=Y_train[train],
                batch_size= 30,
                epochs= 40,
                verbose=0)

        y_pred = model.predict(X_train[test]).ravel()
        fpr_pred, tpr_pred, _ = roc_curve(Y_train[test], y_pred)
        tprs_pred = np.interp(mean_fpr, fpr_pred, tpr_pred)
        tprs_pred[0] = 0.0
        tprs_pred_total.append(tprs_pred)

        y_test_pred = model.predict(X_test).ravel()
        test_fpr,test_tpr, _ = roc_curve(Y_test, y_test_pred)
        tprs_test = np.interp(mean_fpr, test_fpr, test_tpr)
        tprs_test[0] = 0.0
        tprs_test_total.append(tprs_test)

        mean_tprs = (tprs_pred + tprs_test)/2
        tprs.append(mean_tprs)
        # mean_auc = auc(mean_fpr, mean_tprs)
        # auc_total.append(mean_auc)
        # print(mean_auc)

    tprs_pred_mean = np.mean(tprs_pred_total, axis = 0)
    tprs_test_mean = np.mean(tprs_test_total, axis = 0)

    tprs.append(tprs_pred_mean)
    tprs.append(tprs_test_mean)

    return model, tprs 


def plot_roc_curve(tprs, cell_line, cross_cell_line):

    mean_tprs, mean_pred_tprs, mean_test_tprs = tprs[0:4], tprs[5], tprs[6]
    mean_fpr = np.linspace(0, 1, 100)

    mean_tpr = np.mean(mean_tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr,mean_tpr)
    mean_pred_auc = auc(mean_fpr, mean_pred_tprs)
    mean_test_auc = auc(mean_fpr, mean_test_tprs)

    plt.figure(1)
    plt.plot([-0.05, 1], [-0.05, 1], 'k--')
    plt.plot(mean_fpr,mean_tpr,color='b',label='Mean ROC (AUC=%0.5f)' % mean_auc,lw=2,alpha=.8)
    plt.plot(mean_fpr,mean_pred_tprs,color='r',label='Mean Train ROC (AUC=%0.5f)' % mean_pred_auc,lw=2,alpha=.8)
    plt.plot(mean_fpr,mean_test_tprs,color='y',label='Mean Test ROC (AUC=%0.5f)' % mean_test_auc,lw=2,alpha=.8)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of CNN-10(90) model for {}'.format(cell_line))
    plt.legend(loc='lower right')
    if cross_cell_line == None or (cell_line == cross_cell_line):
        plt.savefig('results/{}_roc.png'.format(cell_line))
    else:
        plt.savefig('results/{}_roc.png'.format(cell_line + '_' + cross_cell_line))
    # plt.show()


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
            print('TESTING ON SAME CELL-LINE ({})'.format(args.cell_line))
        else:
            # Overwrite test data as 20% of cross cell-line
            print('TESTING ON CROSS CELL-LINE ({})'.format(args.cross_cell_line))
            _, test = fasta_data_loader(pro_fa='data/{}/enhancers.fa'.format(args.cross_cell_line),
                                        enh_fa='data/{}/promoters.fa'.format(args.cross_cell_line), seed=args.seed)

    # Reshape the data to (n_samples, n_seqs, n_channels)
    X_train,Y_train = train[0].transpose([0,2,1]),train[1]
    X_test,Y_test = test[0].transpose([0,2,1]),test[1]

    print("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    model, tprs = best_model(args.seed)
    model.summary()
    plot_roc_curve(tprs, args.cell_line, args.cross_cell_line)
