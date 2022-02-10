import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split

def custom_f1(y_true, y_pred):
    '''
    Implementing F1 score in Keras
    https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
    https://github.com/YiLi225/NeptuneBlogs/blob/main/Implement_F1score_neptune_git_NewVersion.py
    '''

    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def parse_fasta_seq(seq_file):
    """
    Parse fasta file of sequences
    Arguments: seq_file -- a fasta file
    Returns: seq_list -- a numpy array
    """

    seq = {}
    seqs_no_n = []

    for line in seq_file :
        if line.startswith('>') :
            name = line.replace('>','').split()[0]
            seq[name] = ''
        else :
            seq[name] += line.replace('\n','').strip()

    seqs_list = list(seq.values())
    for i in seqs_list:
        if 'N' not in i:
            seqs_no_n.append(i)

    seqs_list = np.array(seqs_no_n)

    seq_file.close()

    return seqs_list


def dna_to_one_hot(dna_seqs):
    """
    Convert DNA/RNA sequences to a one-hot representation
    Arguments: seq -- a DNA sequence or RNA sequence
    Return: one_hot_seq -- a numpy array
    """

    one_hot_seq = []

    for seq in dna_seqs:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))

    # The first column is A, second is C, third is G, fourth is U or T
        index = [j for j in range(seq_length) if seq[j] == 'A']
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1, index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2, index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3, index] = 1

        one_hot_seq.append(one_hot)

    #convert to numpy array
    one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq


def fasta_data_loader(pro_fa, enh_fa, seed):

    pos_f = open(pro_fa)
    neg_f = open(enh_fa)

    pos = parse_fasta_seq(pos_f)
    neg = parse_fasta_seq(neg_f)

    pos_one_hot = dna_to_one_hot(pos)
    neg_one_hot = dna_to_one_hot(neg)

    one_hot_seq = np.vstack([pos_one_hot, neg_one_hot])
    labels = np.vstack([np.ones((len(pos_one_hot), 1)), np.zeros((len(neg_one_hot), 1))])

    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(one_hot_seq, 
        labels, test_size=0.2, random_state=seed, stratify=labels)

    return (X_train, y_train), (X_test, y_test)
