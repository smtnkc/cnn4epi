import h5py
import numpy as np


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

    # The fisrt column is A, second is C, thrid is G, fourth is U or T
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


def split_dataset(one_hot, labels, test_frac = 0.1):
    """
    Split the dataset into training and test set
    """
    def split_index(num_data, test_frac):
        
        train_frac = 1 - test_frac
        cum_index = np.array(np.cumsum([0,train_frac,test_frac])*num_data).astype(int)
        shuffle = np.random.permutation(num_data)
        train_index = shuffle[cum_index[0]:cum_index[1]]
        test_index =shuffle[cum_index[1]:cum_index[2]]

        return train_index,test_index

    num_data = len(one_hot)
    train_index, test_index = split_index(num_data, test_frac)

    train = (one_hot[train_index],labels[train_index,:])
    test = (one_hot[test_index],labels[test_index,:])
    indices = [train_index,test_index]

    return train,test,indices


def fasta_data_loader(pro_fa, enh_fa):

    pos_f = open('data/US_UU/K562_US.fa')
    neg_f = open('data/US_UU/K562_UU.fa')

    pos = parse_fasta_seq(pos_f)
    neg = parse_fasta_seq(neg_f)

    pos_one_hot = dna_to_one_hot(pos)
    neg_one_hot = dna_to_one_hot(neg)

    one_hot_seq = np.vstack([pos_one_hot, neg_one_hot])
    labels = np.vstack([np.ones((len(pos_one_hot), 1)), np.zeros((len(neg_one_hot), 1))])

    train, test, indices = split_dataset(one_hot_seq, labels, test_frac=0.1)

    return train, test
