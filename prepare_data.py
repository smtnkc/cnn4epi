import argparse
import pandas as pd

def create_fasta_files_from_csv(cell_line, balanced):
    """
    Splits CSV of EP fragment pairs and creates two fasta files
    Arguments: cell_line -- Folder of CSV file including fragments of EP sequences
    Returns: -
    """
    frag_path = 'data/{}/frag_pairs{}.csv'.format(cell_line, '_balanced' if balanced else '')
    df_frag_pairs = pd.read_csv(frag_path)
    df_frag_pairs = df_frag_pairs[['enhancer_frag_name', 'enhancer_frag_seq', 'promoter_frag_name', 'promoter_frag_seq']]
    df_frag_pairs.columns = ['enhancer_name', 'enhancer_seq', 'promoter_name', 'promoter_seq']
    df_enh_frags = df_frag_pairs.drop_duplicates(subset=['enhancer_name'])[['enhancer_name', 'enhancer_seq']].reset_index(drop=True)
    df_pro_frags = df_frag_pairs.drop_duplicates(subset=['promoter_name'])[['promoter_name', 'promoter_seq']].reset_index(drop=True)

    enh_fa = open('data/{}/enhancers.fa'.format(cell_line), 'w')
    for i in range(len(df_enh_frags)):
        enh_fa.write(">{}\n".format(df_enh_frags['enhancer_name'][i]))
        enh_fa.write("{}\n".format(df_enh_frags['enhancer_seq'][i]))
    enh_fa.close()

    pro_fa = open('data/{}/promoters.fa'.format(cell_line), 'w')
    for i in range(len(df_pro_frags)):
        pro_fa.write(">{}\n".format(df_pro_frags['promoter_name'][i]))
        pro_fa.write("{}\n".format(df_pro_frags['promoter_seq'][i]))
    pro_fa.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert4epi')
    parser.add_argument('--balanced', action='store_true') # set to balance enhancers and promoters
    args = parser.parse_args()

    cell_lines = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'combined']

    # Generate Train/Dev/Test CSVs for the cell-line
    for cell_line in cell_lines:
        create_fasta_files_from_csv(cell_line, args.balanced)
