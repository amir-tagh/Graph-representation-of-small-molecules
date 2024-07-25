from jtnn import *
import pandas as pd

def create_vocab_file(smiles_list, output_path):
    vocab = set()
    for smiles in smiles_list:
        mol = MolTree(smiles)
        mol.recover()
        for node in mol.nodes:
            vocab.add(node.smiles)
    
    with open(output_path, "w") as f:
        for v in vocab:
            f.write(f"{v}\n")

if __name__ == "__main__":
    smiles_data = pd.read_csv('your_smiles_dataset.csv')['smiles']
    create_vocab_file(smiles_data, 'vocab.txt')

