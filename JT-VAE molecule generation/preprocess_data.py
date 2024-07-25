from jtnn import *
import pandas as pd

def preprocess_smiles(smiles_list, output_path):
    with open(output_path, 'w') as f:
        for smiles in smiles_list:
            mol_tree = MolTree(smiles)
            mol_tree.recover()
            f.write(mol_tree.smiles + '\n')

if __name__ == "__main__":
    smiles_data = pd.read_csv('your_smiles_dataset.csv')['smiles']
    preprocess_smiles(smiles_data, 'processed_smiles.txt')

