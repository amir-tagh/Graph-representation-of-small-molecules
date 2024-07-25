#pip install rdkit pandas seaborn matplotlib
#calculate all the molecular descriptors available in RDKit
####################
#Airhossein Taghavi
#UF Scripps
#####################


import argparse
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate all molecular descriptors
def calculate_all_descriptors(smiles_list):
    descriptor_names = [desc_name[0] for desc_name in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    descriptors = {name: [] for name in descriptor_names}

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            values = calculator.CalcDescriptors(mol)
            for name, value in zip(descriptor_names, values):
                descriptors[name].append(value)
        else:
            for name in descriptor_names:
                descriptors[name].append(None)  # Append None for invalid SMILES

    return pd.DataFrame(descriptors)

# Function to plot molecular descriptors
def plot_descriptors(descriptor_dfs, labels, output_dir):
    for desc in descriptor_dfs[0].columns:
        plt.figure(figsize=(10, 6))
        for df, label in zip(descriptor_dfs, labels):
            if df[desc].dropna().empty:
                continue
            sns.kdeplot(df[desc].dropna(), label=label, fill=True)
        plt.title(f'Distribution of {desc}')
        plt.legend()
        plt.savefig(f'{output_dir}/{desc}.png')
        plt.close()

# Main function to process multiple SMILES sets and plot descriptors
def main(args):
    smiles_sets = []
    labels = args.labels.split(',')

    for smiles_file in args.smiles_files:
        with open(smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f.readlines()]
            smiles_sets.append(smiles_list)

    descriptor_dfs = [calculate_all_descriptors(smiles_list) for smiles_list in smiles_sets]

    plot_descriptors(descriptor_dfs, labels, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and compare molecular descriptors for multiple SMILES sets.")
    parser.add_argument("smiles_files", type=str, nargs='+', help="List of files containing SMILES strings.")
    parser.add_argument("--labels", type=str, help="Comma-separated labels for the SMILES sets.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the plots.")

    args = parser.parse_args()
    main(args)

