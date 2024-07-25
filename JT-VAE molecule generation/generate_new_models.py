import argparse
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from jtnn import *

def load_trained_jtvae(vocab_path, model_path, hidden_size=450, latent_size=56, depthT=20, depthG=3):
    vocab = [x.strip() for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    return model, vocab

def generate_molecules(smiles, model, vocab, num_samples=10):
    mol = Chem.MolFromSmiles(smiles)
    mol_tree = MolTree(smiles)
    mol_tree.recover()

    for node in mol_tree.nodes:
        node.wid = vocab.get_index(node.smiles)

    _, tree_vec, mol_vec = model.encode_latent([mol_tree])
    new_smiles_list = []

    for _ in range(num_samples):
        z_tree = torch.randn([1, model.latent_size // 2]).cuda()
        z_mol = torch.randn([1, model.latent_size // 2]).cuda()
        new_smiles = model.decode(tree_vec + z_tree, mol_vec + z_mol, prob_decode=False)
        new_smiles_list.append(new_smiles)

    return new_smiles_list

def save_molecules_as_images(smiles_list, output_dir):
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol)
            img.save(f"{output_dir}/molecule_{i}.png")

def main(input_smiles, num_samples, output_dir, vocab_path, model_path):
    model, vocab = load_trained_jtvae(vocab_path, model_path)
    new_smiles_list = generate_molecules(input_smiles, model, vocab, num_samples)

    with open(f"{output_dir}/generated_smiles.txt", "w") as file:
        for smiles in new_smiles_list:
            file.write(f"{smiles}\n")

    save_molecules_as_images(new_smiles_list, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new molecules from a given SMILES string using JT-VAE model.")
    parser.add_argument("input_smiles", type=str, help="Input SMILES string")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of new molecules to generate")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the generated molecules and images")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained JT-VAE model file")

    args = parser.parse_args()
    main(args.input_smiles, args.num_samples, args.output_dir, args.vocab_path, args.model_path)

