import argparse
import os
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from jtnn import *

# Function to create vocab file
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

# Function to preprocess SMILES
def preprocess_smiles(smiles_list, output_path):
    with open(output_path, 'w') as f:
        for smiles in smiles_list:
            mol_tree = MolTree(smiles)
            mol_tree.recover()
            f.write(mol_tree.smiles + '\n')

# Function to train JT-VAE model
def train_jtvae(train_path, vocab_path, save_dir, hidden_size=450, latent_size=56, depthT=20, depthG=3, batch_size=32, lr=1e-3, epochs=10):
    vocab = [x.strip() for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = MoleculeDataset(train_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x)

    for epoch in range(epochs):
        for batch in dataloader:
            model.zero_grad()
            try:
                loss = model(batch)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(e)
                continue
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch+1}.pth')

# Function to load trained JT-VAE model
def load_trained_jtvae(vocab_path, model_path, hidden_size=450, latent_size=56, depthT=20, depthG=3):
    vocab = [x.strip() for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    return model, vocab

# Function to generate molecules
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

# Function to save molecules as images
def save_molecules_as_images(smiles_list, output_dir):
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol)
            img.save(f"{output_dir}/molecule_{i}.png")

# Main function to run all steps
def main(args):
    # Step 1: Create vocabulary file
    smiles_data = pd.read_csv(args.dataset)['smiles']
    create_vocab_file(smiles_data, 'vocab.txt')

    # Step 2: Preprocess SMILES
    preprocess_smiles(smiles_data, 'processed_smiles.txt')

    # Step 3: Train JT-VAE model
    train_jtvae('processed_smiles.txt', 'vocab.txt', args.save_dir, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.batch_size, args.lr, args.epochs)

    # Step 4: Generate new molecules
    model, vocab = load_trained_jtvae('vocab.txt', f'{args.save_dir}/model_epoch_{args.epochs}.pth')
    new_smiles_list = generate_molecules(args.input_smiles, model, vocab, args.num_samples)

    # Step 5: Save generated molecules
    with open(f"{args.output_dir}/generated_smiles.txt", "w") as file:
        for smiles in new_smiles_list:
            file.write(f"{smiles}\n")

    save_molecules_as_images(new_smiles_list, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data, train JT-VAE model, and generate new molecules.")
    parser.add_argument("dataset", type=str, help="Path to the CSV dataset containing SMILES strings.")
    parser.add_argument("input_smiles", type=str, help="Input SMILES string for molecule generation.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of new molecules to generate.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the generated molecules and images.")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save trained models.")
    parser.add_argument("--hidden_size", type=int, default=450, help="Hidden size of the model.")
    parser.add_argument("--latent_size", type=int, default=56, help="Latent size of the model.")
    parser.add_argument("--depthT", type=int, default=20, help="Tree LSTM depth.")
    parser.add_argument("--depthG", type=int, default=3, help="Graph LSTM depth.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    
    args = parser.parse_args()
    main(args)

