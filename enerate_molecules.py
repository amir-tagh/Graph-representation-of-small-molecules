import argparse
from rdkit import Chem
from rdkit.Chem import Draw
from moses.model import VAE
from moses.utils import CharVocab

def load_pretrained_model():
    vocab = CharVocab.load('moses/data/Vocab/canonic_char.pickle')
    model = VAE(vocab)
    model.load_state_dict('moses/checkpoints/VAE/VAE.pt')
    model.eval()
    return model, vocab

def generate_molecules(smiles, model, vocab, num_samples=10):
    # Encode the SMILES string
    encoded_smiles = vocab.string2tensor([smiles])
    encoded_smiles = encoded_smiles.repeat(num_samples, 1)
    
    # Generate new molecules
    with torch.no_grad():
        decoded_molecules = model.decode(encoded_smiles)
    
    # Convert to SMILES strings
    new_smiles = [vocab.tensor2string(smile) for smile in decoded_molecules]
    return new_smiles

def save_molecules_as_images(smiles_list, output_dir):
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol)
            img.save(f"{output_dir}/molecule_{i}.png")

def main(input_smiles, num_samples, output_dir):
    model, vocab = load_pretrained_model()
    new_smiles_list = generate_molecules(input_smiles, model, vocab, num_samples)
    
    with open(f"{output_dir}/generated_smiles.txt", "w") as file:
        for smiles in new_smiles_list:
            file.write(f"{smiles}\n")
    
    save_molecules_as_images(new_smiles_list, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new molecules from a given SMILES string using a VAE model.")
    parser.add_argument("input_smiles", type=str, help="Input SMILES string")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of new molecules to generate")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the generated molecules and images")

    args = parser.parse_args()
    main(args.input_smiles, args.num_samples, args.output_dir)

