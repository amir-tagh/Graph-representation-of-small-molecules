import argparse
import torch
from jtnn import *

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train JT-VAE model.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to preprocessed SMILES file for training.')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocabulary file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save trained models.')
    parser.add_argument('--hidden_size', type=int, default=450, help='Hidden size of the model.')
    parser.add_argument('--latent_size', type=int, default=56, help='Latent size of the model.')
    parser.add_argument('--depthT', type=int, default=20, help='Tree LSTM depth.')
    parser.add_argument('--depthG', type=int, default=3, help='Graph LSTM depth.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    
    args = parser.parse_args()
    train_jtvae(args.train_path, args.vocab_path, args.save_dir, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.batch_size, args.lr, args.epochs)

