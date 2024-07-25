from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import matplotlib.pyplot as plt

from rdkit import __version__ as rdkit_version

print(f"RDKit version: {rdkit_version}")

def molecule_to_graph(molecule):
    """Convert a molecule into a networkx graph object."""
    graph = nx.Graph()
    
    # Add nodes
    for atom in molecule.GetAtoms():
        graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    
    # Add edges
    for bond in molecule.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        graph.add_edge(start_idx, end_idx, bond_type=bond.GetBondTypeAsDouble())
    
    return graph

def check_connectivity(graph):
    """Check if the graph is connected."""
    return nx.is_connected(graph)

def visualize_graph(graph, molecule, output_file):
    """Visualize the graph with atom labels and save the image to a file."""
    pos = nx.spring_layout(graph)
    labels = {node: molecule.GetAtomWithIdx(node).GetSymbol() for node in graph.nodes()}
    
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=500, node_color='lightblue', font_size=16, font_weight='bold')
    plt.title('Molecular Graph')
    plt.savefig(output_file)  # Save the figure as an image file
    plt.close()  # Close the plot to avoid display

def main(mol_file, output_image_file):
    """Main function to load a molecule, convert it to a graph, check connectivity, and save the graph image."""
    # Load the molecule
    mol_supplier = Chem.SDMolSupplier(mol_file)
    mol = mol_supplier[0]  # Assuming the first molecule is the one we want
    
    # Convert to graph
    graph = molecule_to_graph(mol)
    
    # Check connectivity
    is_connected = check_connectivity(graph)
    print(f"Is the molecule graph connected? {'Yes' if is_connected else 'No'}")
    
    # Save the graph image
    visualize_graph(graph, mol, output_image_file)

if __name__ == "__main__":
    mol_file = 'molecule.sdf'  # Replace with your SDF file containing the conformer
    output_image_file = 'molecular_graph.png'  # Replace with the desired output image file name
    main(mol_file, output_image_file)

