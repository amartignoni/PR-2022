from pathlib import Path
from xml.dom import minidom
import networkx as nx
from tqdm import tqdm
import csv


def predict(train_set, train_labels, graph, k):
    distances = []
    for train_id, train_graph in train_set.items():
        # Add a tuple (label, GED distance)
        distance = nx.graph_edit_distance(graph, train_graph, timeout=0.3)
        if distance == None:
            distance = 200.0
        distances.append((train_labels[train_id], distance))
        print(".", end="", flush=True)

    # Sort according the distance
    distances.sort(key=lambda k: k[1])

    # Take only the label(s) for the k-NN
    labels = [e[0] for e in distances[0:k]]

    # Returns the most occurent label
    return max(set(labels), key=labels.count)


# PART 1: Graphs creation, datasets splitting
root_path = Path("MoleculesClassification")
graphs_path = root_path / "gxl"
test_path = root_path / "test"

# Dict of graphs, key is file name and value is graph
molecules = {}

for molecule_path in graphs_path.iterdir():
    # Parsing GXL
    # parse needs a string
    doc = minidom.parse(str(molecule_path))

    # Graph from NetworkX library
    molecule = nx.Graph()

    # Get node label (symbol) and add it to the graph
    for node in doc.getElementsByTagName("node"):
        id = node.getAttribute("id")
        symbol = node.getElementsByTagName("attr")[0].firstChild.firstChild.data
        # symbol is saved as attribute
        molecule.add_node(id, symbol=symbol)

    # Construct the adjacency list and add the edges to the graph
    adjacency_list = [
        (edge.getAttribute("from"), edge.getAttribute("to"))
        for edge in doc.getElementsByTagName("edge")
    ]
    molecule.add_edges_from(adjacency_list)

    molecules[molecule_path.stem] = molecule

# Train / validation split
train_set = {}
train_labels = {}
validation_set = {}
validation_labels = {}
test_set = {}

for molecule_path in test_path.iterdir():
    # Parsing GXL
    # parse needs a string
    doc = minidom.parse(str(molecule_path))

    # Graph from NetworkX library
    molecule = nx.Graph()

    # Get node label (symbol) and add it to the graph
    for node in doc.getElementsByTagName("node"):
        id = node.getAttribute("id")
        symbol = node.getElementsByTagName("attr")[0].firstChild.firstChild.data
        # symbol is saved as attribute
        molecule.add_node(id, symbol=symbol)

    # Construct the adjacency list and add the edges to the graph
    adjacency_list = [
        (edge.getAttribute("from"), edge.getAttribute("to"))
        for edge in doc.getElementsByTagName("edge")
    ]
    molecule.add_edges_from(adjacency_list)

    test_set[molecule_path.stem] = molecule

with open(root_path / "train.txt", "r") as train_file:
    for line in train_file:
        id, label = line.split()
        train_set[id] = molecules[id]
        train_labels[id] = label

with open(root_path / "valid.txt", "r") as validation_file:
    for line in validation_file:
        id, label = line.split()
        validation_set[id] = molecules[id]
        validation_labels[id] = label


# PART 2: Classify each element of the validation set and compute accuracy
# Hyperparameter for the k-NN classifier
k = 10
correct_predictions = 0
n_predictions = len(validation_set)

# with open("mol_val.csv", 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter=",")
#     for validation_id, validation_graph in tqdm(validation_set.items()):
#         prediction = predict(train_set, train_labels, validation_graph, k)
#         ground_truth_label = validation_labels[validation_id]
#         print(prediction, validation_id, ground_truth_label)
#         writer.writerow([validation_id, prediction])
#         if prediction == ground_truth_label:
#             correct_predictions += 1

# print(f"\n{k}-NN ended, accuracy: {correct_predictions / n_predictions}\n")

with open("mol.csv", 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    for test_id, test_graph in tqdm(test_set.items()):
        prediction = predict(train_set, train_labels, test_graph, k)
        writer.writerow([test_id, prediction])
