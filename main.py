import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, concat=True)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = self.conv2(x, edge_index)
        return x

# Layering activity analysis using BFS
def bfs_layering(graph, start, max_depth):
    visited = {node: False for node in graph.nodes()}
    layering = {}
    queue = [(start, 0)]  # (node, layer)

    while queue:
        current, layer = queue.pop(0)
        if not visited[current] and layer <= max_depth:
            visited[current] = True
            layering[current] = layer
            for neighbor in graph.neighbors(current):
                if not visited[neighbor]:
                    queue.append((neighbor, layer + 1))

    return layering

# Streamlit app
# st.title("Fraud Detection and Layering Activity Analysis")
import streamlit as st

# Add custom CSS to center the title
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 36px; /* Adjust the size as needed */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use markdown to apply the centered style
st.markdown('<div class="centered-title">Fraud Detection and Layering Activity Analysis</div>', unsafe_allow_html=True)

st.write("This app uses a Graph Attention Network (GAT) for fraud detection and BFS for layering activity analysis.")

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file containing transaction data", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file) #, encoding='latin1', error_bad_lines=False, warn_bad_lines=True)
    st.write("Dataset Preview:")
    st.dataframe(data.head(10))


    # Preprocessing
    # Display column names
    st.write("Columns in the uploaded dataset:")
    st.write(data.columns)

    # Validate 'amount' column
    if 'amount' not in data.columns:
        st.error("The dataset must contain an 'amount' column for transaction amounts. Please upload a valid dataset.")
        st.stop()

    # Normalize the 'amount' column
    data['amount'] = (data['amount'] - data['amount'].mean()) / data['amount'].std()

    # Construct the graph
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(
            row['From_Account_id'],
            row['To_Account_id'],
            amount=row['amount'],
            timestamp=row['Date/Time']
        )

    # Convert to PyTorch Geometric Data format
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    edge_index = torch.tensor(
        [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor([
        [G.edges[edge]['amount']] for edge in G.edges()
    ], dtype=torch.float)

    node_features = torch.zeros((len(G.nodes()), 1), dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    # Define and train the GAT model
    labels = torch.randint(0, 2, (len(G.nodes()),), dtype=torch.long)
    train_mask, test_mask = train_test_split(range(len(labels)), test_size=0.2, random_state=42)

    model = GAT(in_channels=node_features.shape[1], out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    st.write("Training the GAT model...")
    model.train()
    for epoch in range(100):  # For demo purposes, reduce the number of epochs
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        st.write(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    st.write("Model training completed.")

    # Perform layering activity analysis
    start_node = list(G.nodes())[0]
    max_depth = st.slider("Select the maximum depth for layering analysis", min_value=1, max_value=20, value=9)
    layering = bfs_layering(G, start_node, max_depth)

    layer_counts = {layer: 0 for layer in range(max_depth + 1)}
    for _, layer in layering.items():
        layer_counts[layer] += 1

    st.write(f"Layering Counts up to {max_depth} Layers:")
    st.write(layer_counts)

    # Visualize layering results
    fig, ax = plt.subplots()
    ax.bar(layer_counts.keys(), layer_counts.values(), color='skyblue')
    ax.set_xlabel('Layer Number')
    ax.set_ylabel('Number of Nodes')
    ax.set_title(f'Distribution of Nodes Across Layers (up to {max_depth} Layers)')
    st.pyplot(fig)
