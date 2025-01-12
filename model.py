import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
import dgl.function as fn
from dgl.nn import GraphConv, EdgeWeightNorm
import dgl
import torch
from gensim.models import Word2Vec


class GCNWithEdgeFeatures(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats, out_feats):
        super(GCNWithEdgeFeatures, self).__init__()
        self.edge_norm = EdgeWeightNorm(norm='both')  # Normalize edge weights
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True)
        self.edge_fc = nn.Linear(edge_feats, 1)  # Map edge features to a scalar weight

    def forward(self, graph, node_feats, edge_feats):
        # Compute edge weights from edge features
        edge_weights = self.edge_fc(edge_feats).squeeze(-1)
        edge_weights = self.edge_norm(graph, edge_weights)
        
        # First GCN layer
        h = self.conv1(graph, node_feats, edge_weight=edge_weights)
        h = F.relu(h)

        # Second GCN layer
        h = self.conv2(graph, h, edge_weight=edge_weights)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')

    def forward(self, graph, node_feats):
        h = self.conv1(graph, node_feats)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
class EdgeFeatureGNNLayer(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, out_feats):
        super(EdgeFeatureGNNLayer, self).__init__()
        self.node_linear = nn.Linear(node_in_feats, out_feats)
        self.edge_linear = nn.Linear(edge_in_feats, out_feats)
        self.aggregate_fn = dgl.function.sum("m", "h_new")  # Summation aggregation

    def forward(self, graph, node_feats, edge_feats):
        # Define message function: Combine node and edge features
        graph.ndata["h"] = self.node_linear(node_feats)
        graph.edata["e"] = self.edge_linear(edge_feats)

        graph.apply_edges(lambda edges: {"m": edges.src["h"] + edges.data["e"]})  # Message
        graph.update_all(message_func=dgl.function.copy_e("m", "m"), reduce_func=self.aggregate_fn)

        # Updated node embeddings
        return graph.ndata["h_new"]

class EdgeFeatureAttentionGNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, out_feats, num_heads):
        super(EdgeFeatureAttentionGNN, self).__init__()
        self.attn = dgl.nn.GATConv(node_in_feats, out_feats, num_heads, feat_drop=0.1, edge_feats=edge_in_feats)

    def forward(self, graph, node_feats, edge_feats):
        # Apply GAT with edge features
        return self.attn(graph, node_feats, edge_feats)

def generat_emmbedding_by_random_walk(graph):

    # Generate random walks
    walks = dgl.sampling.random_walk(graph, nodes=list(graph.nodes()), length=10)

    # Prepare sequences for Word2Vec
    walk_sequences = walks[0].tolist()
    print("walk_sequences:",walk_sequences)
    # Train Word2Vec
    model = Word2Vec(walk_sequences, vector_size=128, window=5, min_count=1, sg=1, workers=4)
    node_embeddings = model.wv
    return node_embeddings
class EdgeGNNLayer(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats):
        super(EdgeGNNLayer, self).__init__()
        # Node and edge linear transformations
        self.node_fc = nn.Linear(in_feats, out_feats)
        self.edge_fc = nn.Linear(edge_feats, out_feats)
        self.msg_fc = nn.Linear(out_feats, out_feats)  # Use 'out_feats' instead of 128

    def forward(self, graph, node_features, edge_features):
        with graph.local_scope():
            # Apply transformations to node and edge features
            node_features = self.node_fc(node_features)  # (num_nodes, out_feats)
            edge_features = self.edge_fc(edge_features)  # (num_edges, out_feats)

            # Store transformed features
            graph.ndata['h'] = node_features
            graph.edata['e'] = edge_features

            # Define the message function (combine node and edge features)
            graph.apply_edges(fn.u_add_e('h', 'e', 'msg'))  # u_add_e sums node and edge features

            # Apply transformation to messages
            graph.edata['msg'] = self.msg_fc(graph.edata['msg'])  # Ensure input shape matches
            
            # Aggregate messages (e.g., mean)
            graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'h'))

            return graph.ndata['h']

def train_model(raw_model,optimizer ,loss_fun ,graph, node_features, edge_features, epochs=100, lr=0.01):
    # Initialize model and optimizer
   
    
    # True adjacency matrix
    adj_true = graph.adjacency_matrix().to_dense()

    # Training loop
    for epoch in range(epochs):
        raw_model.train()
        optimizer.zero_grad()
        
        # Forward pass
        _, adj_pred = raw_model(graph, node_features, edge_features)
        
        # Compute loss
        loss = loss_fun(adj_pred, adj_true)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return raw_model

class GraphAutoencoder(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats, latent_feats):
        super(GraphAutoencoder, self).__init__()
        # Encoder layers
        self.encoder1 = EdgeGNNLayer(node_in_feats, edge_in_feats, hidden_feats)
        self.encoder2 = EdgeGNNLayer(hidden_feats, edge_in_feats, latent_feats)
        
        # Decoder (e.g., inner product for reconstruction)
        self.decoder = nn.Linear(latent_feats, latent_feats)

    def forward(self, graph, node_features, edge_features):
        # Encode the graph
        h1 = F.relu(self.encoder1(graph, node_features, edge_features))
        h2 = self.encoder2(graph, h1, edge_features)
        
        # Decode the graph structure
        graph.ndata['z'] = h2
        z = h2
        adj_pred = torch.sigmoid(torch.matmul(z, z.T))
        return z, adj_pred
def loss_function(adj_pred, adj_true):
    return F.binary_cross_entropy(adj_pred, adj_true)
class GraphAutoencoderGNN(nn.Module):
    def __init__(self,args,):
        super(GraphAutoencoderGNN, self).__init__()
        in_feats =args.input_feat_model_graph
        hidden_feats =args.dimension_entity
        out_feats =args.dimension_entity
        self.encoder = GraphSAGE(in_feats, hidden_feats, out_feats)
        self.decoder = DotProductDecoder()

    def forward(self, graph, features):
        # Encode
        z = self.encoder(graph, features)
        # Decode
        adj_pred = self.decoder(z)
        return z, adj_pred
class DotProductDecoder(nn.Module):
    def forward(self, z):
        # Dot product to predict edges
        adj_pred = torch.sigmoid(torch.matmul(z, z.T))
        return adj_pred


class EntInit(nn.Module):
    def __init__(self, args):
        super(EntInit, self).__init__()
        self.args = args

        self.rel_head_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))
        self.rel_tail_emb = nn.Parameter(torch.Tensor(args.num_rel, args.ent_dim))

        nn.init.xavier_normal_(self.rel_head_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_tail_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, g_bidir):
        num_edge = g_bidir.num_edges()
        etypes = g_bidir.edata['type']
        g_bidir.edata['ent_e'] = torch.zeros(num_edge, self.args.ent_dim).to(self.args.gpu)
        rh_idx = etypes < self.args.num_rel
        rt_idx = etypes >= self.args.num_rel
        
        g_bidir.edata['ent_e'][rh_idx] = self.rel_head_emb[etypes[rh_idx]]
        g_bidir.edata['ent_e'][rt_idx] = self.rel_tail_emb[etypes[rt_idx] - self.args.num_rel]

        message_func = dgl.function.copy_e('ent_e', 'msg')
        reduce_func = dgl.function.mean('msg', 'feat')
        g_bidir.update_all(message_func, reduce_func)
        g_bidir.edata.pop('ent_e')   