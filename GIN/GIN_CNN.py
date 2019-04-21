import torch.nn.functional as F
import torch.nn as nn
import torch
import dgl
from MLP import MLP
import numpy as np

class GIN_CNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
        '''

        super(GIN_CNN, self).__init__()

        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.eps = nn.Parameter(torch.normal(mean=torch.zeros(self.num_layers-1),
                                            std=torch.ones(self.num_layers-1)/100))
        self.edge_features = nn.Parameter(torch.normal(mean=torch.zeros(6),
                                            std=torch.ones(6)/100), requires_grad=True)
        self.calc_edges = nn.Linear(6, 1)
        
        
        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))


    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            elem.extend([1]*len(graph))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool



    def next_layer_update(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 
        pooled = torch.mm(Adj_block, h) #aggregating
        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + self.eps[layer]*h #add central node - combine
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    
    def forward(self, batch_graph):
        X_concat = torch.cat([graph.ndata['h'] for graph in batch_graph], 0)
        graph_pool = self.__preprocess_graphpool(batch_graph)


        Adj_block = dgl.batch(batch_graph)
        Adj_block = torch.sparse.FloatTensor(Adj_block.adjacency_matrix().float()._indices(), Adj_block.edata['h'].float()).to_dense()
        Adj_block = Adj_block @ self.edge_features.view(-1,1)
        Adj_block = Adj_block.view(Adj_block.shape[0], Adj_block.shape[1])
            

        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat.float()

        for layer in range(self.num_layers-1):
            h = self.next_layer_update(h.float(), layer, Adj_block = Adj_block)
            hidden_rep.append(h)

        score_over_layer = 0
    
        #perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.mm(graph_pool.float(), h.float())
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        return score_over_layer
