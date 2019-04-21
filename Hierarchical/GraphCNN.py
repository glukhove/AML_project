import torch.nn.functional as F
import torch.nn as nn
import torch
import dgl
from MLP import MLP
import numpy as np

class GraphCNN(nn.Module):
    def __init__(self, batch_size, num_mlp_layers, num_mlp_pooling_layers, input_dim, hidden_dim, output_dim, final_dropout, poolings, update_layers, mpl_layers_pred, mlp_pred_factor, print_mode):
        '''
        	batch_size: batch size used in training
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            num_mlp_pooling_layers: number of layers in mlps for pooling layers
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            poolings: a list with units for pooling
            update_layers: number of update layers between every pooling operation
            mpl_layers_pred: number of layers in mlps used for predictions
            mlp_pred_factor: multiplication factor used in prediction mlp hidden number
            print_mode: setting of printing function
        '''

        super(GraphCNN, self).__init__()

        self.batch_size = batch_size
        self.final_dropout = final_dropout
        self.print_mode = print_mode
        self.poolings = poolings
        self.update_layers = update_layers
        self.num_update_layers = update_layers * len(poolings)
        self.eps = nn.Parameter(torch.zeros(self.num_update_layers))
        self.edge_features = nn.Parameter(torch.normal(mean=torch.zeros(6),
                                            std=torch.ones(6)/100), requires_grad=True)
        if self.print_mode==1:
            print(self.edge_features)

        
        ###List of MLPs
        self.mlps = torch.nn.ModuleList()
        
        self.pools = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_update_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        self.batch_norms_pool = torch.nn.ModuleList()
        for _ in range(len(poolings)):
            self.batch_norms_pool.append(nn.BatchNorm1d(hidden_dim))
            
        ################
        
        #inputlayer
        self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
        
        for i_block, _ in enumerate(self.poolings):
            for _ in range(self.update_layers):
                #node features update layer
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            
            #pooling layer
            self.pools.append(MLP(num_mlp_pooling_layers, hidden_dim, hidden_dim, poolings[i_block]*self.batch_size))        


        # MLP that maps the hidden representation at different layers into a prediction score
        self.prediction = MLP(mpl_layers_pred, hidden_dim, int(np.round(hidden_dim*mlp_pred_factor)), output_dim)

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



    def next_layer_update(self, h, layer, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 
        #If sum or average pooling
        pooled = torch.mm(Adj_block, h) #aggregating
        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + self.eps[layer]*h #add central node - combine
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h
    
    
    def next_layer_pool(self, h, pool_layer, Adj_block = None, batch_num_nodes=None, new_nodes=None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 
        #If sum or average pooling
        pooled = torch.mm(Adj_block, h) #aggregating
        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + h #add central node - combine
        S = self.pools[pool_layer](pooled) # new node features of claster size
         # apply mask on S to make sure that S will contain zeros at plases where there are different graphs
        max_num_nodes = Adj_block.size()[1]
        mask = self.construct_mask(max_num_nodes, batch_num_nodes, new_nodes=new_nodes) 
            
        S = S * mask
        S = S - torch.where(S != 0, torch.zeros_like(S), torch.ones_like(S) * float('inf'))
    
        # apply softmax to make rows sums = 1 - we will get probabilities as result
        S = F.softmax(S, dim=1)
          
        S = torch.where(S == -float('inf'), torch.zeros_like(S), S)
        S = S * mask 
            
        # multiply S and h
        h_new = torch.t(S) @ h
        
        # multiply S and A
        A_new = torch.t(S) @ Adj_block @ S
        
        h_new = self.batch_norms_pool[pool_layer](h_new)
        
        return h_new, A_new
    
    
    def construct_mask(self, max_nodes, batch_num_nodes, new_nodes): 
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        
        batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        # masks
        batch_num_nodes = np.repeat(batch_num_nodes, new_nodes)
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        shift = 0
        window = 0
        for i, mask in enumerate(packed_masks):
            end = shift+batch_num_nodes[i]
            out_tensor[i, shift:end] = mask
            window += 1
            if window == new_nodes:
                shift += len(mask)
                window = 0
                
        return torch.t(out_tensor)



    
    def forward(self, batch_graph):
        X_concat = torch.cat([graph.ndata['h'] for graph in batch_graph], 0)
        graph_pool = self.__preprocess_graphpool(batch_graph)
        
        batch_num_nodes = torch.sparse.sum(graph_pool, dim=1)._values().int().numpy()
        
        Adj_block = dgl.batch(batch_graph)
        Adj_block = torch.sparse.FloatTensor(Adj_block.adjacency_matrix().float()._indices(), Adj_block.edata['h'].float()).to_dense()
        Adj_block = Adj_block @ self.edge_features.view(-1,1)
        Adj_block = Adj_block.view(Adj_block.shape[0], Adj_block.shape[1])
            

        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat.float()
        
        if self.print_mode==1:
            print('first h')
            print(h.data.numpy()[:100])

        for i_block, next_pooling in enumerate(self.poolings):
            update_layers_local = min(self.update_layers, next_pooling)-1
            for u_layer in range(update_layers_local):
                n_update_layer = i_block * self.update_layers + u_layer
                h = self.next_layer_update(h.float(), n_update_layer, Adj_block = Adj_block)
                if self.print_mode==1:
                    print(f'{n_update_layer} layer h')
                    print(h.data.numpy())
        
            h, Adj_block = self.next_layer_pool(h.float(), i_block, Adj_block = Adj_block, batch_num_nodes=batch_num_nodes, new_nodes=next_pooling)
                
        
            if self.print_mode==1:
                print(f'{i_block} reduced h')
                print(h.data.numpy())
                
            # change for next layer
            batch_num_nodes = [self.poolings[i_block]]*self.batch_size
        

        score_over_layer = 0
    
        if self.print_mode==1:
            print('pooled_h')
            print(h.data.numpy())
        score_over_layer += F.dropout(self.prediction(h), self.final_dropout, training = self.training)

        if self.print_mode == 2:
            print(score_over_layer)

        return score_over_layer