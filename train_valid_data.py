from tool import (
    read_triplet_nell,
    read_triplet_fb,
    divid_entities_by_type_nell,
    create_graph,
    add_feature_to_graph_nodes,
    add_feature_to_graph_edges,
    generate_group_triples ,
    generate_group_triples_v1 ,
    create_main_graph,
	add_model_feature_to_main_graph,
	remove_duplicate ,
    
)
from torch.optim import Adam
from tqdm import tqdm
import torch
import numpy as np
import random

from model import (
    GraphSAGE,
    GraphAutoencoder,
    GraphAutoencoderGNN,
    loss_function,
)
import os

class TrainData():
    def __init__(self, args,path, valdiation_time = False):
        print(f"Loading {path} data...")
        # Read and process triplets
        if "nell" in args.data_name.lower() :
              id2ent, id2rel, self.triplets, en2id, rel2id ,self.rel_info, self.pair_info ,self.spanning ,self.triplets2id = read_triplet_nell(path)
              endic, en_dic_id,self.ent_type = divid_entities_by_type_nell(id2ent, en2id)
        elif "fb" in args.data_name.lower():
              id2ent, id2rel, self.triplets, en2id, rel2id ,self.rel_info, self.pair_info ,self.spanning ,self.triplets2id,en_dic_id,self.ent_type = read_triplet_fb(path)
        
        self.num_relations = args.num_rel if valdiation_time else len(id2rel)
            
         # Extract relations and create graph
        self.graph  = create_main_graph(self.triplets) 
        # entity_type_triples ,inner_rel ,output_relations,input_relations = generate_group_triples(en_dic_id,self.triplets,self.ent_type)
        entity_type_triples ,inner_rel ,output_relations,input_relations = generate_group_triples_v1(self.triplets,self.ent_type,self.num_relations)
        model_graph = create_graph(entity_type_triples)
       
        # Add features to the graph
        self.num_ent = self.graph.num_nodes()
        self.num_triplets = len(self.triplets)
        # print("number relations:", num_relations)
        self.model_graph = add_feature_to_graph_nodes(model_graph, inner_rel,output_relations,input_relations, self.num_relations)
        # self.model_graph = add_feature_to_graph_edges(model_graph, entity_type_triples, self.num_relations)
        
        self.node_features = model_graph.ndata["feat"]
        # self.edge_features = model_graph.edata["rel_feat"]
        self.args =args
        self.adj_true = model_graph.adjacency_matrix().to_dense()
        self.filter_dict = {}
        for triplet in self.triplets:
            h,r,t = triplet
            if ('_', r, t) not in self.filter_dict:
                self.filter_dict[('_', r, t)] = [h]
            else:
               self.filter_dict[('_', r, t)].append(h)

            if (h, '_', t) not in self.filter_dict:
                self.filter_dict[(h, '_', t)] = [r]
            else:
                self.filter_dict[(h, '_', t)].append(r)
                
            if (h, r, '_') not in self.filter_dict:
                self.filter_dict[(h, r, '_')] = [t]
            else:
                self.filter_dict[(h, r, '_')].append(t)


    def split_transductive(self, p):
            msg, sup = [], []

            rels_encountered = np.zeros(self.num_relations)
            remaining_triplet_indexes = np.ones(self.num_triplets)

            for h,t in self.spanning:
                r = random.choice(self.rel_info[(h,t)])
                msg.append((h, r, t))
                remaining_triplet_indexes[self.triplets2id[(h,r,t)]] = 0
                rels_encountered[r] = 1


            for r in (1-rels_encountered).nonzero()[0].tolist():
                h,t = random.choice(self.pair_info[int(r)])
                msg.append((h, r, t))
                remaining_triplet_indexes[self.triplets2id[(h,r,t)]] = 0

            # start = time.time()

            sup = [self.triplets[idx] for idx, tf in enumerate(remaining_triplet_indexes) if tf]
            
            msg = np.array(msg)
            random.shuffle(sup)
            sup = np.array(sup)
            add_num = max(int(self.num_triplets * p) - len(msg), 0)
            msg = np.concatenate([msg, sup[:add_num]])
            sup = sup[add_num:]

            msg_inv = np.fliplr(msg).copy()
            msg_inv[:,1] += self.num_relations
            msg = np.concatenate([msg, msg_inv])

            return msg, sup

         # Train GraphSAGE
    def train_model_graph(self):
         gnn_gae_model = GraphAutoencoderGNN(self.node_features.shape[1], 32, 32)
         gnn_gae_optimizer = Adam(gnn_gae_model.parameters(), lr=self.args.learning_rate)
         print(f"Training {gnn_gae_model.__class__.__name__}...")
         pbar = tqdm(range(self.args.num_epoch_befor_train_kg))
         adj_true=self.adj_true
         loss_fn=loss_function
		 
         for epoch in pbar:
                gnn_gae_model.train()
                gnn_gae_optimizer.zero_grad()

                # Forward pass
                outputs = gnn_gae_model(self.model_graph, self.node_features)
                # Handle models with one or two outputs
                if isinstance(outputs, tuple):
                    embeddings, adj_pred = outputs
                else:
                    embeddings = outputs
                    adj_pred = None

                # Compute loss
                if loss_fn and adj_true is not None and adj_pred is not None:
                    loss = loss_fn(adj_pred, adj_true)
                else:
                    loss = torch.mean(embeddings)  # Placeholder loss for unsupervised training

                # Backward and optimize
                loss.backward()
                gnn_gae_optimizer.step()

                print(f"Epoch {epoch+1}/{self.args.num_epoch_befor_train_kg}, Loss: {loss.item():.4f}")
		 
         
         
         self.graph =add_model_feature_to_main_graph(embeddings,self.graph,self.ent_type)
				
         return embeddings
	
class ValidationData():
	def __init__(self, path, data_type = "valid"):
		self.path = path
		self.data_type = data_type
		self.ent2id = None
		self.rel2id = None
		self.id2ent, self.id2rel, self.msg_triplets, self.sup_triplets, self.filter_dict = self.read_triplet()
		self.num_ent, self.num_relations = len(self.id2ent), len(self.id2rel)
		

	def read_triplet(self):
		id2ent, id2rel, msg_triplets, sup_triplets = [], [], [], []
		total_triplets = []

		with open(self.path + "msg.txt", 'r') as f:
			for line in f.readlines():
				h, r, t = line.strip().split('\t')
				id2ent.append(h)
				id2ent.append(t)
				id2rel.append(r)
				msg_triplets.append((h, r, t))
				total_triplets.append((h, r, t))

		id2ent = remove_duplicate(id2ent)
		id2rel = remove_duplicate(id2rel)
		self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
		self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
		num_rel = len(self.rel2id)
		msg_triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in msg_triplets]
		msg_inv_triplets = [(t, r+num_rel, h) for h,r,t in msg_triplets]

		with open(self.path + self.data_type + ".txt", 'r') as f:
			for line in f.readlines():
				h, r, t = line.strip().split('\t')
				sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
				assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
					(self.ent2id[h], self.rel2id[r], self.ent2id[t]) 
				total_triplets.append((h,r,t))		
		for data_type in ['valid', 'test']:
			if data_type == self.data_type:
				continue
			with open(self.path + data_type + ".txt", 'r') as f:
				for line in f.readlines():
					h, r, t = line.strip().split('\t')
					assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
						(self.ent2id[h], self.rel2id[r], self.ent2id[t]) 
					total_triplets.append((h,r,t))	


		filter_dict = {}
		for triplet in total_triplets:
			h,r,t = triplet
			if ('_', self.rel2id[r], self.ent2id[t]) not in filter_dict:
				filter_dict[('_', self.rel2id[r], self.ent2id[t])] = [self.ent2id[h]]
			else:
				filter_dict[('_', self.rel2id[r], self.ent2id[t])].append(self.ent2id[h])

			if (self.ent2id[h], '_', self.ent2id[t]) not in filter_dict:
				filter_dict[(self.ent2id[h], '_', self.ent2id[t])] = [self.rel2id[r]]
			else:
				filter_dict[(self.ent2id[h], '_', self.ent2id[t])].append(self.rel2id[r])
				
			if (self.ent2id[h], self.rel2id[r], '_') not in filter_dict:
				filter_dict[(self.ent2id[h], self.rel2id[r], '_')] = [self.ent2id[t]]
			else:
				filter_dict[(self.ent2id[h], self.rel2id[r], '_')].append(self.ent2id[t])
		
		print(f"-----{self.data_type.capitalize()} Data Statistics-----")
		print(f"Message set has {len(msg_triplets)} triplets")
		print(f"Supervision set has {len(sup_triplets)} triplets")
		print(f"{len(self.ent2id)} entities, " + \
			  f"{len(self.rel2id)} relations, "+ \
			  f"{len(total_triplets)} triplets")

		msg_triplets = msg_triplets + msg_inv_triplets

		return id2ent, id2rel, np.array(msg_triplets), np.array(sup_triplets), filter_dict
       



        
        
        