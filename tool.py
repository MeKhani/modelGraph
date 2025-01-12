import numpy as np
import scipy.sparse as sp
import torch
import time
import random
import dgl
import os
from collections import defaultdict
import igraph


def read_data(path):
    
    """read file and return the triples containing its ground truth label (0/1)"""
    
    f = open(path)
    triples_with_label = []
    for line in f:
        triple_with_label = line.strip().split("\t")
        triples_with_label.append(triple_with_label)
    f.close()
    return triples_with_label

def write_dic(write_path, array):
    
    """generate a dictionary"""
    
    f = open(write_path, "w+")
    for i in range(len(array)):
        f.write("{}\t{}\n".format(i, array[i]))
    f.close()
    print("saved dictionary to {}".format(write_path))
    
def dictionary(input_list):
    
    """
    To generate a dictionary.
    Index: item in the array.
    Value: the index of this item.
    """
    
    return dict(zip(input_list, range(len(input_list))))

def normalize(mx):
    
    """Row-normalize sparse matrix"""
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_entites_dic(path ):
    entities = {}

    with open(path, 'r') as file:
        for line in file:
            # Split each line into ID and label
            entity_id, entity_label = line.strip().split()
            # Add to dictionary
            entities[entity_id] = entity_label
    return entities


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def split_known(triples):
    
    """
    Further split the triples into 2 sets:
    1. an incomplete graph: known
    2. a set of missing facts we want to recover: unknown
    """
    
    DATA_LENGTH = len(triples)
    split_ratio = [0.9, 0.1]
    candidate = np.array(range(DATA_LENGTH))
    np.random.shuffle(candidate)
    idx_known = candidate[:int(DATA_LENGTH * split_ratio[0])]
    idx_unknown = candidate[int(DATA_LENGTH * split_ratio[0]):]
    known = []
    unknown = []
    for i in idx_known:
        known.append(triples[i])
    for i in idx_unknown:
        unknown.append(triples[i])
    return known, unknown

def read_triplet(path):
    id2ent = set()
    id2rel = set()
    triplets = []
    list_spanning = []
    rel_info = defaultdict(list)
    pair_info = defaultdict(list)

    # Read file and process lines
    with open(path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            id2ent.update([h, t])  # Add head and tail entities (set prevents duplicates)
            id2rel.add(r)          # Add relation
            triplets.append((h, r, t))

    # Convert sets to lists to preserve order
    id2ent = sorted(id2ent)  
    id2rel = sorted(id2rel)  

    # Map entities and relations to unique IDs
    ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
    rel2id = {rel: idx for idx, rel in enumerate(id2rel)}

    # Map triplets to ID-based format and populate rel_info and pair_info
    triplets = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triplets]
    triplets2id = {tri: idx for idx, tri in enumerate(triplets)}
    for h, r, t in triplets:
        rel_info[(h, t)].append(r)  # Store all relations for a (head, tail) pair
        pair_info[r].append((h, t))  # Store all (head, tail) pairs for a relation
    G = igraph.Graph.TupleList(np.array(triplets)[:, 0::2])
    G_ent = igraph.Graph.TupleList(np.array(triplets)[:, 0::2], directed = True)
    spanning = G_ent.spanning_tree()
    G_ent.delete_edges(spanning.get_edgelist())
    for e in spanning.es:
        e1,e2 = e.tuple
        e1 = spanning.vs[e1]["name"]
        e2 = spanning.vs[e2]["name"]
        list_spanning.append((e1,e2))

    return id2ent, id2rel, triplets, ent2id, rel2id ,rel_info, pair_info,list_spanning,triplets2id


def remove_duplicate(x):
	    return list(dict.fromkeys(x))
def divid_entities(entities , en2id ):
     dicOfEntities={}
     dicOfEntitiesIds={}
     type_id = {}
     ent_type ={}
     count = -1
     for en , val in en2id.items():
          consept, typeOfEn, en1= en.strip().split(':')
          if typeOfEn in dicOfEntities:
                    dicOfEntities[typeOfEn].append(en1)
                    dicOfEntitiesIds[type_id[typeOfEn]].append(val)
          else:
                count +=1
                type_id[typeOfEn] = count
                dicOfEntities[typeOfEn]  = [en]
                dicOfEntitiesIds[type_id[typeOfEn]]  = [val]
          ent_type[val] =type_id[typeOfEn] 
    #  print (typeOfEn+"\n")
     return dicOfEntities , dicOfEntitiesIds,ent_type
def exrtract_relation_in_types_entities(triple_entities, en_dic_id):
    inner_relations_for_every_type = {}
    outer_recived_relations_for_every_type = {}
    outer_sent_relations_for_every_type = {}

    for key, val in en_dic_id.items():
        for en in val:
            # Collect unique relations where `en` is a subject or object
            inner_relations_subject = {relation for subj, relation, obj in triple_entities if subj == en and obj in val}
            outer_relations_subject = {relation for subj, relation, obj in triple_entities if subj == en and obj not in  val}
            # objects_en = {obj for subj, relation, obj in triple_entities if subj == en and obj not in  val}
            inner_relations_object =  {relation for subj, relation, obj in triple_entities if obj == en and subj in val}
            outer_relations_object =  {relation for subj, relation, obj in triple_entities if obj == en and subj not in val}
            # subject_en =  {subj for subj, relation, obj in triple_entities if obj == en and subj not in val}
            

            # Update received relations dictionary
            if inner_relations_subject:
                if key not in inner_relations_for_every_type:
                    inner_relations_for_every_type[key] = inner_relations_subject 
                else:
                    inner_relations_for_every_type[key].update(inner_relations_subject) 
            if outer_relations_subject:
                if key not in outer_recived_relations_for_every_type:
                    outer_recived_relations_for_every_type[key] = outer_relations_subject 
                else:
                    outer_recived_relations_for_every_type[key].update(outer_relations_subject)
            
            # Update sent relations dictionary
            if inner_relations_object:
                if key not in inner_relations_for_every_type:
                    inner_relations_for_every_type[key] = inner_relations_object 
                else:
                    inner_relations_for_every_type[key].update(inner_relations_object)
            if outer_relations_object:
                if key not in outer_sent_relations_for_every_type:
                    outer_sent_relations_for_every_type[key] = outer_relations_object 
                else:
                    outer_sent_relations_for_every_type[key].update(outer_relations_object)
    

    return inner_relations_for_every_type,outer_recived_relations_for_every_type,outer_sent_relations_for_every_type
def generate_group_triples(dic, triples):
    """
    Generate triples of the form (k1, rel, k2) from a dictionary and a list of triples.
    
    Args:
        dic (dict): A dictionary where keys are groups (k1, k2, ...) and 
                    values are sets/lists of entities (e1, e2, ...).
        triples (list): A list of triples of the form (e1, rel, e2).
    
    Returns:
        list: A list of triples of the form (k1, rel, k2).
    """
    # Invert the dictionary to map entities to their corresponding keys
    entity_to_group = {}
    inner_relations = defaultdict(set)  # For intra-group relations
    output_relations = defaultdict(set)  # For intra-group relations
    input_relations = defaultdict(set)  # For intra-group relations
    for group, entities in dic.items():
        for entity in entities:
            entity_to_group[entity] = group

    # Generate (k1, rel, k2) triples
    group_triples = []
    for e1, rel, e2 in triples:
        k1 = entity_to_group.get(e1)  # Find the group for e1
        k2 = entity_to_group.get(e2)  # Find the group for e2


        if k1 is not None and k2 is not None:  # Only create triples if both entities are mapped
            if k1 != k2 :
                group_triples.append((k1, rel, k2))
                output_relations[k1].add(rel)
                input_relations[k2].add(rel)
            else:
                inner_relations[k1].add(rel)



    return group_triples ,inner_relations,output_relations,input_relations
def exrtract_relation_in_types_entities1(triple_entities, en_dic_id):
    entity_type_triples = []
    inner_relations_for_every_type = {}
    outer_recived_relations_for_every_type = {}
    outer_sent_relations_for_every_type = {}

    for key, val in en_dic_id.items():
        for en in val:
            # Collect unique relations where `en` is a subject or object
            inner_relations_subject = {relation for subj, relation, obj in triple_entities if subj == en and obj in val}
            outer_relations_subject = {relation for subj, relation, obj in triple_entities if subj == en and obj not in  val}
            en_rel_obj =  {(rel ,obj ) for subj, rel, obj in triple_entities if subj == en and obj not in  val}
            inner_relations_object =  {relation for subj, relation, obj in triple_entities if obj == en and subj in val}
            outer_relations_object =  {relation for subj, relation, obj in triple_entities if obj == en and subj not in val}
            subj_rel_en=  {(subj ,rel ) for subj, rel, obj in triple_entities if obj == en and subj not in val}
            for (rel , obj) in en_rel_obj:
                keys = {k for k, val in en_dic_id.items() if obj in val}
                triples= {(key, rel, k) for k in keys}
                entity_type_triples.append(triples) 
            

            # Update received relations dictionary
            if inner_relations_subject:
                if key not in inner_relations_for_every_type:
                    inner_relations_for_every_type[key] = inner_relations_subject 
                else:
                    inner_relations_for_every_type[key].update(inner_relations_subject) 
            if outer_relations_subject:
                if key not in outer_recived_relations_for_every_type:
                    outer_recived_relations_for_every_type[key] = outer_relations_subject 
                else:
                    outer_recived_relations_for_every_type[key].update(outer_relations_subject)
            
            # Update sent relations dictionary
            if inner_relations_object:
                if key not in inner_relations_for_every_type:
                    inner_relations_for_every_type[key] = inner_relations_object 
                else:
                    inner_relations_for_every_type[key].update(inner_relations_object)
            if outer_relations_object:
                if key not in outer_sent_relations_for_every_type:
                    outer_sent_relations_for_every_type[key] = outer_relations_object 
                else:
                    outer_sent_relations_for_every_type[key].update(outer_relations_object)
    

    return inner_relations_for_every_type,outer_recived_relations_for_every_type,outer_sent_relations_for_every_type, entity_type_triples
def create_entity_type_triple(outer_sent_relations_for_every_type,outer_recived_relations_for_every_type):
     entity_type_triples = []
     for obj , relations in outer_sent_relations_for_every_type.items():
          for rel in relations:
               subj_ent_types = [en_type for en_type, val in outer_recived_relations_for_every_type.items() if rel in val  ]
               for subj  in subj_ent_types:
                    entity_type_triples.append((obj,rel,subj))
                    
          
     
     return entity_type_triples
def create_graph(triples):
     # Extract node and edge information
    src_nodes = [t[0] for t in triples]  # subjects
    dst_nodes = [t[2] for t in triples]  # objects
    edge_types = [t[1] for t in triples]  # relations

    # Create a DGL graph
    g = dgl.heterograph({
        ('node', 'relation_type', 'node'): (src_nodes, dst_nodes)
    })
    
    return g
def create_main_graph(triples):
     # Extract node and edge information
    src_nodes = [t[0] for t in triples]  # subjects
    dst_nodes = [t[2] for t in triples]  # objects
    edge_types = [t[1] for t in triples]  # relations

    # Create a DGL graph
    g = dgl.heterograph({
        ('node', 'relation_type', 'node'): (src_nodes, dst_nodes)
    })
    g.edata['type'] = torch.tensor(edge_types)
    
    return g
def add_model_feature_to_main_graph(feat, main_graph, enttype):
    """
    Assign node features to the main graph based on entity types.

    Args:
        feat (torch.Tensor): Feature matrix for all types, shape: (num_types, feature_dim).
        main_graph (DGLGraph): The graph to which features will be assigned.
        enttype (dict): A dictionary mapping node IDs to type IDs.

    Returns:
        DGLGraph: The graph with node features assigned.
    """
    # Step 1: Validate input
    num_nodes = main_graph.num_nodes()
    if len(enttype) != num_nodes:
        raise ValueError(f"Number of nodes in enttype ({len(enttype)}) does not match main_graph ({num_nodes})")

    # Step 2: Create a mapping tensor of type indices
    type_indices = torch.tensor(
        [enttype[node_id] for node_id in range(num_nodes)],
        dtype=torch.long
    )

    # Step 3: Assign features using the type indices
    main_graph.ndata['feat'] = torch.cat((feat[type_indices],main_graph.ndata['feat']) ,dim=1)

    # Debug: Print summary of assigned features
    # print(f"Assigned features to {num_nodes} nodes. Feature shape: {main_graph.ndata['feat'].shape}")
    
    return main_graph

def add_feature_to_graph_nodes(graph, i_r, output_relations, input_relations, num_rel):
    """
    Add features to graph nodes based on input, output, and relation-based features.

    Args:
        graph (DGLGraph): Input graph.
        i_r (dict): Dictionary mapping nodes to their internal relations.
        output_relations (dict): Dictionary mapping nodes to their output relations.
        input_relations (dict): Dictionary mapping nodes to their input relations.
        num_rel (int): Number of relations.

    Returns:
        DGLGraph: Graph with updated node features.
    """
    # Initialize node features with zeros
    num_nodes = graph.num_nodes()
    graph.ndata["feat"] = torch.zeros(num_nodes, num_rel * 3)

    # Batch update internal relation features
    if i_r:
        for node, rels in i_r.items():
            graph.ndata["feat"][node, list(rels)] = 1

    # Batch update output relation features
    if output_relations:
        for node, rels in output_relations.items():
            graph.ndata["feat"][node, [num_rel + r for r in rels]] = 1

    # Batch update input relation features
    if input_relations:
        for node, rels in input_relations.items():
            graph.ndata["feat"][node, [2 * num_rel + r for r in rels]] = 1

    return graph


def add_feature_to_graph_edges(graph, entity_type_triples, num_relations):
    # Initialize edge features with zeros
    rel_feat = torch.zeros(graph.num_edges(), num_relations)
    
    # Create a lookup set for quick membership testing
    entity_triples_set = set(entity_type_triples)
    
    # Get all edges
    src_nodes, dst_nodes = graph.edges()
    
    # Iterate through relations and update features
    for rel in range(num_relations):
        mask = [
            (src.item(), rel, dst.item()) in entity_triples_set
            for src, dst in zip(src_nodes, dst_nodes)
        ]
        rel_feat[torch.tensor(mask), rel] = 1

    # Assign edge features
    graph.edata["rel_feat"] = rel_feat

    return graph
def set_seed(seed=1):
    """Set the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
def generate_neg(triplets, num_ent, num_neg = 1):
	neg_triplets = triplets.unsqueeze(dim=1).repeat(1,num_neg,1)
	neg_triplets = neg_triplets.float()
	# rand_result = torch.rand((len(triplets),num_neg)).cuda()
	rand_result = torch.rand((len(triplets),num_neg))
	perturb_head = rand_result < 0.5
	perturb_tail = rand_result >= 0.5
	# rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets),num_neg)).cuda()
	rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets),num_neg))
	rand_idxs = rand_idxs.float()
	# rand_idxs[perturb_head] += rand_idxs[perturb_head] >= neg_triplets[:,:,0][perturb_head]
	rand_idxs[perturb_head] += (rand_idxs[perturb_head] >= neg_triplets[:, :, 0][perturb_head]).float()

	# rand_idxs[perturb_tail] += rand_idxs[perturb_tail] >= neg_triplets[:,:,2][perturb_tail]
	rand_idxs[perturb_tail] += (rand_idxs[perturb_tail] >= neg_triplets[:,:,2][perturb_tail]).float()
	neg_triplets[:,:,0][perturb_head] = rand_idxs[perturb_head]
	neg_triplets[:,:,2][perturb_tail] = rand_idxs[perturb_tail]
	neg_triplets = torch.cat(torch.split(neg_triplets, 1, dim = 1), dim = 0).squeeze(dim = 1)

	return neg_triplets
def get_rank(triplet, scores, filters, target = 0):
	thres = scores[triplet[0,target]].item()
	scores[filters] = thres - 1
	rank = (scores > thres).sum() + (scores == thres).sum()//2 + 1
	return rank.item()

def get_metrics(rank):
	rank = np.array(rank, dtype = int)
	mr = np.mean(rank)
	mrr = np.mean(1 / rank)
	hit10 = np.sum(rank < 11) / len(rank)
	hit3 = np.sum(rank < 4) / len(rank)
	hit1 = np.sum(rank < 2) / len(rank)
	return mr, mrr, hit10, hit3, hit1

