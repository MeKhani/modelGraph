import numpy as np
import pandas as pd 
import scipy.sparse as sp
import torch
import time
import random
import dgl
import os
import csv
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

def read_triplet_nell(path):
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

def read_triplet_fb(path):
    id2ent = set()
    id2rel = set()
    triplets = []
    list_spanning = []
    rel_info = defaultdict(list)
    pair_info = defaultdict(list)
    dicOfEntities={}
    ent_type={}
    type_ids ={}
    dicOfEntitiesIds={}
    # Read file and process lines
    with open(path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            id2ent.update([h, t])  # Add head and tail entities (set prevents duplicates)
            id2rel.add(r)          # Add relation
            triplets.append((h, r, t))
            classify_by_type_entity(dicOfEntities,r,h,t)
    type_ids = {type_en :id for id , type_en in enumerate(dicOfEntities.keys())}

    # Convert sets to lists to preserve order
    id2ent = sorted(id2ent)  
    id2rel = sorted(id2rel)  

    # Map entities and relations to unique IDs
    ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
    rel2id = {rel: idx for idx, rel in enumerate(id2rel)}

    # Map triplets to ID-based format and populate rel_info and pair_info
    dicOfEntitiesIds = convert_to_ids(dicOfEntities, ent2id, type_ids)
    ent_type = {ent: type_id for type_id, entities in dicOfEntitiesIds.items() for ent in entities}
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

    return id2ent, id2rel, triplets, ent2id, rel2id ,rel_info, pair_info,list_spanning,triplets2id,dicOfEntitiesIds,ent_type

def read_triplet_fb_v1(path):
    id2ent = set()
    id2rel = set()
    triplets = []
    list_spanning = []
    rel_info = defaultdict(list)
    pair_info = defaultdict(list)
    dicOfEntities={}
    ent_type={}
    type_ids ={}
    dicOfEntitiesIds={}
    # Read file and process lines
    path= path.replace(".txt",".tsv")
    with open(path, 'r', newline='', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')  # Specify '\t' as the delimiter
        for row in reader:
            if len(row) == 3:  # Ensure it has exactly 3 columns
                h, r, t = row
                id2ent.update([h, t])  # Add head and tail entities (set prevents duplicates)
                id2rel.add(r)          # Add relation
                triplets.append((h, r, t))
    ent_type1 ,type_ids= classify_by_type_entity_v1(path)
   
    # Convert sets to lists to preserve order
    print(f"number of type is {len(type_ids)}")
    id2ent = sorted(id2ent)  
    id2rel = sorted(id2rel)  

    # Map entities and relations to unique IDs
    ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
    rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
    print(f"the number of entites is {len(ent2id)}")

    # Map triplets to ID-based format and populate rel_info and pair_info

    ent_type ={id: type_ids[ent_type1[ent]] if ent in ent_type1 else len(type_ids) for ent, id in ent2id.items()}
    print(f"the number of entites type  is {len(ent_type)}")
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

    return id2ent, id2rel, triplets, ent2id, rel2id ,rel_info, pair_info,list_spanning,triplets2id,ent_type
def classify_by_type_entity_v1(path):
    """
    This function reads entity-type mappings from files and assigns unique IDs to each type.
    
    Args:
        path (str): The path to the dataset file (train.tsv or valid.tsv).
    
    Returns:
        dict: A dictionary mapping entities to their types.
        dict: A dictionary mapping types to unique IDs.
    """
    en_type = {}  # Map of entities to their types
    type_ids = {}  # Map of types to unique IDs
    count = 0  # Counter for assigning unique IDs to types
      # Process the dev file (shared logic for train and valid cases)
    if "train" in path:
        dev_path_type = path.replace("train.tsv", "dev-ents-class.txt")
    elif "valid" in path:
        dev_path_type = path.replace("valid.tsv", "dev-ents-class.txt")
    
    with open(dev_path_type, 'r') as f:
        for line in f:
            en, en_ty = line.strip().split('\t')
            # Only add to en_type if entity is not already mapped
            if en not in en_type:
                en_type[en] = en_ty
            if en_ty not in type_ids:  # Assign a unique ID if type is new
                type_ids[en_ty] = count
                count += 1

    # Determine the path to the type file
    if "train" in path:
        path_type1 = path.replace("train.tsv", "train-ents-class.txt")
        path_type2 = path.replace("train.tsv", "valid-ents-class.txt")
    elif "valid" in path:
        path_type1 = path.replace("valid.tsv", "valid-ents-class.txt")
        path_type2 = path.replace("valid.tsv", "train-ents-class.txt")
    else:
        raise ValueError("Invalid file path. Must contain 'train' or 'valid'.")

    # Process the first file (train or valid)
    with open(path_type1, 'r') as f:
       for line in f:
            en, en_ty = line.strip().split('\t')
            # Only add to en_type if entity is not already mapped
            if en not in en_type:
                en_type[en] = en_ty
            if en_ty not in type_ids:  # Assign a unique ID if type is new
                type_ids[en_ty] = count
                count += 1
    with open(path_type2, 'r') as f:
       for line in f:
            en, en_ty = line.strip().split('\t')
            # Only add to en_type if entity is not already mapped
            if en not in en_type:
                en_type[en] = en_ty
            if en_ty not in type_ids:  # Assign a unique ID if type is new
                type_ids[en_ty] = count
                count += 1

  

    return en_type, type_ids

      

def convert_to_ids(dicOfEntities, ent2id ,type_ids):
    """
    Convert entities in dicOfEntities to their corresponding IDs.

    Args:
        dicOfEntities (dict): Dictionary where keys are types and values are lists of entities.
        ent2id (dict): Dictionary mapping entities to unique IDs.

    Returns:
        dict: A new dictionary with the same keys (types), but entities replaced with their IDs.
    """
    dicOfEntitiesIds = {}

    for type_key, entities in dicOfEntities.items():
        # Convert entities to their IDs using ent2id
        dicOfEntitiesIds[type_ids[type_key]] = [ent2id[ent] for ent in entities if ent in ent2id]
    
    return dicOfEntitiesIds


def classify_by_type_entity(dicOfEntities, relation, head, tail):
    types = relation.strip().split('.')
    if len(types) == 1:  # Single-part relation
        t1 = types[0].strip().split('/')
        # print(f" lenght of types is {len(t1)}")
        if len(t1) > 2:  # Ensure t1[2] exists
            dicOfEntities.setdefault(t1[1], []).append(head)
        if len(t1) > 3:  # Ensure t1[3] exists
            dicOfEntities.setdefault(t1[2], []).append(tail)
    else:  # Multi-part relation
        t1 = types[0].strip().split('/')
        t2 = types[1].strip().split('/')
        if len(t1) > 2:
            dicOfEntities.setdefault(t1[1], []).append(head)
        if len(t2) > 2:
            dicOfEntities.setdefault(t2[1], []).append(tail)



def remove_duplicate(x):
	    return list(dict.fromkeys(x))
def divid_entities_by_type_nell(entities , en2id ):
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
def generate_group_triples(dic, triples,type_ent):
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
   

    # Generate (k1, rel, k2) triples
    group_triples = []
    for e1, rel, e2 in triples:
        k1 = type_ent[e1]  # Find the group for e1
        k2 = type_ent[e2]  # Find the group for e2


        if k1 is not None and k2 is not None:  # Only create triples if both entities are mapped
            if k1 != k2 :
                group_triples.append((k1, rel, k2))
                output_relations[k1].add(rel)
                input_relations[k2].add(rel)
            else:
                inner_relations[k1].add(rel)



    return group_triples ,inner_relations,output_relations,input_relations
from collections import defaultdict

def generate_group_triples_v1(triples, type_ent, num_rel):
    """
    Generate group-level triples and intra-group/outgoing/incoming relations.

    Args:
        triples (list): A list of triples of the form (e1, rel, e2).
        type_ent (dict): A dictionary mapping entities (e1, e2, ...) to their groups (k1, k2, ...).
        num_rel (int): Total number of relations.

    Returns:
        tuple: 
            - group_triples (set): A set of triples of the form (k1, rel_score, k2).
            - inner_relations (defaultdict): Relations within the same group (k1 -> relations).
            - output_relations (defaultdict): Relations leaving each group (k1 -> relations).
            - input_relations (defaultdict): Relations entering each group (k2 -> relations).
    """
    # Initialize relation tracking dictionaries
    group_relations = defaultdict(list)  # (k1, k2) -> list of rel
    inner_relations = defaultdict(set)   # k1 -> set of relations within the group
    output_relations = defaultdict(set)  # k1 -> set of outgoing relations
    input_relations = defaultdict(set)   # k2 -> set of incoming relations

    # Iterate through triples
    for e1, rel, e2 in triples:
        k1 = type_ent.get(e1)  # Group of e1
        k2 = type_ent.get(e2)  # Group of e2

        if k1 is not None and k2 is not None:  # Both entities are mapped to groups
            if k1 != k2:
                group_relations[(k1, k2)].append(rel)
                output_relations[k1].add(rel)
                input_relations[k2].add(rel)
            else:
                inner_relations[k1].add(rel)

    # Generate group triples with relation scores
    group_triples = {
        (k1, len(rels) / num_rel, k2)  # Score is the proportion of relations over total
        for (k1, k2), rels in group_relations.items()
    }

    return group_triples, inner_relations, output_relations, input_relations

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
    weight =torch.tensor([t[1] for t in triples] ) # relations

    # Create a DGL graph
    g = dgl.heterograph({
        ('node', 'weight', 'node'): (src_nodes, dst_nodes)
    })
    g.edata["weight"] = weight
    
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
    # main_graph.ndata['feat'] = feat[type_indices]+ main_graph.ndata['feat']

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


def weidner_dgl_graph(triples, pair_info):
    """
    Constructs a directed, weighted relation-relation graph using DGL.

    Args:
        triples (list of tuples): List of (head, relation, tail) triples from the KG.

    Returns:
        dgl.DGLGraph: A directed, weighted graph where nodes are relations,
                      and edge weights represent relation co-occurrences.
    """
    relation_to_triples = defaultdict(list)  # Store triples grouped by relation
    relation_to_id = {}  # Unique ID mapping for each relation
    relation_edges = []  # List of relation edges (source, target)
    edge_weights = []  # List of edge weights

    # # Group triples by relation
    # for h, r, t in triples:
    #     relation_to_triples[r].append((h, t))
    relation_to_triples = pair_info

    relations = list(relation_to_triples.keys())

    # Assign unique IDs to each relation (as nodes in DGL)
    # relation_to_id = {r: i for i, r in enumerate(relations)}

    # Iterate over all pairs of relations to compute weights
    for ra in relations:
        for rb in relations:
            if ra == rb:
                continue  # Skip self-relations for now
            
            direct_count = 0
            indirect_count = 0
            
            ra_pairs = set(relation_to_triples[ra])
            rb_pairs = set(relation_to_triples[rb])

            # Compute direct and indirect connections
            for h1, t1 in ra_pairs:
                for h2, t2 in rb_pairs:
                    if t1 == h2:  # Direct connection
                        direct_count += 1
                    elif h1 == h2 or t1 == t2:  # Indirect connection
                        indirect_count += 1

            weight_ra_rb = direct_count + indirect_count
            if weight_ra_rb > 0:
                relation_edges.append((ra, rb))
                edge_weights.append(weight_ra_rb)

    # Create a DGL graph
    src_nodes, dst_nodes = zip(*relation_edges) if relation_edges else ([], [])
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))

    # Assign edge weights as a feature
    g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)

    # Assign relation names as node features
    g.ndata['relation'] = torch.tensor(list(relation_to_id.values()))

    return g  # Return graph and relation mapping


import dgl
import torch
from collections import defaultdict

def weidner_dgl_graph_torch_fast( pair_info,args, device="cpu"):
    """
    Constructs a directed, weighted relation-relation graph using DGL with PyTorch optimizations.
    
    Args:
        triples (list of tuples): List of (head, relation, tail) triples from the KG.
        pair_info (dict): Dictionary mapping relations to entity pairs (head, tail).
        device (str): 'cpu' or 'cuda' for GPU acceleration.
    
    Returns:
        dgl.DGLGraph: A directed, weighted graph where nodes are relations,
                      and edge weights represent relation co-occurrences.
                      
    """
    
    # Extract unique relations & map them to node IDs
    relations = list(pair_info.keys())
    num_relations = len(relations)
    # relation_to_id = {r: i for i, r in enumerate(relations)}

    # Convert pair_info into tensor-based mappings for speed
    relation_entity_pairs = {
        r: torch.tensor(list(pairs), dtype=torch.long, device=device) for r, pairs in pair_info.items()
    }

    # Create adjacency matrix in PyTorch (Sparse Tensor for memory efficiency)
    adj_matrix = torch.zeros((num_relations, num_relations), dtype=torch.int32, device=device)

    # Convert relation mappings to index-based format for fast computation
    relation_sets = {r: set(map(tuple, pairs.tolist())) for r, pairs in relation_entity_pairs.items()}

    # **Optimized Direct & Indirect Computation Using Matrices**
    for i, ra in enumerate(relations):
        ra_pairs = relation_sets[ra]

        for j, rb in enumerate(relations):
            if i == j:
                continue  # Skip self-relations
            
            rb_pairs = relation_sets[rb]

            # Convert sets to Torch tensors
            if len(ra_pairs) == 0 or len(rb_pairs) == 0:
                continue  # Skip empty relations

            ra_pairs_tensor = relation_entity_pairs[ra]
            rb_pairs_tensor = relation_entity_pairs[rb]

            ra_heads, ra_tails = ra_pairs_tensor[:, 0], ra_pairs_tensor[:, 1]
            rb_heads, rb_tails = rb_pairs_tensor[:, 0], rb_pairs_tensor[:, 1]

            # **Compute Direct & Indirect Weights (Matrix Multiplication)**
            direct_count = (ra_tails.unsqueeze(1) == rb_heads.unsqueeze(0)).sum().item()
            indirect_count = (ra_heads.unsqueeze(1) == rb_heads.unsqueeze(0)).sum().item() + \
                             (ra_tails.unsqueeze(1) == rb_tails.unsqueeze(0)).sum().item()

            weight = direct_count + indirect_count
            if weight > 0:
                adj_matrix[i, j] = weight

    # **Convert Adjacency Matrix to DGL Graph (Efficiently)**
    src_nodes, dst_nodes = adj_matrix.nonzero(as_tuple=True)
    edge_weights = adj_matrix[src_nodes, dst_nodes].float()

    g = dgl.graph((src_nodes, dst_nodes), device=device)

    # Assign edge weights
    g.edata['weight'] = edge_weights
    g.ndata['feat'] = torch.ones(num_relations, args.emb_dim, device=device)

    return g  # Return graph 






def WiDNeR(structured_triples):
    structured_triples = structured_triples.replace(".txt",".tsv")
    structured_triples_ = pd.read_csv(structured_triples, header=None, names=['head','relation', 'tail'], delimiter='\t')


    ## different relations - direct
    result_direct = structured_triples_.merge(structured_triples_, left_on='tail', right_on='head', how='inner')
    diff_rels_direct = result_direct.loc[result_direct['relation_x'] != result_direct['relation_y']]
    diff_rels_direct.drop_duplicates(inplace=True)
    diff_rels_direct = diff_rels_direct.groupby(['relation_x', 'relation_y']).size().to_frame('#direct').reset_index()

    ## different relations - shared_head
    result_shared_head = structured_triples_.merge(structured_triples_, left_on='head', right_on='head', how='inner')
    diff_rels_shared_head = result_shared_head.loc[result_shared_head['relation_x'] != result_shared_head['relation_y']]
    diff_rels_shared_head.drop_duplicates(inplace=True)
    diff_rels_shared_head = diff_rels_shared_head.groupby(['relation_x', 'relation_y']).size().to_frame('#shared_head').reset_index()

    ## different relations - shared_tail
    result_shared_tail = structured_triples_.merge(structured_triples_, left_on='tail', right_on='tail', how='inner')
    diff_rels_shared_tail = result_shared_tail.loc[result_shared_tail['relation_x'] != result_shared_tail['relation_y']]
    diff_rels_shared_tail.drop_duplicates(inplace=True)
    diff_rels_shared_tail = diff_rels_shared_tail.groupby(['relation_x', 'relation_y']).size().to_frame('#shared_tail').reset_index()

    ## combine different relations - direct, shared_head, and shared_tail
    diff_rels = diff_rels_direct.merge(diff_rels_shared_head, on=['relation_x', 'relation_y'], how='outer')
    diff_rels = diff_rels.merge(diff_rels_shared_tail, on=['relation_x', 'relation_y'], how='outer')
    diff_rels.fillna(0, inplace=True)


    ## same relations - direct
    same_rels_direct = result_direct.loc[(result_direct['relation_x'] == result_direct['relation_y']) & ((result_direct['head_x'] != result_direct['tail_x']) | (result_direct['head_x'] != result_direct['tail_y'])) ]
    same_rels_direct.drop_duplicates(inplace=True)
    same_rels_direct = same_rels_direct.groupby(['relation_x', 'relation_y']).size().to_frame('#direct').reset_index()


    ## same relations - shared_head
    same_rels_shared_head = result_shared_head.loc[(result_shared_head['relation_x'] == result_shared_head['relation_y']) & (result_shared_head['tail_x'] != result_shared_head['tail_y'])]
    same_rels_shared_head.drop_duplicates(inplace=True)
    same_rels_shared_head = same_rels_shared_head.groupby(['relation_x', 'relation_y']).size().to_frame('#shared_head').reset_index()
    same_rels_shared_head['#shared_head'] = same_rels_shared_head['#shared_head'].map(lambda x: x/2)


    ## same relations - shared_tail
    same_rels_shared_tail = result_shared_tail.loc[(result_shared_tail['relation_x'] == result_shared_tail['relation_y']) & (result_shared_tail['head_x'] != result_shared_tail['head_y'])]
    same_rels_shared_tail.drop_duplicates(inplace=True)
    same_rels_shared_tail = same_rels_shared_tail.groupby(['relation_x', 'relation_y']).size().to_frame('#shared_tail').reset_index()
    same_rels_shared_tail['#shared_tail'] = same_rels_shared_tail['#shared_tail'].map(lambda x: x/2)


    ## combine same relations - direct, shared_head, and shared_tail
    same_rels = same_rels_direct.merge(same_rels_shared_head, on=['relation_x', 'relation_y'], how='outer')
    same_rels = same_rels.merge(same_rels_shared_tail, on=['relation_x', 'relation_y'], how='outer')
    same_rels.fillna(0, inplace=True)

    ## combine different and same relations
    diff_same = pd.concat([diff_rels, same_rels], ignore_index=True)

    diff_same['weight'] = diff_same['#direct'] + diff_same['#shared_head'] + diff_same['#shared_tail']

    rel_rel_net = diff_same.drop(columns=['#direct', '#shared_head', '#shared_tail'])

    return rel_rel_net



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





def extract_triples(all_data_pd):
    """
    Extracts triples (x_index, relation, y_index) and assigns unique IDs to entities and relations.
    
    Args:
        all_data_pd (pd.DataFrame): DataFrame with ['x_index', 'relation', 'y_index', 'x_type']

    Returns:
        list: Triples with relation IDs
        dict: Mapping of entity indices to type IDs
        dict: Pair information mapping relation to (x, y) pairs
        dict: Mapping (x, y) to list of relations (rel_info)
        int: Number of unique relations
        list: Spanning tree edges
    """

    # ✅ Unique ID mappings for entity types & relations
    ent_type_id = {etype: idx for idx, etype in enumerate(all_data_pd["x_type"].unique())}
    ent_x_id = {en: idx for idx, en in enumerate(all_data_pd["x_name"].unique())}
    ent_y_id = {en: idx for idx, en in enumerate(all_data_pd["y_name"].unique())}
    relations_id = {rel: idx for idx, rel in enumerate(all_data_pd["relation"].unique())}
    num_rel = len(relations_id)  # Count unique relations
    all_en = set(ent_x_id.keys()).union(set(ent_y_id.keys()))
    ent_id = {en: idx for idx, en in enumerate(all_en)}


    # ✅ Extract triples with numerical relation IDs (FAST)
    triples = all_data_pd[['x_name', 'relation', 'y_name']].to_numpy()
    triples_id = [(ent_id[x], relations_id[r],ent_id[y]) for x, r, y in triples]
    triplets2id = {tuple(tri): idx for idx, tri in enumerate(triples_id)}
    print(f"the number entityies is {len(ent_id)}")

    # ✅ Efficient relation-to-pair mapping
    pair_info = defaultdict(set)
    rel_info = defaultdict(set)  

    for x, r, y in triples_id:
        pair_info[r].add((x, y))  # Store relation-specific (x, y) pairs
        rel_info[(x, y)].add(r)  # Store all relations for (x, y)

    # ✅ Extract entity-type mapping (no duplicates)
    ent_type_x = {ent_id[row['x_name']]: ent_type_id[row['x_type']] for _, row in all_data_pd[['x_name', 'x_type']].drop_duplicates().iterrows()}
    ent_type_y= {ent_id[row['y_name']]: ent_type_id[row['y_type']] for _, row in all_data_pd[['y_name', 'y_type']].drop_duplicates().iterrows()}
    # Merge keys from both dictionaries and assign new unique indices
    all_types = set(ent_type_x.keys()).union(set(ent_type_y.keys()))
    print(f"the number of ent_type_x  {len(ent_type_x)} ")
    print(f"the number of ent_type_y  {len(ent_type_y)} ")
    print(f"size of  all_type is {len(all_types)}")

    # Create a new dictionary with unique indices
    ent_type = {**ent_type_x, **ent_type_y}
    print(f"all_type is {ent_type}")
    print(f"the number of entity type is : {len(ent_type)}")
    print(f"the number of triplest is : {len(triples)}")
    # print(f"the numer of entity type id is : {len(ent_type_id)}")

    # ✅ Build Graph & Compute Spanning Tree
    list_spanning = []
    graph_data = np.array(triples_id)[:, [0, 2]]  # Extract (head, tail) pairs
    G_ent = igraph.Graph.TupleList(graph_data, directed=True)
    spanning_tree = G_ent.spanning_tree()
    G_ent.delete_edges(spanning_tree.get_edgelist())

    # ✅ Extract Spanning Tree Edges
    list_spanning = [(spanning_tree.vs[e.source]["name"], spanning_tree.vs[e.target]["name"]) for e in spanning_tree.es]

    return triples_id, ent_type, pair_info, rel_info, num_rel, list_spanning,triplets2id


   


    # # Convert defaultdict to regular dictionary
    # print(f"entity type is {clean_dict}")
    # print(f"entity type is id  {ent_type}")

    
    