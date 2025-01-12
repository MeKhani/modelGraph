import os
import numpy as np
import torch
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm
from train_valid_data import TrainData ,ValidationData
from tool import set_seed ,generate_neg ,create_main_graph
from initialize import initialize
from evaluation import evaluate1
from my_parser import parse
from ingram_model import InGram
from modelType import TYPMODEL









def main():
    """Main entry point of the program."""
    # Parse arguments and set seed
    args = parse()
    set_seed()
    # Load data Train Data and train  graph 
    train_path = os.path.join(args.data_path, args.data_name, "train.txt")
    valid_path = os.path.join(args.data_path, args.data_name, "valid.txt")
    train_data = TrainData(args,train_path)
    # print(f"number relation in train data is {args.num_rel}")
    # print(f"number relation in train data is {args.num_rel}")
    valid_data = TrainData(args,valid_path)
    path = args.data_path + args.data_name + "/"
   
    # embedding  = train_data.train_model_graph()

    # print(f"the embedding is size  {embedding.shape}")
    if not args.no_write:
        os.makedirs(f"./dataset/ckpt/{args.exp}/{args.data_name}", exist_ok=True)
    file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
                f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
                f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
                f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
                f"_head_{args.num_head}_margin_{args.margin}"
    
    epochs = args.num_epoch
    valid_epochs = args.validation_epoch
    num_neg = args.num_neg
    type_model = TYPMODEL(args)
    loss_fn = torch.nn.MarginRankingLoss(margin = args.margin, reduction = 'mean')
    optimizer1 = torch.optim.Adam(type_model.parameters(), lr = args.learning_rate)
    pbar = tqdm(range(epochs))
    total_loss = 0
    args.input_feat_model_graph =train_data.model_graph.ndata["feat"] 
    # return 
    for epoch in pbar:
        optimizer1.zero_grad()
        msg, sup = train_data.split_transductive(0.75)
        # init_emb_ent, init_emb_rel, relation_triplets = initialize(train_data, msg, d_e, d_r, B,add_feat= True)

        msg = torch.tensor(msg)
        sup = torch.tensor(sup)
        msg_graph = create_main_graph(msg)
        emb_ent ,emb_rel = type_model(train_data.model_graph,train_data.graph,train_data.ent_type)
        # print(f"emb_ent : { emb_ent}")
        # print(f"emb_rel : { emb_rel}")
        pos_scores = type_model.score(emb_ent, emb_rel, sup)
        neg_scores = type_model.score(emb_ent, emb_rel, generate_neg(sup, train_data.num_ent, num_neg = num_neg))
        
        loss = loss_fn(pos_scores.repeat(num_neg), neg_scores, torch.ones_like(neg_scores))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(type_model.parameters(), max_norm=0.1)
        # torch.nn.utils.clip_grad_norm_(my_model.parameters(), 0.1, error_if_nonfinite = False)

        optimizer1.step()
        total_loss += loss.item()
        # print(f"loss : {loss }")
        # print(f"total loss : {total_loss }")
        pbar.set_description(f"loss {loss.item()}")	

        if ((epoch + 1) % valid_epochs) == 0:
            # print("Validation")
            type_model.eval()
            # val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(validation_data, validation_data.msg_triplets, \
                                                                                    # d_e, d_r, B)

            evaluate1(type_model, valid_data)

            if not args.no_write:
                torch.save({'model_state_dict': type_model.state_dict(), \
                            'optimizer_state_dict': optimizer1.state_dict(), }, \
                    f"dataset/ckpt/{args.exp}/{args.data_name}/{file_format}_{epoch+1}.ckpt")

            type_model.train()

    

   
main()

