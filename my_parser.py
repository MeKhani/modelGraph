import argparse
import json

def parse(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "dataset/", type = str)
    # parser.add_argument('--data_name', default = 'NELL-995-v1', type = str)
    # parser.add_argument('--data_name', default = 'nell_v1', type = str)
    # parser.add_argument('--data_name', default = 'fb237_v1', type = str)
    parser.add_argument('--data_name', default = 'primekg', type = str)
    # parser.add_argument('--data_name', default = 'FB15k-237', type = str)
    parser.add_argument('--exp', default = 'exp', type = str)
    parser.add_argument('-m', '--margin', default = 2, type = float)
    parser.add_argument('-md', '--mainData', default = True , type = float)
    parser.add_argument('-lr', '--learning_rate', default=5e-4, type = float)
    parser.add_argument('-nle', '--num_layer_ent', default = 2, type = int)
    parser.add_argument('-nlr', '--num_layer_rel', default = 2, type = int)
    parser.add_argument('-nr', '--num_rel', default = 2, type = int)
    parser.add_argument('-d_e', '--dimension_entity', default = 32, type = int)
    parser.add_argument('-d_r', '--dimension_relation', default = 32, type = int)
   
    parser.add_argument('-e', '--num_epoch', default = 100, type = int)
    parser.add_argument('-e_1', '--num_epoch_befor_train_kg', default = 100, type = int)
   
    parser.add_argument('-v', '--validation_epoch', default = 4, type = int)
    parser.add_argument('--num_head', default = 8, type = int)
    parser.add_argument('--num_neg', default = 10, type = int)
    parser.add_argument('--best', action = 'store_true')


    parser.add_argument('--add_rel_graph', default = 0 ,choices=[0,1] , help="1 when you want to add relational graph and 0 else ", type = int)
# params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)
    #params for model graph 
    parser.add_argument('--input_feat_model_graph', default=42, type=int)
    parser.add_argument('--gpu', default='cpu', type=str)
    parser.add_argument('--score_fun', default='DistMult', type=str, choices=['TransE','TypeEn', 'DistMult', 'ComplEx', 'RotatE'])


    if not test:
        parser.add_argument('--no_write', action = 'store_true')

    args = parser.parse_args()

    if test and args.best:
        remaining_args = []
        with open(f"./ckpt/best/{args.data_name}/config.json") as f:
            configs = json.load(f)
        for key in vars(args).keys():
            if key in configs:
                vars(args)[key] = configs[key]
            else:
                remaining_args.append(key)

    return args