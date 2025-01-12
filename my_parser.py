import argparse
import json

def parse(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "dataset/", type = str)
    # parser.add_argument('--data_name', default = 'NELL-995-v1', type = str)
    parser.add_argument('--data_name', default = 'nell_v1', type = str)
    parser.add_argument('--exp', default = 'exp', type = str)
    parser.add_argument('-m', '--margin', default = 2, type = float)
    parser.add_argument('-lr', '--learning_rate', default=5e-4, type = float)
    parser.add_argument('-nle', '--num_layer_ent', default = 2, type = int)
    parser.add_argument('-nlr', '--num_layer_rel', default = 2, type = int)
    parser.add_argument('-nr', '--num_rel', default = 2, type = int)
    parser.add_argument('-d_e', '--dimension_entity', default = 32, type = int)
    parser.add_argument('-d_r', '--dimension_relation', default = 32, type = int)
    parser.add_argument('-hdr_e', '--hidden_dimension_ratio_entity', default = 8, type = int)
    parser.add_argument('-hdr_r', '--hidden_dimension_ratio_relation', default = 4, type = int)
    parser.add_argument('-b', '--num_bin', default = 10, type = int)
    parser.add_argument('-e', '--num_epoch', default = 100, type = int)
    parser.add_argument('-e_1', '--num_epoch_befor_train_kg', default = 100, type = int)
    if test:
        parser.add_argument('--target_epoch', default = 6600, type = int)
    parser.add_argument('-v', '--validation_epoch', default = 4, type = int)
    parser.add_argument('--num_head', default = 8, type = int)
    parser.add_argument('--num_neg', default = 10, type = int)
    parser.add_argument('--best', action = 'store_true')


# params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)
    #params for model graph 
    parser.add_argument('--input_feat_model_graph', default=42, type=int)



    parser.add_argument('--name', default='fb237_v1_transe', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='../dataset/deep-con/tb_log', type=str)
    parser.add_argument('--gpu', default='cpu', type=str)
    parser.add_argument('--metatrain_bs', default=64, type=int)
    parser.add_argument('--num_train_subgraph', default=100)
    parser.add_argument('--metatrain_num_neg', default=32)
    parser.add_argument('--adv_temp', default=1, type=float)
    parser.add_argument('--metatrain_check_per_step', default=5, type=int)
    parser.add_argument('--num_valid_subgraph', default=200)
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