import torch
from tool import get_rank, get_metrics
from tqdm import tqdm
import json


def evaluate(my_model, target, epoch, init_emb_ent, init_emb_rel, relation_triplets):
    with torch.no_grad():
        my_model.eval()
        # msg = torch.tensor(target.msg_triplets).cuda()
        # sup = torch.tensor(target.sup_triplets).cuda()
        msg = torch.tensor(target.msg_triplets)
        sup = torch.tensor(target.sup_triplets)

        emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)

        head_ranks = []
        tail_ranks = []
        ranks = []
        for triplet in tqdm(sup):
            triplet = triplet.unsqueeze(dim = 0)

            head_corrupt = triplet.repeat(target.num_ent, 1)
            head_corrupt[:,0] = torch.arange(end = target.num_ent)
            
            head_scores = my_model.score(emb_ent, emb_rel, head_corrupt)
            head_filters = target.filter_dict[('_', int(triplet[0,1].item()), int(triplet[0,2].item()))]
            head_rank = get_rank(triplet, head_scores, head_filters, target = 0)

            tail_corrupt = triplet.repeat(target.num_ent, 1)
            tail_corrupt[:,2] = torch.arange(end = target.num_ent)

            tail_scores = my_model.score(emb_ent, emb_rel, tail_corrupt)
            tail_filters = target.filter_dict[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]
            tail_rank = get_rank(triplet, tail_scores, tail_filters, target = 2)

            ranks.append(head_rank)
            head_ranks.append(head_rank)
            ranks.append(tail_rank)
            tail_ranks.append(tail_rank)


        print("--------LP--------")
        mr, mrr, hit10, hit3, hit1 = get_metrics(ranks)
        print(f"MR: {mr:.1f}")
        print(f"MRR: {mrr:.3f}")
        print(f"Hits@10: {hit10:.3f}")
        print(f"Hits@1: {hit1:.3f}")

def evaluate1( my_model, target,epoch):
    with torch.no_grad():
        my_model.eval()
        # msg = torch.tensor(target.msg_triplets).cuda()
        # sup = torch.tensor(target.sup_triplets).cuda()
        # msg = torch.tensor(target.msg_triplets)
        # sup = torch.tensor(target.sup_triplets)
        emb_ent, emb_rel = my_model(target.model_graph,target.graph, target.rel_graph,target.ent_type)

        head_ranks = []
        tail_ranks = []
        ranks = []
        result ={}
        triplets=torch.tensor(target.triples)
        for triplet in triplets:
            triplet = triplet.unsqueeze(dim = 0)

            head_corrupt = triplet.repeat(target.num_ent, 1)
           
            head_corrupt[:,0] = torch.arange(end = target.num_ent)
            
            head_scores = my_model.score(emb_ent, emb_rel, head_corrupt)
            # print(f"the head_score : {head_scores.shape}")
            head_filters = target.filter_dict[('_', int(triplet[0,1].item()), int(triplet[0,2].item()))]
            head_rank = get_rank(triplet, head_scores, head_filters, target = 0)

            tail_corrupt = triplet.repeat(target.num_ent, 1)
            tail_corrupt[:,2] = torch.arange(end = target.num_ent)

            tail_scores = my_model.score(emb_ent, emb_rel, tail_corrupt)
            tail_filters = target.filter_dict[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]
            tail_rank = get_rank(triplet, tail_scores, tail_filters, target = 2)

            ranks.append(head_rank)
            head_ranks.append(head_rank)
            ranks.append(tail_rank)
            tail_ranks.append(tail_rank)


        print("--------LP--------")
        mr, mrr, hit10, hit3, hit1 = get_metrics(ranks)
        print(f"MR: {mr:.1f}")
        print(f"MRR: {mrr:.3f}")
        print(f"Hits@10: {hit10:.3f}")
        print(f"Hits@1: {hit1:.3f}")
        result['MR']= mr
        result['MRR']= mrr
        result['Hits@10']= hit10
        result['Hits@3']= hit3
        result['Hits@1']= hit1
        result['epoch']= epoch
        return result
def write_evaluation_result( clear_first,results, path):
      if clear_first :
          with open(path,"w") as f:
              pass
      
      with open(path, "a") as f:  # Open the file in append mode
        f.write(json.dumps(results) + "\n")  # Write the data as a single JSON line