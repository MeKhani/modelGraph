from tqdm import tqdm
from model import EntInit
from rgcn_model import RGCN
from kge_model import KGEModel
import torch
import random
import torch.nn.functional as F
class TrainMainGraph():
    def __init__(self,main_graph ,args,triples):
        self.args = args
        # models
        # self.ent_init = EntInit(args).to(args.gpu)
        # self.rgcn = RGCN(args).to(args.gpu)
        self.main_graph = main_graph
        self.triples = torch.tensor(triples)
        self.graph = main_graph
        # self.neg_triplest = self.generate_negative_samples1(self.triples,main_graph.num_nodes(),3)
        self.neg_triplest = self.generate_neg(self.triples,main_graph.num_nodes(),self.args.num_neg)
        # self.kge_model = KGEModel(args).to(args.gpu)
           
    def generate_negative_samples(self, triples, num_entities):
        neg_samples = triples.clone()
        neg_samples[:, 2] = torch.randint(0, num_entities, (len(triples),))  # Random tail entity
        return neg_samples
    def train(self):
        pbr = tqdm(range(self.args.num_epoch))

        # Initialize model and optimizer
        num_nodes = self.main_graph.num_nodes()
        model = KGEModel(self.graph,num_nodes,self.args)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Loss function

        loss_fn = torch.nn.MarginRankingLoss(margin = self.args.margin, reduction = 'mean')

        # Training loop
        print(f"the size of negtive triplest is {self.neg_triplest.shape}")
        for epoch in pbr:
            model.train()

            # Compute positive scores
            pos_scores = model(self.triples)

            # Compute negative scores
            neg_scores = model(self.neg_triplest)
            

            # Create target tensor for ranking loss (+1 for positive, -1 for negative)
            target = torch.ones(pos_scores.size(0))
            loss = loss_fn(pos_scores.repeat(self.args.num_neg), neg_scores, torch.ones_like(neg_scores))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            # if ((epoch + 1) % valid_epochs) == 0:
            #     print("Validation")
            #     model.eval()
            #     val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(valid, valid.msg_triplets, \
            #                                                                             d_e, d_r, B)

            #     evaluate(model, valid, epoch, val_init_emb_ent, val_init_emb_rel, val_relation_triplets
        return model
            
    def evaluate_model(model, triples):
        model.eval()
        with torch.no_grad():
            scores = model(triples)
            sorted_indices = torch.argsort(-scores)  # Sort scores in descending order
            return sorted_indices  # Use this for computing metrics
    
    def generate_neg(self,triplets, num_ent, num_neg = 1):
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
    def generate_negative_samples1(self ,triples, num_entities, num_negatives=1):
        negatives = []
        for triple in triples:
            h, r, t = triple
            for _ in range(num_negatives):
                if random.random() < 0.5:
                    h_neg = random.randint(0, num_entities - 1)
                    negatives.append([h_neg, r, t])
                else:
                    t_neg = random.randint(0, num_entities - 1)
                    negatives.append([h, r, t_neg])
        return torch.tensor(negatives[:len(triples)])  # Ensure negatives match positives
