
import torch
import torch.nn as nn
import torch.nn.functional as F
from rgcn_model import RGCN
from ent_init_model import EntInit
from model import GraphAutoencoderGNN ,WeightedGraphAutoEncoder,WeightedGraphAutoEncoder1
from tool import add_model_feature_to_main_graph
class TYPMODEL(nn.Module):
    def __init__(self, args,):
        super(TYPMODEL, self).__init__()
        self.args = args
        self.nrelation = args.num_rel
        self.emb_dim = args.emb_dim
        self.epsilon = 2.0


        self.gamma = torch.Tensor([args.margin])  
        self.rel_proj = nn.Linear(args.emb_dim, args.emb_dim*2, bias = True)   
        # self.rel_proj = nn.Linear(args.emb_dim, args.emb_dim, bias = True)   
        self.embedding_range = torch.Tensor([(self.gamma.item() + self.epsilon) / args.emb_dim])
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.args.dimension_relation))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.xavier_normal_(self.rel_proj.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.rel_proj.bias)
              # models
        self.ent_init = EntInit(args).to(args.gpu)
        # self.model_graph = GraphAutoencoderGNN(args)
        self.model_graph = WeightedGraphAutoEncoder(args)
        if args.add_rel_graph==1:
             self.rel_graph_model= WeightedGraphAutoEncoder1(args)
        self.rgcn = RGCN(args).to(args.gpu)
        # self.kge_model = KGEModel(args).to(args.gpu)
        self.model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
        }
       
        

    
            

    def forward(self,model_graph, main_graph, rel_graph,ent_type):
        self.ent_init(main_graph)
        node_embeddings, _ =self.model_graph(model_graph,model_graph.ndata["feat"],)
        if self.args.add_rel_graph==1:
             rel_embedding ,_ =  self.rel_graph_model(rel_graph,rel_graph.ndata["feat"] )
        else:
             rel_embedding= self.rel_proj(self.relation_embedding)
        # print(f"embedding size of model alocated is {embeddings.shape}")
        main_graph =add_model_feature_to_main_graph(node_embeddings,main_graph,ent_type)
        # print(f"feature size of main graph is  {main_graph.ndata['feat'].shape}")
        ent_embediing = self.rgcn(main_graph)
        return ent_embediing, rel_embedding

    def score(self, emb_ent, emb_rel, triplets):

        head_idxs = triplets[..., 0]
        rel_idxs = triplets[..., 1]
        tail_idxs = triplets[..., 2]
        head_embs = emb_ent[head_idxs.to(torch.int64)]
        tail_embs = emb_ent[tail_idxs.to(torch.int64)]
        rel_emb = emb_rel[[rel_idxs.to(torch.int64)]]
        # output = (head_embs * rel_embs * tail_embs).sum(dim = -1)
        
        output = self.model_func[self.args.score_fun](head_embs, rel_emb, tail_embs)
        return output
    def DistMult(self, head, relation, tail):
        
            score = (head * relation) * tail

            score = score.sum(dim=1)
            return score
    
        
    def TransE(self, head, relation, tail):
            score = (head + relation) - tail
            score =  - torch.norm(score, p=1, dim=1)
            return score
    def ComplEx(self, head, relation, tail, mode,tri=None):
            re_head, im_head = torch.chunk(head, 2, dim=2)
            re_relation, im_relation = torch.chunk(relation, 2, dim=2)
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)

            if mode == 'head-batch':
                re_score = re_relation * re_tail + im_relation * im_tail
                im_score = re_relation * im_tail - im_relation * re_tail
                score = re_head * re_score + im_head * im_score
            else:
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                score = re_score * re_tail + im_score * im_tail

            score = score.sum(dim=2)
            return score

    def RotatE(self, head, relation, tail, mode):
            pi = 3.14159265358979323846

            re_head, im_head = torch.chunk(head, 2, dim=2)
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)

            # Make phases of relations uniformly distributed in [-pi, pi]

            phase_relation = relation / (self.embedding_range.item() / pi)

            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)

            if mode == 'head-batch':
                re_score = re_relation * re_tail + im_relation * im_tail
                im_score = re_relation * im_tail - im_relation * re_tail
                re_score = re_score - re_head
                im_score = im_score - im_head
            else:
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                re_score = re_score - re_tail
                im_score = im_score - im_tail

            score = torch.stack([re_score, im_score], dim=0)
            score = score.norm(dim=0)

            score = self.gamma.item() - score.sum(dim=2)
            return score

