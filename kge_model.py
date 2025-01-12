import torch.nn as nn
import torch.nn.functional as F
import torch

class KGEModel(nn.Module):
    def __init__(self,graph, num_nodes, args):
        super(KGEModel, self).__init__()
        self.entity_embeddings = nn.Embedding(num_nodes, args.dimension_entity)
        print(f"the size of entity embedding is  : {self.entity_embeddings}")
        self.relation_embeddings = nn.Embedding(args.num_rel, args.dimension_entity)
        self.embedding_dim = args.dimension_entity
        self.args= args
        self.graph = graph
        self.triples=[]

        # Initialize embeddings
        # nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        self.model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'TypeEn': self.TypeEn,
            'RotatE': self.RotatE,
        }

    def score(self, head, relation, tail):
        """
        Compute the score of a triple (head, relation, tail).
        """
        self.triple = [head,relation,tail]
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)

        # TransE scoring function: ||h + r - t||
        # score = -torch.norm(head_emb + relation_emb - tail_emb, p=1, dim=1)
        score = self.model_func[self.args.score_fun](head_emb, relation_emb, tail_emb)

        return score

    def forward(self, triples):
        """
        Compute scores for a batch of triples.
        """
        head, relation, tail = triples[:, 0], triples[:, 1], triples[:, 2]
        head = head.long()
        relation = relation.long()
        tail = tail.long()
        return self.score(head, relation, tail)
    def DistMult(self, head, relation, tail):
       
        score = (head * relation) * tail

        score = score.sum(dim=1)
        return score
    def TypeEn(self, head, relation, tail):
        feat = self.graph.ndata["feat"].detach()

        h,r,t =self.triple
        score = (feat[h] * head + relation )- (feat[t] * tail)
        score =  - torch.norm(score, p=1, dim=1)
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
