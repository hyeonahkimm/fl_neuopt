import torch
from torch import nn
from einops import rearrange
from nets.graph_layers import MultiHeadEncoder, EmbeddingNet, MultiHeadPosCompat, kopt_Decoder

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
class Actor(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_layers,
                 normalization,
                 v_range,
                 seq_length,
                 k,
                 with_RNN,
                 with_feature1,
                 with_feature3,
                 with_simpleMDP,
                 with_timestep=False,
                 T_max = 100,
                 ):
        super(Actor, self).__init__()

        problem_name = problem.NAME
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.seq_length = seq_length                
        self.k = k
        self.with_RNN = with_RNN
        self.with_feature1 = with_feature1
        self.with_feature3 = with_feature3
        self.with_simpleMDP = with_simpleMDP
        self.with_timestep = with_timestep
        
        if problem_name == 'tsp':
            self.node_dim = 2
        elif problem_name == 'cvrp':
            self.node_dim = 8 if self.with_feature1 else 6
        else:
            raise NotImplementedError()
        
        if with_timestep:
            # From https://github.com/zdhNarsil/Diffusion-Generative-Flow-Samplers
            self.register_buffer(
                "timestep_coeff", torch.linspace(start=0.1, end=100, steps=self.embedding_dim)[None]
            )
            self.timestep_phase = nn.Parameter(torch.randn(self.embedding_dim)[None])
            
            self.timestep_embed = nn.Sequential(
                nn.Linear(2 * self.embedding_dim, self.embedding_dim),
                nn.GELU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
        
            
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim,
                            self.seq_length)
        
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor, 
                                self.embedding_dim, 
                                self.hidden_dim,
                                number_aspect = 2,
                                normalization = self.normalization
                                )
            for _ in range(self.n_layers)))

        self.pos_encoder = MultiHeadPosCompat(self.n_heads_actor, 
                                self.embedding_dim, 
                                self.hidden_dim, 
                                )
        
        self.decoder = kopt_Decoder(self.n_heads_actor, 
                                    input_dim = self.embedding_dim, 
                                    embed_dim = self.embedding_dim,
                                    v_range = self.range,
                                    k = self.k,
                                    with_RNN = self.with_RNN,
                                    with_feature3 = self.with_feature3,
                                    simpleMDP = self.with_simpleMDP
                                    )

        print('# params in Actor', self.get_parameter_number())
        
    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, problem, batch, x_in, solution, context, context2,last_action, fixed_action = None, require_entropy = False, to_critic = False, only_critic  = False, time_cond=None):
        # the embedded input x
        bs, gs, in_d = x_in.size()
        
        if problem.NAME == 'cvrp':
            
            visited_time, to_actor = problem.get_dynamic_feature(solution, batch, context)
            if self.with_feature1:
                x_in = torch.cat((x_in, to_actor), -1)
            else:
                x_in = torch.cat((x_in, to_actor[:,:,:-2]), -1)
            del context, to_actor

        elif problem.NAME == 'tsp':
            visited_time = problem.get_order(solution, return_solution = False)
        else: 
            raise NotImplementedError()
        
        
        h_embed, h_pos = self.embedder(x_in, solution, visited_time)
        aux_scores = self.pos_encoder(h_pos)
        
        if self.with_timestep and time_cond is not None:
            cond = time_cond.view(-1, 1).expand((x_in.shape[0], 1))
            sin_embed_cond = torch.sin(
                # (1, channels) * (bs, 1) + (1, channels)
                (self.timestep_coeff * cond.float()) + self.timestep_phase
            )
            cos_embed_cond = torch.cos(
                (self.timestep_coeff * cond.float()) + self.timestep_phase
            )
            embed_cond = self.timestep_embed(
                rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
            )
            
            h_embed += embed_cond.unsqueeze(1).repeat(1, h_embed.shape[1], 1)
        
        h_em_final, _ = self.encoder(h_embed, aux_scores)
        
        if only_critic:
            return (h_em_final)
        
        action, log_ll, entropy = self.decoder(problem,
                                               h_em_final,
                                               solution,
                                               context2,
                                               visited_time,
                                               last_action,
                                               fixed_action = fixed_action,
                                               require_entropy = require_entropy)
        
        # assert (visited_time == visited_time_clone).all()
        if require_entropy:
            return action, log_ll, (h_em_final) if to_critic else None, entropy
        else:
            return action, log_ll, (h_em_final) if to_critic else None