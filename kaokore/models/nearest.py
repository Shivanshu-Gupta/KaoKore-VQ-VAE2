import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F

class NearestEmbedFunc(Function):
    """
    Args:
        x: (batch_size, emb_dim, *)
            Last dimensions may be arbitrary
        emb: (emb_dim, n_embs)

    Returns:
        x_emb: (batch_size, emb_dim, *)
            Nearest embeddings
        argmin: (batch_size, *)
            Embedding index
    """
    @staticmethod
    def forward(ctx, input, emb, return_argmin=False):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.n_embs = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1))\
            .view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        if return_argmin: return result.contiguous(), argmin
        else: return result.contiguous()

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.n_embs).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.n_embs), 0)
        return grad_input, grad_emb, None


nearest_embed = NearestEmbedFunc().apply

class NearestEmbed(nn.Module):
    """
    Get nearest embedding to input
    Args:
        x: (batch_size, emb_dim, *)
            Last dimensions may be arbitrary
        emb: (emb_dim, n_embs)

    Returns:
        x_emb: (batch_size, emb_dim, *)
            Nearest embeddings
        argmin: (batch_size, *)
            Embedding index
    """
    def __init__(self, n_embs, emb_dim):
        super(NearestEmbed, self).__init__()
        self.n_embs = n_embs
        self.emb_dim = emb_dim
        self.embs = nn.Parameter(torch.rand(emb_dim, n_embs))

    def forward(self, x, emb_sg=False, return_argmin=False):
        return nearest_embed(x, self.embs.detach() if emb_sg else self.embs, return_argmin)

    def extra_repr(self):
        return f'emb_dim={self.emb_dim}, n_embs={self.n_embs}'
