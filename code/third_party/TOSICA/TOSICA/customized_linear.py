#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From Uchida Takumi https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
extended torch.nn module which cusmize connection.
This code base on https://pytorch.org/docs/stable/notes/extending.html
"""
import math
import torch
import torch.nn as nn


class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True, embed_dim=48):
        """
        extended torch.nn module which mask connection.
        Args:
        mask [torch.tensor / np.ndarray]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = int(mask.shape[0])
        self.output_features = int(mask.shape[1])
        if isinstance(mask, torch.Tensor):
            m_np = mask.detach().cpu().numpy().astype('float32')
        else:
            m_np = mask.astype('float32')

        # Detect block-sparse pathway structure (mask repeated by embed_dim along axis=1)
        D = embed_dim if (self.output_features % embed_dim == 0) else 1
        P = self.output_features // D
        p_mask = m_np[:, ::D]
        col_counts = p_mask.sum(axis=0).astype(int)
        sp_cols = [c for c in range(P) if col_counts[c] <= min(256, self.input_features - 1)]
        dn_cols = [c for c in range(P) if col_counts[c] > min(256, self.input_features - 1)]
        self.num_patches = P
        self.embed_dim = D
        self.sp_cols = sp_cols
        self.dn_cols = dn_cols
        self.col_counts = col_counts

        if len(sp_cols) > 0:
            K_max = max(1, max(col_counts[c] for c in sp_cols))
            idx_pad = torch.zeros(len(sp_cols), K_max, dtype=torch.long)
            mask_pad = torch.zeros(len(sp_cols), K_max, 1, dtype=torch.float32)
            for i, c in enumerate(sp_cols):
                nz = (p_mask[:, c] > 0).nonzero()[0]
                if len(nz) > 0:
                    idx_pad[i, :len(nz)] = torch.from_numpy(nz.astype('int64'))
                    mask_pad[i, :len(nz), 0] = 1.0
            self.register_buffer('idx_pad', idx_pad)
            self.register_buffer('mask_pad', mask_pad)
            self.W_sp = nn.Parameter(torch.Tensor(len(sp_cols), K_max, D))
        else:
            self.register_parameter('W_sp', None)

        if len(dn_cols) > 0:
            mask_dn = torch.zeros(len(dn_cols), D, self.input_features, dtype=torch.float32)
            for i, c in enumerate(dn_cols):
                mask_dn[i] = torch.from_numpy(p_mask[:, c]).unsqueeze(0).expand(D, -1)
            self.register_buffer('mask_dn', mask_dn)
            self.W_dn = nn.Parameter(torch.Tensor(len(dn_cols), D, self.input_features))
        else:
            self.register_parameter('W_dn', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_features)
        if self.W_sp is not None:
            self.W_sp.data.uniform_(-stdv, stdv)
            self.W_sp.data.mul_(self.mask_pad)
        if self.W_dn is not None:
            self.W_dn.data.uniform_(-stdv, stdv)
            self.W_dn.data.mul_(self.mask_dn)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """ Same as reset_parameters, but only initialize to positive values. """
        stdv = 1. / math.sqrt(self.input_features)
        if self.W_sp is not None:
            self.W_sp.data.uniform_(0, stdv)
            self.W_sp.data.mul_(self.mask_pad)
        if self.W_dn is not None:
            self.W_dn.data.uniform_(0, stdv)
            self.W_dn.data.mul_(self.mask_dn)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def full_weight(self):
        ref = self.W_sp if self.W_sp is not None else self.W_dn
        full_w = torch.zeros(
            self.num_patches, self.embed_dim, self.input_features,
            device=ref.device, dtype=ref.dtype,
        )
        if self.W_sp is not None:
            w_sp = (self.W_sp * self.mask_pad).detach()
            for i, c in enumerate(self.sp_cols):
                k = self.col_counts[c]
                if k > 0:
                    full_w[c, :, self.idx_pad[i, :k]] = w_sp[i, :k, :].t()
        if self.W_dn is not None:
            w_dn = (self.W_dn * self.mask_dn).detach()
            for i, c in enumerate(self.dn_cols):
                full_w[c] = w_dn[i]
        return full_w.reshape(self.output_features, self.input_features)

    def forward(self, input):
        B = input.shape[0]
        out = input.new_empty(B, self.num_patches, self.embed_dim)
        if self.W_sp is not None:
            x_sp = input[:, self.idx_pad]
            out[:, self.sp_cols, :] = torch.einsum('bpk,pkd->bpd', x_sp, self.W_sp * self.mask_pad)
        if self.W_dn is not None:
            w_dn_2d = (self.W_dn * self.mask_dn).reshape(-1, self.input_features)
            out[:, self.dn_cols, :] = torch.nn.functional.linear(input, w_dn_2d).view(B, len(self.dn_cols), self.embed_dim)
        out = out.reshape(B, self.output_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )





if __name__ == 'check grad':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    customlinear = CustomizedLinearFunction.apply

    input = (
            torch.randn(20,20,dtype=torch.double,requires_grad=True),
            torch.randn(30,20,dtype=torch.double,requires_grad=True),
            None,
            None,
            )
    test = gradcheck(customlinear, input, eps=1e-6, atol=1e-4)
    print(test)
