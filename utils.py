import torch
import torch.nn as nn
import torch.nn.functional as F

class ste_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantize(torch.nn.Module):
    def __init__(self, num_bits=8, symmetric=False, per_channel=False, weight=False):
        super().__init__()
        self.num_bits = num_bits
        # self.symmetric = symmetric
        # self.per_channel = per_channel
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buufer('zero_point', torch.tensor(0.))
        
        self.register_buffer('alpha', torch.tensor(0.))
        self.register_buffer('beta', torch.tensor(0.))
        
        self.weight = weight
        self.init = False
    
    @torch.no_grad()
    def init_range(self, x):
        self.alpha.data = x.amin()
        self.beta.data = x.amax()
    
    @torch.no_grad()
    def record_range(self, x):
        self.alpha.data = 0.99 * self.alpha + 0.01 * x.amin()
        self.beta.data = 0.99 * self.beta + 0.01 * x.amax()
    
    @torch.no_grad()
    def get_quantizers(self, a, b):
        scale = (b - a) / (2 ** self.num_bits - 1)
        zero_point = torch.round(-a / scale)
        return scale, zero_point
    
    def forward(self, x):
        if self.training:
            if not self.weight:
                if self.init is False:
                    self.init_range(x)
                    self.init = True
                else:
                    self.record_range(x)
            else:
                self.alpha.data, self.beta.data = x.amin(), x.amax()
            self.scale.data, self.zero_point.data = self.get_quantizers(self.alpha, self.beta)
        
        x = x / self.scale + self.zero_point
        x = ste_round.apply(torch.clamp(x, self.alpha, self.beta))
        x = x * self.scale + self.zero_point
        return x
    
    
class QuantConv2d(nn.Module):
    def __init__(self, conv, bn, activation=None):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.activation = activation if activation is not None else nn.Identity()
        
        self.aq_fn = Quantize()
        self.wq_fn = Quantize(weight=True)
        
        self.fold_bn = False
        
    def get_fold_params(self):
        scale = self.bn.weight / (self.bn.running_var + self.bn.eps).sqrt() 
        weight = self.conv.weight * scale.view(-1, 1, 1, 1)
        
        if self.conv_module.bias is not None:
            bias = (self.conv.bias - self.bn.running_mean) * scale + self.bn.bias
        else:
            bias = self.bn.bias - self.bn.running_mean * scale
        return weight, bias
        
    def forward(self, x):
        if self.fold_bn is False:
            x_q = self.aq_fn(x)
            w_q = self.wq_fn(self.conv.weight)
            out = F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
            
            out = self.bn(out)
        else:
            if self.training:
                x_q = self.aq_fn(x)
                out = F.conv2d(x_q, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
                _ = self.bn(out)
                w, b = self.get_fold_params()
                w_q = self.wq_fn(w)
                out = F.conv2d(x_q, w_q, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
            else:
                x_q = self.aq_fn(x)
                w, b = self.get_fold_params()
                w_q = self.wq_fn(w)
                out = F.conv2d(x_q, w_q, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        out = self.activation(out)
        return out
    

class QuantLinear(nn.Module):
    def __init__(self, fc, activation=None):
        super().__init__()
        self.fc = fc
        self.activation = activation if activation is not None else nn.Identity()
        
        self.aq_fn = Quantize()
        self.wq_fn = Quantize(weight=True)
        
    def forward(self, x):
        x_q = self.aq_fn(x)
        w_q = self.wq_fn(self.fc.weight)
        out = F.linear(x_q, w_q, self.fc.bias)
        out = self.activation(out)
        return out