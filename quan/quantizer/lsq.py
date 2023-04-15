import torch as t

from .quantizer import Quantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, shift=False):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        
        self.per_channel = per_channel
        self.shift = shift
        self.init_flag = False
        self.s = t.nn.Parameter(t.ones(1))
        if shift:
            self.shift_value = t.nn.Paramter(t.ones(1))
        self.sum = None
        
    def static_from(self, x):
        if self.sum == None:
            self.sum = x
        else:
            self.sum += x
        
    def init_from(self, x, step, *args, **kwargs):
        x = self.sum / step
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            if self.shift:
                shift_init = (t.max(x) - t.min(x)) / (self.thd_pos - self.thd_neg)
                x_max = t.max(x)
                self.shift_value = x_max - self.thd_pos * shift_init
           
             self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            
        self.init_flag = True

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)
        x = x / (s_scale + eps)
        
        if self.shift:
            shift_grad_scale = 1.0 / (x.numel() ** 0.5)
            shift_scaled = grad_scale(self.shift_value, shift_grad_scale)
            x = round_pass(x) - round_pass(shift_scaled / (s_scale + eps))
        else:
             x = round_pass(x)
                
        x = t.clamp(x, self.thd_neg, self.thd_pos)        
        if self.shift:
            x = x * s_scale + round_pass(shift_scaled / (s_scale + eps)) * s_scale
       else:
            x = x * s_scale
        return x
