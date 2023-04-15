import torch as t

class ConvQuant(nn.Conv):
    def __init__(self, in_ch, out_ch, isBase): # 输入，输出通道，卷积是否带base
        super.__init__(in_ch, out_ch, isBase, staticStepNum)

        self.weight_quant = Quant(in_ch, out_ch, isBase)
        self.input_quant = Quant(in_ch, out_ch, isBase)

        self.step = 0
        self.staticStepNum = staticStepNum

    def forward(self, x):
        if self.init_flag:
            if self.step < self.staticStepNum:
                self.weight_quant.init_static(self.weight)
                self.input_quant.init_static(x)
            elif self.step == self.staticStepNum:
                self.weight_quant.init(self.weight)
                self.input_quant.init(x)

            out = self._conv_forward(x, self.weight, self.bias)
            self.step += 1
        else:
            quant_weight = self.weight_quant(self.weight)
            quant_input = self.input_quant(x)
            out = self._conv_forward(quant_input, quant_weight, self.bias)

        return out

class TConvQuant(nn.Conv):
    def __init__(self, in_ch, out_ch, isBase): # 输入，输出通道，卷积是否带base
        super.__init__(in_ch, out_ch, isBase, staticStepNum)

        self.weight_quant = Quant(in_ch, out_ch, isBase)
        self.input_quant = Quant(in_ch, out_ch, isBase)

        self.step = 0
        self.staticStepNum = staticStepNum

    def forward(self, x):
        if self.init_flag:
            if self.step < self.staticStepNum:
                self.weight_quant.init_static(self.weight)
                self.input_quant.init_static(x)
            elif self.step == self.staticStepNum:
                self.weight_quant.init(self.weight)
                self.input_quant.init(x)

            out = F.conv_transpose(x, self.weight, self.bias, self.stride, )
            self.step += 1
        else:
            quant_weight = self.weight_quant(self.weight)
            quant_input = self.input_quant(x)
            out = F.conv_transpose(x, self.weight, self.bias, self.stride, )

        return out

class LinearQuant(nn.Conv):
    def __init__(self, in_ch, out_ch, isBase): # 输入，输出通道，卷积是否带base
        super.__init__(in_ch, out_ch, isBase, staticStepNum)

        self.weight_quant = Quant(in_ch, out_ch, isBase)
        self.input_quant = Quant(in_ch, out_ch, isBase)

        self.step = 0
        self.staticStepNum = staticStepNum

    def forward(self, x):
        if self.init_flag:
            if self.step < self.staticStepNum:
                self.weight_quant.init_static(self.weight)
                self.input_quant.init_static(x)
            elif self.step == self.staticStepNum:
                self.weight_quant.init(self.weight)
                self.input_quant.init(x)

            out = F.Linear(x, self.weight, self.bias, self.stride, )
            self.step += 1
        else:
            quant_weight = self.weight_quant(self.weight)
            quant_input = self.input_quant(x)
            out = F.Linear(x, self.weight, self.bias, self.stride, )

        return out

class Quantizer(t.nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x):
        return x
