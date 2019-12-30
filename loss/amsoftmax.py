import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AMSoftmax(nn.Module):
    def __init__(self):
        super(AMSoftmax, self).__init__()

    def forward(self, input, target, scale=10.0, margin=0.35):
        # self.it += 1
        cos_theta = input[0]
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B, Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        output = cos_theta * 1.0  # size=(B, Classnum)
        output[index] -= margin
        output = output * scale

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss
