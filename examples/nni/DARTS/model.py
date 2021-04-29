import torch
import torch.nn as nn

from nni.nas.pytorch import mutables
from ops import *

from collections import OrderedDict

class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, out_shape):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                mutables.LayerChoice(OrderedDict([
                    ("block1", block1(out_shape)),
                    ("block2", block2(out_shape)),
                    ("block3", block3(out_shape)),
                    ("block4", block4(out_shape)),
                    ("block5", block5(out_shape))
                ]), key=choice_keys[-1]))
        self.input_switch = mutables.InputChoice(choose_from=choice_keys, n_chosen=2, key="{}_switch".format(node_id))

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        return self.input_switch(out)


class Cell(nn.Module):

    def __init__(self, n_nodes,channels_pp, channels_p, out_shape):
        super().__init__()
        self.n_nodes = n_nodes
        self.preproc0 = nn.Sequential(nn.Linear(channels_pp, out_shape),
                                      nn.BatchNorm1d(out_shape),
                                      Swish())
        self.preproc1 = nn.Sequential(nn.Linear(channels_p, out_shape),
                                      nn.BatchNorm1d(out_shape),
                                      Swish())

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("normal", depth), depth, out_shape))

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)
#             print([i.shape for i in tensors])
        output = torch.cat(tensors[2:], dim=1)
        return output


class GeneralNetwork(nn.Module):

    def __init__(self, input_shape, out_shape, n_layers=4, n_nodes=4, num_classes=86 , regression = False, last_act = 'sigmoid'):
        super().__init__()
        self.num_classes = num_classes
        self.n_layers = n_layers
        
        self.stem = nn.Linear(input_shape, out_shape)
        channels_pp, channels_p, c_cur = out_shape, out_shape, out_shape
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            cell = Cell(n_nodes,channels_pp, channels_p, out_shape)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

        self.dense1 = nn.Linear(channels_p, self.num_classes)
        self.regression = regression
        self.act = build_activation(last_act)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        logits = self.dense1(s1)
        if self.regression:
            return logits
        else:
            logits = self.act(logits)
            return logits