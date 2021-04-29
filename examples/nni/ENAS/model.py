import torch.nn as nn

from nni.nas.pytorch import mutables
from ops import *


class ENASLayer(mutables.MutableScope):

    def __init__(self, key, prev_labels, out_shape):
        super().__init__(key)
        self.mutable = mutables.LayerChoice([
            block1(out_shape),
            block2(out_shape),
            block3(out_shape),
            block4(out_shape),
            block5(out_shape),
        ])
        if len(prev_labels) > 0:
            self.skipconnect = mutables.InputChoice(choose_from=prev_labels, n_chosen=None)
        else:
            self.skipconnect = None
        self.batch_norm = nn.BatchNorm1d(out_shape)

    def forward(self, prev_layers):
        out = self.mutable(prev_layers[-1])
        if self.skipconnect is not None:
            connection = self.skipconnect(prev_layers[:-1])
            if connection is not None:
                out += connection
                self.batch_norm(out)
        return out

class GeneralNetwork(nn.Module):
    def __init__(self, input_shape, out_shape, num_layers=6, num_classes=86 , regression = False, last_act = 'sigmoid'):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.out_shape = out_shape
        self.stem = nn.Sequential(
            nn.Linear(input_shape, out_shape),
            nn.BatchNorm1d(out_shape)
        )

        self.layers = nn.ModuleList()
        labels = []
        for layer_id in range(self.num_layers):
            labels.append("layer_{}".format(layer_id))
            self.layers.append(ENASLayer(labels[-1], labels[:-1], out_shape))

        self.dense1 = nn.Linear(out_shape, self.num_classes)
        self.regression = regression
        self.act = build_activation(last_act)

    def forward(self, x):
        cur = self.stem(x)
        layers = [cur]
        for layer_id in range(self.num_layers):
            cur = self.layers[layer_id](layers)
            layers.append(cur)
        logits = self.dense1(cur)
        if self.regression:
            return logits
        else:
            logits = self.act(logits)
            return logits
