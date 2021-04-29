import collections

import tensorflow as tf

from deephyper.nas.space import KSearchSpace
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.basic import Tensor
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting
from deephyper.nas.space.op.op1d import Dense, Identity


def swish(x, beta = 1):
    return (x * tf.keras.activations.sigmoid(beta * x))
tf.keras.utils.get_custom_objects().update({'swish':tf.keras.layers.Activation(swish) })#


def add_dense_to_(node):
    node.add_op(Identity()) # we do not want to create a layer in this case

    activations = [None, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, 'swish']
    for units in range(500, 1500, 200):
        for activation in activations:
            node.add_op(Dense(units=units, activation=activation))


def create_search_space(input_shape=(11937,),
                        output_shape=(86,),
                        num_layers=10,
                        *args, **kwargs):
    
    # print("input_shape:", input_shape, ", output_shape:", output_shape,", num_layers:", num_layers)
    arch = KSearchSpace(input_shape, output_shape)
    source = prev_input = arch.input_nodes[0]

    # look over skip connections within a range of the 3 previous nodes
    anchor_points = collections.deque([source], maxlen=3)

    for _ in range(num_layers):
        vnode = VariableNode()
        add_dense_to_(vnode)

        arch.connect(prev_input, vnode)

        # * Cell output
        cell_output = vnode

        cmerge = ConstantNode()
        cmerge.set_op(AddByProjecting(arch, [cell_output], activation='relu'))

        for anchor in anchor_points:
            skipco = VariableNode()
            skipco.add_op(Tensor([]))
            skipco.add_op(Connect(arch, anchor))
            arch.connect(skipco, cmerge)

        # ! for next iter
        prev_input = cmerge
        anchor_points.append(prev_input)

    cout = ConstantNode(op=Dense(86, activation="sigmoid"))
    arch.connect(prev_input, cout)
    return arch


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf

    search_space = create_search_space(num_layers=10)
    ops = [random() for _ in range(search_space.num_nodes)]

    print(f'This search_space needs {len(ops)} choices to generate a neural network.')

    search_space.set_ops(ops)

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file='sampled_neural_network.png', show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")


if __name__ == '__main__':
    test_create_search_space()