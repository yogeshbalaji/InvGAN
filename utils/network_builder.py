# Copyright 2018 The Defense-GAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Modified for Defense-GAN:
- Added the ReconstructionLayer class for cleverhans.
- The different model architectures that are tested in the paper.

Modified version of cleverhans/model.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cleverhans.model import Model
from cleverhans.picklable_model import Conv2D, ReLU, Flatten, Linear, Dropout, Softmax
from cleverhans.picklable_model import Layer, MLP
from utils.reconstruction import Reconstructor
from models.gan_v2 import InvertorDefenseGAN
import numpy as np
import tensorflow as tf


class ReconstructionLayer(Layer):
    """This layer is used as a wrapper for Defense-GAN's reconstruction
    part.
    """

    def __init__(self, model, input_shape, batch_size, ae_flag, **kwargs):
        """Constructor of the layer.

        Args:
            model: `Callable`. The generator model that gets an input and
                reconstructs it. `def gen(Tensor) -> Tensor.`
            input_shape: `List[int]`.
            batch_size: int.
        """
        super(ReconstructionLayer, self).__init__(**kwargs)
        self.rec_model = model
        self.rec_obj = Reconstructor(model)
        self.version = 'v2' if isinstance(model, InvertorDefenseGAN) else 'v1'

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.name = 'reconstruction'
        self.ae_flag = ae_flag

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x, **kwargs):
        x.set_shape(self.input_shape)
        if self.ae_flag == True:
            self.rec = self.rec_model.autoencode(
                x, batch_size=self.batch_size)
        elif self.version == 'v2':
            self.rec, self.rec_zs = self.rec_obj.reconstruct(x, batch_size=self.batch_size, back_prop=False, reconstructor_id=123)
        else:
            self.rec, self.rec_zs = self.rec_model.reconstruct(
                x, batch_size=self.batch_size, back_prop=False, reconstructor_id=123)

        return self.rec

    def get_params(self):
        return []


class DefenseMLP(MLP):
    def __init__(self, layers, input_shape, feature_layer='none'):
        super(DefenseMLP, self).__init__(layers, input_shape)
        self.feature_layer = feature_layer

    def add_rec_model(self, reconstructor_model, batch_size, ae_flag=False):
        rec_layer = ReconstructionLayer(reconstructor_model, self.input_shape, batch_size, ae_flag)
        rec_layer.set_input_shape(self.input_shape)
        self.layers = [rec_layer] + self.layers
        self.layer_names = [rec_layer.name] + self.layer_names

        self.layers[0].parent = "input"
        self.layers[1].parent = rec_layer.name

    def extract_feature(self, x):
        outputs = self.fprop(x)

        return outputs[self.feature_layer]

    @property
    def rec(self):
        return self.layers[0].rec

    @property
    def rec_zs(self):
        return self.layers[0].rec_zs


class DefenseWrapper(Model):
    """A wrapper that turns a callable into a valid Model"""
    def __init__(self, callable_fn, output_layer):
        """
        Wrap a callable function that takes a tensor as input and returns
        a tensor as output with the given layer name.
        :param callable_fn: The callable function taking a tensor and
                            returning a given layer as output.
        :param output_layer: A string of the output layer returned by the
                             function. (Usually either "probs" or "logits".)
        """

        super(DefenseWrapper, self).__init__()
        self.output_layer = output_layer
        self.callable_fn = callable_fn
        self.rec_model = None

    def fprop(self, x, **kwargs):
        assert self.output_layer == 'logits'

        if self.rec_model is not None:
            if self.ae_flag is True:
                self._rec = self.rec_model.autoencode(x, batch_size=self.batch_size)
            elif self.version == 'v2':
                self._rec, self._rec_zs = self.rec_obj.reconstruct(x, batch_size=self.batch_size, back_prop=False, reconstructor_id=123)
            else:
                self._rec, self._rec_zs = self.rec_model.reconstruct(x, batch_size=self.batch_size, back_prop=False, reconstructor_id=123)

        else:
            self._rec = x

        output = self.callable_fn(self._rec, **kwargs)

        assert output.op.type != 'Softmax'

        return {self.output_layer: output}

    def add_rec_model(self, reconstructor_model, batch_size, ae_flag=False):
        self.rec_model = reconstructor_model
        self.rec_obj = Reconstructor(reconstructor_model)
        self.version = 'v2' if isinstance(reconstructor_model, InvertorDefenseGAN) else 'v1'

        self.batch_size = batch_size
        self.ae_flag = ae_flag

    def extract_feature(self, x):
        # Build graph by forward propagation, and then access intermediate features
        _ = self.callable_fn(x)

        return self.callable_fn.feature

    @property
    def rec(self):
        return self._rec

    @property
    def rec_zs(self):
        return self._rec_zs


def model_f(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME", use_bias=True),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID", use_bias=True),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID", use_bias=True),
              ReLU(),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = DefenseMLP(layers, input_shape)
    return model


def model_e(input_shape=(None, 28, 28, 1), nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """

    # Define a fully connected model (it's different than the black-box).
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return DefenseMLP(layers, input_shape)


def model_d(input_shape=(None, 28, 28, 1), nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Dropout(0.5),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return DefenseMLP(layers, input_shape)


def model_b(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1)):
    layers = [Dropout(0.2),
              Conv2D(nb_filters, (8, 8), (2, 2), "SAME", use_bias=True),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID", use_bias=True),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID", use_bias=True),
              ReLU(),
              Dropout(0.5),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = DefenseMLP(layers, input_shape)
    return model


def model_a(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (5, 5), (1, 1), "SAME", use_bias=True),
              ReLU(),
              Conv2D(nb_filters, (5, 5), (2, 2), "VALID", use_bias=True),
              ReLU(),
              Flatten(),
              Dropout(0.25),
              Linear(128),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = DefenseMLP(layers, input_shape, feature_layer='ReLU7')
    return model


def model_c(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME", use_bias=True),
              ReLU(),
              Conv2D(nb_filters, (5, 5), (2, 2), "VALID", use_bias=True),
              ReLU(),
              Flatten(),
              Dropout(0.25),
              Linear(128),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = DefenseMLP(layers, input_shape)
    return model


def model_y(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Linear(256),
              ReLU(),
              Dropout(0.5),
              Linear(256),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = DefenseMLP(layers, input_shape, feature_layer='ReLU13')
    return model


def model_q(nb_filters=32, nb_classes=10,
            input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Linear(256),
              ReLU(),
              Dropout(0.5),
              Linear(256),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = DefenseMLP(layers, input_shape)
    return model


def model_z(nb_filters=32, nb_classes=10,
            input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(4 * nb_filters, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(4 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Linear(600),
              ReLU(),
              Dropout(0.5),
              Linear(600),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = DefenseMLP(layers, input_shape)
    return model
