import logging

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.model import CallableModelWrapper, Model
from cleverhans import utils

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

_logger = utils.create_logger("cleverhans.attacks.MadryEtAl")
_logger.setLevel(logging.INFO)


class MadryEtAl(Attack):

    """
    The Projected Gradient Descent Attack (Madry et al. 2017).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(self, model, sess, dtypestr='float32', **kwargs):
        """
        Create a MadryEtAl instance.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(MadryEtAl, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables
        adv_x = self.attack(x, labels)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.01, nb_iter=40, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, rand_init=True, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import model_loss, clip_eta

        adv_x = x + eta
        preds = self.model.get_logits(adv_x)
        loss = model_loss(y, preds)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta

    def attack(self, x, y):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.

        :param x: A tensor with the input image.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        if self.rand_init:
            eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
            eta = clip_eta(eta, self.ord, self.eps)
        else:
            eta = tf.zeros_like(x)

        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, y)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x