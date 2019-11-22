import cPickle
import tensorflow as tf
from classifiers.cifar_model import Model as CIFARModel
import utils
import numpy as np
import inception
import fid


def ComputeClassificationAccuracy(images, recons, labels, args, debug=True):
    model_paths = {'CIFAR': 'classifiers/model/cifar-10', 
                   'CelebA': 'classifiers/model/celeba'}
    batch_size = 50
    
    dset = utils.data_loader(images, recons, labels, batch_size)
    
    # normalization, accuracy
    sess = tf.Session()
    if args.dataset == 'CIFAR':
        model = CIFARModel(model_paths[args.dataset], tiny=False, mode='eval', sess=sess)
    
    # TODO: Write CelebA model class
    
    n_data = 0
    n_correct_orig = 0
    n_correct = 0
    total = 0
    for images, recons, labels in dset:
        total += 1
        
        n_correct_orig += sess.run(model.num_correct, feed_dict={model.x_input: images, model.y_input: labels})
        n_correct += sess.run(model.num_correct, feed_dict={model.x_input: recons, model.y_input: labels})
        n_data += len(images)


    acc_orig = float(n_correct_orig) / n_data
    acc = float(n_correct) / n_data
    print('Original acc: {}'.format(acc_orig))
    print('Accuracy: {}'.format(acc))
    
    return acc


def ComputeMSE(reconstructions, images):
    recons = np.reshape(reconstructions, (reconstructions.shape[0], -1))
    img = np.reshape(images, (images.shape[0], -1))
    mse = ((recons - img)**2).mean(axis=1)
    mse_avg = np.mean(mse)
    mse_std = np.std(mse)
    return (mse_avg, mse_std)


def ComputeInception(images):
    images = ((images + 1) / 2.0)*255.0
    images = images.astype(np.uint8)
    IS = inception.get_inception_score(images)
    return IS


def ComputeFID(reconstructions, images):
    reconstructions = ((reconstructions + 1) / 2.0)*255.0
    reconstructions = reconstructions.astype(np.uint8)
    images = ((images + 1) / 2.0)*255.0
    images = images.astype(np.uint8)
    
    images = np.transpose(images, (0, 3, 1, 2))
    reconstructions = np.transpose(reconstructions, (0, 3, 1, 2))
    
    FID = fid.get_fid(images, reconstructions)
    return FID
