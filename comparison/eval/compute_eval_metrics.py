# Code for computing four evaluation metrics: MSE error, classification accuracy, inception score and FID 

import os
import argparse
import utils
import pickle
from metrics import ComputeClassificationAccuracy, ComputeMSE, ComputeInception, ComputeFID
import numpy as np

def main():
    
    # reading arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='name of the dataset| CIFAR, CelebA')
    parser.add_argument('--pkl_file', required=True, help='path to pickle file containing reconstructions')
    parser.add_argument('--log_path', default='logs', help='path to output log file')
    args = parser.parse_args()

    # Creating log file
    utils.mkdirp(args.log_path)
    
    # Reading data and reconstructions
    with open(args.pkl_file, 'rb') as f:
        data_dict = pickle.load(f)
    
    real_images = data_dict['real_images']
    labels = data_dict['labels']
    reconstructions = data_dict['reconstructions']
    
    # checking data format
    if real_images.shape[1] == 3:
        real_images = np.transpose(real_images, (0, 2, 3, 1))
        real_images = real_images*2 - 1.0
    if reconstructions.shape[1] == 3:
        reconstructions = np.transpose(reconstructions, (0, 2, 3, 1))
        reconstructions = reconstructions*2 - 1.0
    
    class_acc = ComputeClassificationAccuracy(real_images, reconstructions, labels, args)
    mse = ComputeMSE(reconstructions, real_images)
    inception_score = ComputeInception(reconstructions)
    fid = ComputeFID(reconstructions, real_images)
    
    log_file = open('{}/results_{}.txt'.format(args.log_path, args.dataset), 'w')
    
    msg1 = 'Classification accuracy: {}'.format(class_acc)
    msg2 = 'Mean Squared Error: {}, {}'.format(mse[0], mse[1])
    msg3 = 'Inception score: {},  {}'.format(inception_score[0], inception_score[1])
    msg4 = 'FIS score: {}'.format(fid)
    
    print(msg1)
    print(msg2)
    print(msg3)
    print(msg4)
    
    log_file.write(msg1 + '\n')
    log_file.write(msg2 + '\n')
    log_file.write(msg3 + '\n')
    log_file.write(msg4 + '\n')
    
    log_file.close()

if __name__ == '__main__':
    main()
