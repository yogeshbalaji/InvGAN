import numpy as np
import tensorflow as tf
from tflib.inception_score import get_inception_score


def compute_inception_score(gan_model, ckpt_path=None):
    sess = gan_model.sess
    gan_model.load_generator(ckpt_path=ckpt_path)
    sess.run(tf.local_variables_initializer())

    print("Evaluating...")
    num_images_to_eval = 50000
    eval_images = []
    num_batches = num_images_to_eval // gan_model.batch_size + 1
    print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
    np.random.seed(0)
    for _ in range(num_batches):
        images = sess.run(gan_model.fake_data, feed_dict={gan_model.is_training: False})
        eval_images.append(images)
    np.random.seed()
    eval_images = np.vstack(eval_images)
    eval_images = eval_images[:num_images_to_eval]
    eval_images = np.clip((eval_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    # Calc Inception score
    eval_images = list(eval_images)
    inception_score_mean, inception_score_std = get_inception_score(eval_images)
    print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))

def save_mse(reconstruction_dict, gan_model):
    ## reconstruction_dict is a dictionary of reconstructions and orig images
    ## should contain train, dev and test splits

    splits = ['train', 'dev', 'test']
    [all_recs, all_targets, orig_imgs] = reconstruction_dict['train']
    train_loss = np.mean((all_recs - orig_imgs)**2)
    
    [all_recs, all_targets, orig_imgs] = reconstruction_dict['dev']
    dev_loss = np.mean((all_recs - orig_imgs)**2)
    
    [all_recs, all_targets, orig_imgs] = reconstruction_dict['test']
    test_loss = np.mean((all_recs - orig_imgs)**2)
    
    ## Logging the error 
    logfile = open('output/reconstruction_results.txt', 'a+')
    config_ = 'rec-iter_{}_rec-rr_{}_rec-lr_{}\n'.format(gan_model.rec_iters, gan_model.rec_rr, gan_model.rec_lr)
    losses_ = 'Train loss: %f, Dev loss: %f, Test loss: %f\n\n\n'%(train_loss, dev_loss, test_loss)
    logfile.writelines(config_)
    logfile.writelines(losses_)
    logfile.close()