# Invert and Defend: Model-based Approximate Inversion of Generative Adversarial Network For Secure Inference

## Installation
1. Install pyenv from https://github.com/pyenv/pyenv#installation
2. Create a directory for InvGAN: 
```
mkdir <path-to-invgan>
cd <path-to-invgan>
```
2. `git clone git@github.com:invgan/rebuttal.git invgan`
3. `cd invgan` Now you should be in the project root directory. 
4. Run the setup script `./setup.sh`
5. Download datasets:
```
python download_dataset.py mnist
python download_dataset.py f-mnist
python download_dataset.py cifar-10
```
Download the pre-trained gans and classifiers from the following links into
the root directory:

Classifiers: https://drive.google.com/file/d/1wws8pvE2slWXGIcQt5QYlH4N78yNZ8lw/view?usp=sharing

MNIST: https://drive.google.com/file/d/1L_U8A_aKvmuLNo9hdz4IgEK7UQ8d5t10/view?usp=sharing

FMNIST: https://drive.google.com/file/d/1NkktHH7VDMk8A2pMz2XSDQwf2dFX59o0/view?usp=sharing

CIFAR10: https://drive.google.com/file/d/1NkktHH7VDMk8A2pMz2XSDQwf2dFX59o0/view?usp=sharing

Put the `*.tar.gz` files in the root directory and run:
```
tar -xzvf classifiers.tar.gz
tar -xzvf mnist.tar.gz
tar -xzvf fmnist.tar.gz
tar -xzvf cifar10.tar.gz
```
These should create an `output/` and `classifiers/model` directories.
# Experiments

## Prepare the cache
```
python train.py --cfg experiments/cfgs/gans_inv_notrain/mnist.yml --save_ds
python train.py --cfg experiments/cfgs/gans_inv_notrain/fmnist.yml --save_ds
python train.py --cfg experiments/cfgs/gans_inv_notrain/cifar_resnet.yml --save_ds
```
These scripts will normalize the datasets and save them in pickle files.

## White-box experiments (Table 2)
To reproduce the results of Table 2 run: 
### InvGAN
```
python whitebox.py --cfg experiments/cfgs/gans_inv_notrain/{mnist,fmnist,cifar_resnet}.yml \ 
       --results_dir whitebox \
       --attack_type {fgsm,cw,bpda,bpda-l2} \
       --model A \
       --load_classifier \
       --defense_type defense_gan \
       --rec_iters 1000 \
       --debug
```
The defense_type is intentionally `defense_gan`, the config file instructs the code
to use InvGAN. This will be fixed for the camera ready version. 

### DefenseGAN
```
python whitebox.py --cfg experiments/cfgs/gans/{mnist,fmnist,cifar10_hinge_resnet}.yml \ 
       --results_dir whitebox \
       --attack_type {fgsm,cw,bpda,bpda-l2} \
       --model A \
       --load_classifier \
       --defense_type defense_gan \
       --debug
```
## Detection experiments (Table 3)
#### InvGAN
After running the whitebox experiments, the ROC results can be produced:

```
python plots/plot_ROC.py \
       --pkl_file_adv results/gans_inv_notrain/{mnist,fmnist,cifar_resnet}/whitebox/0_model=A_orig_Iter=1000_RR=1_LR=0.0100_defense_ganattack={fgsm,cw,bpda,bpda-l2}_roc.pkl \
       --pkl_file_clean results/gans_inv_notrain/{mnist,fmnist,cifar_resnet,celeba_resnet}/whitebox/0_model=A_orig_Iter=1000_RR=1_LR=0.0100_defense_ganattack={fgsm,cw,bpda,bpda-l2}_roc_clean.pkl
```

#### DefenseGAN
```
python plots/plot_ROC.py \
--pkl_file_adv results/gans/{mnist,fmnist,cifar10_hinge_resnet}/whitebox/0_model=A_orig_Iter=200_RR=10_LR=10.0000_defense_ganattack={fgsm,cw,bpda,bpda-l2}_roc.pkl \
--pkl_file_clean results/gans/{mnist,fmnist,cifar10_hinge_resnet}/whitebox/0_model=A_orig_Iter=200_RR=10_LR=10.0000_defense_ganattack={fgsm,cw,bpda,bpda-l2}_roc_clean.pkl
```

## Train the Inverter

To train the inverter and the pre-trained models instead of the provided pre-trained models:
```
python train.py --cfg experiments/cfgs/gans_inv_notrain/{mnist,fmnist,cifar_resnet}.yml \
    --is_train
```

The code for attacks are in `whitebox.py` and `blackbox.py`.

The code for the inverter is in `model/gan_v2.py`.
