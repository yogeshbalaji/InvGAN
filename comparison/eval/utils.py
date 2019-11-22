import os

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def data_loader(images, recons, labels, batch_size):
    for i in range(0, len(images), batch_size):
        image_batch = images[i:i + batch_size]
        recon_batch = recons[i:i + batch_size]
        label_batch = labels[i:i + batch_size]
        
        yield image_batch, recon_batch, label_batch

