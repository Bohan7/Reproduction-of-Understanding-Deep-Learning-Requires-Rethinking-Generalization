import torch
import torch.nn as nn
import time
import random
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt


class cifar_10_train(Dataset):
    """ Returns the train set of CIFAR-10, under a chosen modification """
    def __init__(self, type='true labels', corrupt_p=None, data_augmentation=False):
        np.random.seed(42)

        self.type = type
        if data_augmentation == False:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.CenterCrop((28, 28)),
                                            transforms.Lambda(lambda x: x / 255.0),
                                            transforms.Lambda(per_image_standardization)])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomCrop((28, 28)),
                                            transforms.Lambda(lambda x: x / 255.0),
                                            transforms.Lambda(per_image_standardization)])
        self.train_set = torchvision.datasets.CIFAR10(root='datasets', train=True, transform=transform, download=True)
        length = len(self.train_set)

        if type == 'true labels':
            pass
        elif type == 'partially corrupted labels':
            self.images = [self.train_set[i][0] for i in range(length)]
            true_targets = [self.train_set[i][1] for i in range(length)]
            random_targets = list(np.random.choice(10, length))
            corrupt_probability = np.random.choice(2, length, p=[corrupt_p, 1 - corrupt_p])
            self.targets = corrupt_probability * true_targets + (1 - corrupt_probability) * random_targets
        elif type == 'random label':
            self.images = [self.train_set[i][0] for i in range(length)]
            self.targets = list(np.random.choice(10, length))
        elif type == 'shuffle pixels':
            random_pixel_permutation = np.random.choice(784, 784, replace=False)
            self.images = [self.train_set[i][0].view(3, 28 * 28)[:, random_pixel_permutation].view(3, 28, 28) for i in
                           range(length)]
            self.targets = [self.train_set[i][1] for i in range(length)]
        elif type == 'random pixels':
            self.images = [
                self.train_set[i][0].view(3, 28 * 28)[:, np.random.choice(784, 784, replace=False)].view(3, 28, 28) for
                i in range(length)]
            self.targets = [self.train_set[i][1] for i in range(length)]
        elif type == 'Gaussian':
            self.images = torch.tensor(np.random.normal(loc=0, scale=1, size=(50000, 3, 28, 28))).float()
            self.targets = [self.train_set[i][1] for i in range(length)]

    def __getitem__(self, index):
        if self.type == 'true labels':
            return self.train_set[index]
        else:
            return self.images[index], self.targets[index]

    def __len__(self):
        return len(self.train_set)

def per_image_standardization(image):
    """
    This function creates a custom per image standardization
    transform which is used for data augmentation.
    params:
        - image (torch Tensor): Image Tensor that needs to be standardized.
    
    returns:
        - image (torch Tensor): Image Tensor post standardization.
    """
    # get original data type
    orig_dtype = image.dtype

    # compute image mean
    image_mean = torch.mean(image, dim=(-1, -2, -3))

    # compute image standard deviation
    stddev = torch.std(image, axis=(-1, -2, -3))

    # compute number of pixels
    num_pixels = torch.tensor(torch.numel(image), dtype=torch.float32)

    # compute minimum standard deviation
    min_stddev = torch.rsqrt(num_pixels)

    # compute adjusted standard deviation
    adjusted_stddev = torch.max(stddev, min_stddev)

    # normalize image
    image -= image_mean
    image = torch.div(image, adjusted_stddev)

    # make sure that the image output dtype is the same as that of input dtype
    assert image.dtype == orig_dtype

    return image

def return_loaders(root, batch_size, test_bs=None, db='mnist', degrees=None, rsz=None,
                   augmentation=True, scale=None, shear=None, num_workers=2,**kwargs):
    """
    Return the loader for the data. This is used both for training and for
    validation.
    :param root: (str) Path of the root for finding the appropriate pkl/npy.
    :param batch_size: (int) The batch size for training.
    :param test_bs:  (int) The batch size for testing.
    :param db:  (str) The name of the database.
    :param kwargs:
    :return: The train and validation time loaders.
    """
    def _loader(db, batch_size, train=True, **kwargs):
        return DataLoader(db, shuffle=train, batch_size=batch_size,
                          num_workers=num_workers)

    if test_bs is None:
        test_bs = batch_size
        
    # # by default (i.e. with no normalization)
    # # cifar10 in the range [0, 1].
    if 'cifar' in db:
        if augmentation:
            trans = [transforms.RandomCrop(28, padding=4),
                     transforms.ColorJitter(.25,.25,.25,.25)]
            if degrees is not None:
                trans.append(transforms.RandomAffine(degrees, scale=scale, shear=shear))
            if rsz is not None:
                trans.append(transforms.Resize(rsz))
        else:
            trans = [transforms.CenterCrop((28, 28))]
            if rsz is not None:
                trans.append(transforms.Resize(rsz))

        trans += [
            transforms.ToTensor(),
            transforms.Lambda(per_image_standardization)
        ]        
        
        transform_train = transforms.Compose(trans)
        print('Transformation of train dataset:')
        print(transform_train)
        
        test_trans = [transforms.CenterCrop((28, 28))] + trans[-2:]
        transform_test = transforms.Compose(test_trans)
        print('Transformation of test dataset:')
        print(transform_test)
        
        if db == 'cifar100':
            trainset = datasets.CIFAR100(root=root, train=True,
                                         download=True, transform=transform_train)
            testset = datasets.CIFAR100(root=root, train=False,
                                        download=True, transform=transform_test)
        else:
            trainset = datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=transform_train)
            testset = datasets.CIFAR10(root=root, train=False,
                                       download=True, transform=transform_test)

    else:
        raise NotImplementedError('db: {}'.format(db))


    train_loader = _loader(trainset, batch_size, train=True, **kwargs)
    val_loader = _loader(testset, test_bs, train=False, **kwargs)
    
    return train_loader, val_loader


def train(model, train_dataloader, lr, max_epoch=150, save_dir=None, gamma=0.95):
    """ train the model without evaluation on test set """
    # use SGD momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # lr decay: decay by 0.95 for each epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # use cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    # use GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    model, criterion = model.to(device), criterion.to(device)

    n_batch = len(train_dataloader)

    # a list to record the state of training
    training_stats = []

    print('Training start for {}!'.format(save_dir))
    for e in range(max_epoch):

        # train model
        model.train()

        train_loss, train_acc = 0, 0

        for b, (images, targets) in enumerate(train_dataloader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)

            train_loss += loss
            train_acc += (outputs.max(1)[1] == targets).float().mean().item()

            loss.backward()
            optimizer.step()
            print("\rEpoch: {:d} batch: {:d} / {} loss: {:.4f} | {:.2%}".format(e + 1, b + 1, n_batch, loss,
                                                                                (b + 1) * 1.0 / n_batch), end='',
                  flush=True)
        scheduler.step()  # lr decays after each epoch

        train_loss, train_acc = train_loss / len(train_dataloader), train_acc / len(train_dataloader)

        training_stats.append(
            {
                'epoch': e + 1,
                'train_loss': train_loss,
                'train_acc': train_acc
            }
        )

        # save models
        torch.save(model, 'saved_models/{}.pkl'.format(save_dir))

        # save states of training
        np.save('saved_models/{}-train_stats.npy'.format(save_dir), training_stats)

        print('\n----------------------- Epoch {} -----------------------'.format(e + 1))
        print('Train_loss: {:.4f} | Train_acc: {:.4f}'.format(train_loss, train_acc))

    print('Training complete for {}!\n'.format(save_dir))

def predict_batch(model, train_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_batch = len(train_dataloader)
    print('predict acc for a batch!')
    
    train_acc = 0
    model.eval()
    for b, (images, targets) in enumerate(train_dataloader):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        train_acc += (outputs.max(1)[1] == targets).float().mean().item()
           
    
    train_acc = train_acc / len(train_dataloader)
    print('train acc: {}'.format(train_acc))
    return train_acc


def train_val(model, train_dataloader, test_loader, lr, max_epoch=150, save_dir=None, early_stop=False, patience=5,
              min_delta=1e-4, weight_decay=0, record_batch=100, gamma=0.95, predict_batch_bool=False):
    """ train the model with evaluation on test set """
    # use SGD momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # lr decay: decay by 0.95 for each epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # use cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    # use GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    model, criterion = model.to(device), criterion.to(device)

    n_batch = len(train_dataloader)

    # a list to record the state of training
    training_stats = []
    training_stats_batch = []
    train_time = 0

    if early_stop == True: opt_loss = 1e5

    print('Training start for {}!'.format(save_dir))
    for e in range(max_epoch):

        # train model
        model.train()

        train_loss, train_acc = 0, 0

        tic = time.time()
        

        for b, (images, targets) in enumerate(train_dataloader):
            model.train()
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)

            train_loss += loss
            train_acc += (outputs.max(1)[1] == targets).float().mean().item()

            loss.backward()
            optimizer.step()
            print("\rEpoch: {:d} batch: {:d} / {} loss: {:.4f} | {:.2%}".format(e + 1, b + 1, n_batch, loss,
                                                                                (b + 1) * 1.0 / n_batch), end='',
                  flush=True)
            
            if predict_batch_bool == True:
                if b % record_batch == 0:
                    train_acc_batch = predict_batch(model, train_dataloader)
                    training_stats_batch.append(
                        {
                            'epoch': e + 1,
                            'batch': b + 1,
                            'train_loss': train_loss,
                            'train_acc': train_acc_batch
                        }
                    )

        scheduler.step()  # lr decays after each epoch

        toc = time.time()
        train_time += (toc - tic)

        # evaluate model

        model.eval()

        eval_acc, eval_loss = 0, 0

        for b, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, targets)

            eval_loss += loss
            eval_acc += (outputs.max(1)[1] == targets).float().mean().item()

        train_loss, train_acc = train_loss / len(train_dataloader), train_acc / len(train_dataloader)
        test_loss, test_acc = eval_loss / len(test_loader), eval_acc / len(test_loader)

        training_stats.append(
            {
                'epoch': e + 1,
                'train_time': train_time,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }
        )

        # save models
        torch.save(model, 'saved_models/{}.pkl'.format(save_dir))

        # save states of training
        np.save('saved_models/{}-train_stats.npy'.format(save_dir), training_stats)
        np.save('saved_models/{}-train_stats_batch.npy'.format(save_dir), training_stats_batch)

        print('\n----------------------- Epoch {} -----------------------'.format(e + 1))
        print('Train_loss: {:.4f} | Train_acc: {:.4f} | Test_loss: {:.4f} | Test_acc: {:.4f}' \
              .format(train_loss, train_acc, test_loss, test_acc))

        # eartly stop monitors train loss, terminate training process if no improvement made for 5 epochs
        if early_stop == True:
            if train_loss > opt_loss * (1 - min_delta) or float(train_acc) == 1.0:
                patience -= 1
                if patience == 0:
                    print('Early stop, training converges after {} epochs, training time spent: {:.4f}s'.format(e + 1,
                                                                                                                train_time))
                    break
            else:
                patience = 5
                opt_loss = train_loss if train_loss <= opt_loss else opt_loss

    print('Training complete for {}!\n'.format(save_dir))

def random_seed(seed=42):
    ''' set random seed '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # set random seed of cuda
    torch.cuda.manual_seed_all(seed) # set random seed of multi-GPU

def get_stats(train_stats, train_stats_batch, max_epoch=51):
    '''A function to get training accuracy and testing accuracy from the training states'''
    train_acc = []
    train_acc_batch = []
    test_acc = []   
    for index in range(max_epoch):
        train_acc.append(train_stats[index]['train_acc'])
        test_acc.append(train_stats[index]['test_acc'])
    
    for index in range(max_epoch*4):
        train_acc_batch.append(train_stats_batch[index]['train_acc'])
    
    return train_acc, train_acc_batch, test_acc

def culmulative_test_acc(test_acc):
    '''A function to compute culmulative test accuracy'''
    result = np.zeros(len(test_acc))
    result[0] = test_acc[0]
    
    for i in np.arange(len(test_acc))[1:]:
        if test_acc[i] > result[i-1]:
            result[i] = test_acc[i]
        else:
            result[i] = result[i-1]
    
    return result
