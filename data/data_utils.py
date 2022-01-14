import torch.utils.data
import numpy as np
import random
import copy
from collections import defaultdict
from data.data_distribution_config import data_utils_config
from torchvision import datasets, transforms
from configs import args_parser

# -------------------------------------------------------------------------------------------------------
# DATASETS
# -------------------------------------------------------------------------------------------------------
# DATA_PATH = '~/CODES/Dataset'
DATA_PATH = 'E:/Data/'
np.random.seed(1)
args = args_parser()


class ImageHelper:
    def __init__(self):
        if args.dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.train_dataset = datasets.CIFAR10(DATA_PATH, train=True, download=False, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(DATA_PATH, train=False, download=False, transform=transform_test)
        elif args.dataset == 'mnist':
            transform_train = transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize((0.06078,), (0.1957,))
            ])
            transform_test = transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize((0.06078,), (0.1957,))
            ])
            self.train_dataset = datasets.MNIST(DATA_PATH, train=True, download=False, transform=transform_train)
            self.test_dataset = datasets.MNIST(DATA_PATH, train=False, download=False, transform=transform_test)
        elif args.dataset == 'fmnist':
            transform_train = transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            transform_test = transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.FashionMNIST(DATA_PATH, train=True, download=False, transform=transform_train)
            self.test_dataset = datasets.FashionMNIST(DATA_PATH, train=False, download=False, transform=transform_test)
        if args.trigger_type == 'swap':
            classes = {}
            for ind, x in enumerate(self.train_dataset):
                _, label = x
                if label in classes:
                    classes[label].append(ind)
                else:
                    classes[label] = [ind]
            self.swap_range_no_id = classes[args.swap_class_raw]

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())
        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def get_train(self, indices):
        # This method is used along with Dirichlet distribution
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=args.batch_size,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def get_test(self):
        # if args.defense == 'no' and args.trigger_type == 'swap':
        #     range_no_id = list(range(10000))
        #     classes = {}
        #     for ind, x in enumerate(self.test_dataset):
        #         _, label = x
        #         if label in classes:
        #             classes[label].append(ind)
        #         else:
        #             classes[label] = [ind]
        #     remove_no_id = classes[args.swap_class_raw]
        #     for image in remove_no_id:
        #         if image in range_no_id:
        #             range_no_id.remove(image)
        #     return torch.utils.data.DataLoader(self.test_dataset, batch_size=args.batch_size,
        #                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(range_no_id))
        # else:
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=True)

    def poison_train_dataset(self):
        range_no_id = list(range(50000))
        if args.trigger_type == 'semantic':
            semantic_type = args.semantic_type
            for image in data_utils_config[semantic_type + '_train'] + data_utils_config[semantic_type + '_test']:
                if image in range_no_id:
                    range_no_id.remove(image)
        elif args.trigger_type == 'swap':
            for image in self.swap_range_no_id:
                if image in range_no_id:
                    range_no_id.remove(image)
        mal_data_loaders = list()
        for _ in range(int(args.n_clients * args.mal_ratio)):
            indices = list()
            for _ in range(0, args.batches_of_mal_data):
                sampling = random.sample(range_no_id, args.batch_size)
                indices.extend(sampling)
            mal_data_loaders.append(torch.utils.data.DataLoader(
                self.train_dataset, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)))
        return mal_data_loaders

    def poison_test_dataset(self):
        if args.trigger_type == 'pixel':
            # delete the test data with target label
            test_classes = {}
            for ind, x in enumerate(self.test_dataset):
                _, label = x
                if label in test_classes:
                    test_classes[label].append(ind)
                else:
                    test_classes[label] = [ind]

            range_no_id = list(range(0, len(self.test_dataset)))
            for image_ind in test_classes[args.poison_label_swap]:
                if image_ind in range_no_id:
                    range_no_id.remove(image_ind)
            return torch.utils.data.DataLoader(self.test_dataset,
                                               batch_size=args.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   range_no_id))
        elif args.trigger_type == 'semantic':
            return torch.utils.data.DataLoader(self.train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   range(1000)))
        elif args.trigger_type == 'swap':
            test_classes = {}
            for ind, x in enumerate(self.test_dataset):
                _, label = x
                if label in test_classes:
                    test_classes[label].append(ind)
                else:
                    test_classes[label] = [ind]
            range_no_id = test_classes[args.swap_class_raw]
            return torch.utils.data.DataLoader(self.test_dataset,
                                               batch_size=args.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   range_no_id))

    def load_data(self):
        # sample indices for participants using Dirichlet distribution
        indices_per_participant = self.sample_dirichlet_train_data(
            args.n_clients, alpha=args.dirichlet_alpha)
        # construct local dataloaders for each client
        train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                         indices_per_participant.items()]
        # construct test dataloader
        test_loader = self.get_test()
        # local data size of each client
        stats = indices_per_participant.values()
        stats = [len(indices) for indices in stats]
        test_data_poison = self.poison_test_dataset()

        return train_loaders, stats, test_loader, test_data_poison

    def get_poison_batch(self, batch, adversarial_index=-1, evaluation=False, certify=False):
        images, targets = batch
        poison_count = 0
        new_images = images
        new_targets = targets
        for index in range(0, len(images)):
            if evaluation:  # poison all data when testing
                if not certify:
                    new_targets[index] = args.poison_label_swap
                if args.trigger_type == 'pixel':
                    new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)
                elif args.trigger_type == 'semantic':
                    new_images[index] = self.train_dataset[
                        random.choice(data_utils_config[args.semantic_type + '_test'])][0]
                elif args.trigger_type == 'swap':
                    new_targets[index] = args.poison_label_swap
                poison_count += 1
            else:  # poison part of data when training
                if index < args.num_mal_samples:
                    new_targets[index] = args.poison_label_swap
                    if args.trigger_type == 'pixel':
                        new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)
                    elif args.trigger_type == 'semantic':
                        new_images[index] = self.train_dataset[
                            random.choice(data_utils_config[args.semantic_type + '_train'])][0]
                        new_images[index].add_(torch.FloatTensor(new_images[index].shape).normal_(0, 0.05))
                    elif args.trigger_type == 'swap':
                        new_images[index] = self.train_dataset[random.choice(self.swap_range_no_id)][0]
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]
        new_images = new_images
        new_targets = new_targets
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images, new_targets, poison_count

    def add_pixel_pattern(self, ori_image, adversarial_index):
        image = copy.deepcopy(ori_image)
        poison_patterns = []
        if adversarial_index == -1:
            for i in range(0, data_utils_config['trigger_num']):
                poison_patterns = poison_patterns + data_utils_config[str(i) + '_poison_pattern']
        else:
            poison_patterns = data_utils_config[str(adversarial_index) + '_poison_pattern']
        if args.dataset == 'cifar10':
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1
        elif args.dataset == 'mnist':
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
        elif args.dataset == 'fmnist':
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
        return image


if __name__ == '__main__':
    from torchvision.utils import save_image
    import matplotlib.pyplot as plt
    helper = ImageHelper()
    # local_dataloaders, local_data_sizes, test_dataloader, test_poison_dataloader = helper.load_data()
    # for batch in test_poison_dataloader:
    #     X, y, poison_count = helper.get_poison_batch(batch, evaluation=True, certify=True)
    #     # for i in range(poison_count):
    #     #     print(y[i])
    #     #     plt.imshow(X[i].reshape(28, 28))
    #     #     plt.show()
    #     for i in range(poison_count):
    #         print(y[i])
    #         save_image(X[i], str(i) + 'cifar10.png')
    #     break
    semantic_type = args.semantic_type
    for image in data_utils_config[semantic_type + '_train'] + data_utils_config[semantic_type + '_test']:
        save_image(helper.train_dataset[image][0], str(image) + '.png')
