import math
import os
from collections import defaultdict
from multiprocessing import Pool
import pickle

from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, utils
from torchvision import models, datasets, transforms

from .vision import VisionDataset


image_size = [224, 224]


def preprocess_images(args, image_path, cache=True, to_features=True, device=-1, num_workers=40):
    cache_path = image_path / 'cache' / 'image.pickle'
    if not (image_path / 'cache').is_dir():
        (image_path / 'cache').mkdir()

    if not to_features:
        return load_images(image_path, num_workers=num_workers)
    else:
        if cache_path.is_file():
            print("Loading Image Cache")
            with open(str(cache_path.resolve()), 'rb') as f:
                image_vectors = pickle.load(f)
        else:
            print("Loading Image Files and Building Image Cache")
            episode_paths = list(image_path.glob('*'))
            image_vectors = {}
            for e in tqdm(episode_paths, desc='Episode'):
                shot_paths = list(e.glob('*/*'))  # episode/scene/shot
                image_vectors_chunk = load_and_extract(shot_paths, args, device, num_workers)
                image_vectors = {**image_vectors, **image_vectors_chunk}
            image_vectors = merge_scene_features(image_vectors)

            print("Saving Image Cache")
            if cache:
                with open(str(cache_path), 'wb') as f:
                    pickle.dump(image_vectors, f)

        return image_vectors


def load_and_extract(shot_paths, args, device, num_workers=-1):
    images = load_images(shot_paths, num_workers=num_workers)
    image_vectors_chunk = extract_features(args, images, device=device, num_workers=num_workers, extractor_batch_size=args.extractor_batch_size)
    return image_vectors_chunk


def load_images(shot_paths, num_workers=1):
    images = {}
    if num_workers < 2:
        images = list(tqdm(map(load_image, shot_paths), total=len(shot_paths), desc='loading images'))
    else:
        with Pool(num_workers) as p:
            images = list(tqdm(p.imap(load_image, shot_paths), total=len(shot_paths), desc='loading images'))
    images = {k: v for k, v in images}

    return images


def load_image(shot_path):
    image_paths = shot_path.glob('*')
    vid = '_'.join(shot_path.parts[-3:])
    # vid = shot_path.parts[-1]
    res = {}
    image_paths = sorted(list(image_paths))
    for image_path in image_paths:
        name = image_path.parts[-1]
        image = Image.open(image_path)
        image = image.resize(image_size)
        res[name] = image
    return (vid, res)


class ObjectDataset(VisionDataset):
    def __init__(self, images, **kwargs):
        super(ObjectDataset, self).__init__('~/', **kwargs)

        self.images = list([(k, v) for k, v in images.items()])

    def __getitem__(self, idx):
        key, tensor = self.images[idx]
        if self.transform is not None:
            tensor = self.transform(tensor)

        return key, tensor

    def __len__(self):
        return len(self.images)


def extract_features(args, images, device=-1, num_workers=1, extractor_batch_size=384):
    delimiter = '/'
    # flatten images
    images = {"{}{}{}".format(vid, delimiter, name): image for vid, shots in images.items() for name, image in shots.items()}
    # print("Extracting feature for {} images".format(len(images)))

    dataset = ObjectDataset(images, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    dataloader = utils.data.DataLoader(dataset,
                                       batch_size=extractor_batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)

    model = get_model(device)
    model.eval()

    images = {}
    # print("Extracting Features")
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=math.ceil(len(dataset) / extractor_batch_size), desc='extracting features'):
            keys, tensor = data
            tensor = tensor.to(device)

            features = model(tensor).cpu()

            for key, feature in zip(keys, features):
                vid, name = key.split(delimiter)
                images.setdefault(vid, {}).__setitem__(name, feature)

    del model

    image_vectors = {}
    pooling = {
        'max': lambda x, dim: torch.max(x, dim=dim, keepdim=False)[0],
        'mean': lambda x, dim: torch.mean(x, dim=dim, keepdim=False),
    }[args.feature_pooling_method]
    '''
    if fixed_vector:
        for vid, dt in images.items():
            feature = list([v.to(device) for v in dt.values()])
            feature = torch.stack(feature, dim=0)



            feature = pooling(feature, -1)
            feature = pooling(feature, -1)  # pooling image size
            feature = pooling(feature, 0)  # pooling all images in shot (or scene)

            image_vectors[vid] = feature.cpu().numpy()
    else:
    '''
    for vid, dt in images.items():
        feature = list([v.to(device) for v in dt.values()])
        feature = torch.stack(feature, dim=0)
        feature = pooling(feature, -1)
        feature = pooling(feature, -1)  # pooling image size
        # feature = pooling(feature, 0)  # pooling all images in shot (or scene)
        image_vectors[vid] = feature.cpu().numpy()

    return image_vectors


def get_empty_image_vector(sample_image_size=[]):
    return torch.zeros(sample_image_size).cpu().numpy()


def get_model(device):
    # print("Using Resnet18")
    model = models.resnet18(pretrained=True)
    extractor = nn.Sequential(*list(model.children())[:-2])
    extractor.to(device)

    return extractor


def merge_scene_features(image_vectors):
    # mean pool
    keys = list(image_vectors.keys())
    scene = defaultdict(list)
    for key in keys:
        scene_key = '_'.join(key.split('_')[:-1] + ['0000'])
        scene[scene_key].append(torch.from_numpy(image_vectors[key]))
    for k, v in scene.items():
        image_vectors[k] = torch.cat(v, dim=0).numpy()

    return image_vectors
