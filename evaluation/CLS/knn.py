import argparse

import torch
from backbone import build  # noqa: F401
from dataset import stl10  # noqa: F401
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_backbone
from mmcv import Config
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):

    gap = torch.nn.AdaptiveAvgPool2d(1)
    flatten = torch.nn.Flatten()

    if use_cuda:
        model = model.cuda()

    features = list()
    labels = list()
    for data in tqdm(data_loader):

        img = data['img']
        label = data['gt_label']

        if use_cuda:
            img = img.cuda()
            label = label.cuda()

        feature = model(img)[-1]
        feature = flatten(gap(feature))

        features.append(feature[0])
        labels.append(label[0])

        if len(features) == 1000:
            break

    features = torch.stack(features)
    labels = torch.stack(labels)

    return features, labels


@torch.no_grad()
def knn_classifier(train_features,
                   train_labels,
                   test_features,
                   test_labels,
                   k,
                   T,
                   num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx:min(
            (
                idx +  # noqa: W504
                imgs_per_chunk),
            num_test_images), :]
        targets = test_labels[idx:min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(
            5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    knn_pipeline = cfg.data.val.pipeline
    knn_pipeline[-1] = cfg.data.train.pipeline[-1]
    cfg.data.train.pipeline = knn_pipeline
    cfg.data.val.pipeline = knn_pipeline

    train_dataset = build_dataset(cfg.data.train)
    valid_dataset = build_dataset(cfg.data.val)

    train_dataloader = build_dataloader(
        train_dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )

    valid_dataloader = build_dataloader(
        valid_dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )

    model = build_backbone(cfg.model.backbone)
    model = model.eval()

    train_features, train_labels = extract_features(
        model, train_dataloader, use_cuda=True)
    valid_features, valid_labels = extract_features(
        model, valid_dataloader, use_cuda=True)

    top1, top5 = knn_classifier(
        train_features,
        train_labels,
        valid_features,
        valid_labels,
        k=20,
        T=0.1,
        num_classes=10)


if __name__ == '__main__':
    main()
