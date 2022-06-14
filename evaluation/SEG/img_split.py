import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def read_filename(filename):

    lines = list()
    f = open(filename, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        line = line.replace('\n', '')
        lines.append(line)
    f.close()
    return lines


def make_pair(src, dst, filename, split, img_size, overlay):

    src_image_path = os.path.join(src, 'train', 'images', filename)
    src_label_path = os.path.join(src, 'train', 'gt', filename)

    assert os.path.exists(src_image_path), src_image_path
    assert os.path.exists(src_label_path), src_label_path

    return dict(
        src_image_path=src_image_path,
        src_label_path=src_label_path,
        split=split,
        dst=dst,
        img_size=img_size,
        overlay=overlay)


def make_mmseg_style(src_image_path, src_label_path, split, dst, img_size,
                     overlay):

    scene_name = os.path.splitext(os.path.basename(src_image_path))[0]
    image_path = os.path.join(dst, split, 'images', scene_name)
    gt_path = os.path.join(dst, split, 'gt', scene_name)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = cv2.imread(src_label_path, 0)

    idx = 0
    height, width, _ = image.shape
    for h in range(0, height, img_size - overlay):
        for w in range(0, width, img_size - overlay):
            if h + img_size > height:
                h = height - img_size
            if w + img_size > width:
                w = width - img_size

            patch_image = image[h:h + img_size, w:w + img_size, :]
            patch_label = label[h:h + img_size, w:w + img_size]
            patch_label = np.where(patch_label == 255, 1, 0)

            patch_image = np.array(patch_image, dtype=np.uint8)
            patch_label = np.array(patch_label, dtype=np.uint8)

            patch_image = Image.fromarray(patch_image).convert('RGB')
            patch_label = Image.fromarray(patch_label).convert('P')
            patch_label.putpalette(
                np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8))

            patch_image.save(os.path.join(image_path, f'{idx}.png'))
            patch_label.save(os.path.join(gt_path, f'{idx}.png'))

            idx += 1


def map_function(data):
    make_mmseg_style(**data)


def main(args):

    train_lists = read_filename(args.train_file)
    valid_lists = read_filename(args.valid_file)

    pairs = list()
    for t in train_lists:
        pairs.append(
            make_pair(args.src, args.dst, t, 'train', args.img_size,
                      args.overlay))

    for v in valid_lists:
        pairs.append(
            make_pair(args.src, args.dst, v, 'valid', args.img_size,
                      args.overlay))

    pool = Pool(args.nproc)
    for _ in tqdm(pool.imap_unordered(map_function, pairs), total=len(pairs)):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=3, type=int)
    parser.add_argument('--img-size', default=1024, type=int)
    parser.add_argument('--overlay', default=128, type=int)
    parser.add_argument('--train-file', default='train.txt', type=str)
    parser.add_argument('--valid-file', default='valid.txt', type=str)
    parser.add_argument('--src', default='inria/AerialImageDataset', type=str)
    parser.add_argument('--dst', default='dataset', type=str)
    args = parser.parse_args()
    main(args)
