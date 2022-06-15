import random
from typing import List, Tuple

import albumentations
import cv2
import numpy as np

from ..builder import PIPELINES
from .colorspace import bgr2hsv, hsv2bgr
from .geometric import impad, impad_to_multiple
from .utils import imflip, imrotate

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


@PIPELINES.register_module()
class GaussianBlur(albumentations.augmentations.transforms.GaussianBlur):

    def __init__(self, blur_limit=(3, 7), sigma_limit=(0, 1), prob=0.5):

        assert isinstance(blur_limit, tuple)
        assert len(blur_limit) == 2
        assert isinstance(sigma_limit, tuple)
        assert len(sigma_limit) == 2
        assert 0 <= prob <= 1.0, (
            f'The prob should be in range [0, 1], got {prob} instead.')

        self.prob = prob
        self.blur_limit = blur_limit
        self.sigma_limit = sigma_limit

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        orig_dtype = img.dtype
        params = self.get_params()
        img = self.apply(img, **params)
        results['img'] = np.array(img, dtype=orig_dtype)
        return results


@PIPELINES.register_module()
class Solarization(object):

    def __init__(self, prob: float = 0.5, thr: int = 127):

        self.prob = prob
        self.thr = thr

    def __call__(self, results):

        if np.random.rand() > self.prob:
            return results

        img = results['img']
        img = np.where(img < self.thr, img, 255 - img)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, thr={self.thr})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):

    def __init__(self, mean: List[float], std: List[float], to_rgb=True):

        self.mean = np.array(mean)
        self.std = np.array(std)
        self.to_rgb = to_rgb

    def normalize(self, img, mean, std, to_rgb):

        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def __call__(self, results):

        img = results['img']
        img = img.copy().astype(np.float32)
        img = self.normalize(img, self.mean, self.std, self.to_rgb)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomChannelShift(object):

    def __init__(
        self,
        prob: float = 0.5,
    ):
        self.prob = prob

    def __call__(self, results):

        img = results['img']
        num_channels = img.shape[-1]
        shuffled_channel = list(range(num_channels))
        np.random.shuffle(shuffled_channel)

        img = np.array([img[:, :, i] for i in shuffled_channel])
        img = np.transpose(img, (1, 2, 0))
        results['img'] = img

        return results


@PIPELINES.register_module()
class PhotoMetricDistortion(object):

    def __init__(
            self,
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18,
            prob=0.5,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.prob = prob

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                beta=np.random.uniform(-self.brightness_delta,
                                       self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                alpha=np.random.uniform(self.contrast_lower,
                                        self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.randint(2):
            img = bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=np.random.uniform(self.saturation_lower,
                                        self.saturation_upper))
            img = hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.randint(2):
            img = bgr2hsv(img)
            img[:, :, 0] = (img[:, :, 0].astype(int) + np.random.randint(
                -self.hue_delta, self.hue_delta)) % 180  # noqa
            img = hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand() < self.prob:
            return results

        img = results['img']
        img = np.array(img, dtype=np.uint8)
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):

    def __init__(self, prob: float, direction: str = 'horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):

        flip = True if np.random.rand() < self.prob else False
        if flip:
            results['img'] = imflip(results['img'], direction=self.direction)
            return results

        else:
            return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob}, direction={self.direction})'


@PIPELINES.register_module()
class RandomRotate(object):

    def __init__(self,
                 prob: float,
                 degree: Tuple[int, int],
                 pad_val: int = 0,
                 center=None,
                 auto_bound=False):

        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0
            self.degree = (-degree, degree)
        else:
            self.degree = degree

        assert len(self.degree) == 2

        self.pad_val = pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            results['img'] = imrotate(
                results['img'],
                angle=degree,
                border_value=self.pad_val,
                center=self.center,
                auto_bound=self.auto_bound,
            )
            return results

        else:
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(prob={self.prob}, '
                     f'degree={self.degree}, '
                     f'pad_val={self.pad_val}, '
                     f'center={self.center}, '
                     f'auto_bound={self.auto_bound})')
        return repr_str


@PIPELINES.register_module()
class Pad(object):

    def __init__(
        self,
        size=None,
        size_divisor: int = None,
        pad_val: int = 0,
    ):

        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val

        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):

        if self.size is not None:
            padded_img = impad(img, shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val)
        return padded_img

    def __call__(self, results):

        img = results['img']
        img = self._pad_img(img)
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'size={self.size}, '
                    f'size_divisor={self.size_divisor}, '
                    f'pad_val={self.pad_val})')
        return repr_str


@PIPELINES.register_module()
class RandomCrop(object):

    def __init__(
        self,
        crop_size: Tuple[int, int],
    ):

        self.crop_size = crop_size

    def get_crop_bbox(self, img):
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        img = self.crop(img, crop_bbox)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'crop_size={self.crop_size})')
        return repr_str


@PIPELINES.register_module()
class Resize(object):

    def __init__(
        self,
        img_size: Tuple[int, int],
        ratio_range: Tuple[float, float],
        interpolation: str = 'bilinear',
    ):

        self.img_size = img_size
        self.ratio_range = ratio_range
        self.interpolation = interpolation

    def sample_range(self):
        return random.uniform(self.ratio_range[0], self.ratio_range[1])

    def resize_img(self, img, ratio):

        height, width = self.img_size

        img = cv2.resize(
            img,
            dsize=(
                int(width * ratio),
                int(height * ratio),
            ),
            interpolation=cv2_interp_codes[self.interpolation])

        return img

    def __call__(self, results):

        img = results['img']
        ratio = self.sample_range()
        img = self.resize_img(img, ratio)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'img_size={self.img_size}, '
                    f'ratio_range={self.ratio_range}, '
                    f'interpolation={self.interpolation})')
        return repr_str
