from typing import Tuple

import cv2
import numpy as np

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}


def stretch_image(
    image: np.ndarray,
    new_max=255.0,
    min_percentile=0,
    max_percentile=100,
    clipped_min_val=10,
) -> np.ndarray:
    """특정 범위로 normalization을 진행합니다.

    @param image: 이미지
    @param new_max: 범위 최댓값
    @param min_percent: 범위 최솟값 %
    @param max_percent: 범위 최댓값 %
    @param dtype: 변환할 이미지 데이터 타입
    @return: 변환된 이미지
    """

    def _scale_range(min_val: int,
                     max_val: int,
                     to_range=255.0) -> Tuple[int, int]:
        """최소 범위에 해당하는 최솟값과 최댓값을 반환합니다.

        @param min_val: 범위를 축소시킬 값 중 최솟값
        @param max_val: 범위를 축소시킬 값 중 최댓값
        @param min_range: 축소시킬 범위
        @return: 최솟값과 최댓값
        """
        intensity_gap = max_val - min_val
        if intensity_gap < to_range:
            margin = (to_range - intensity_gap) / 2
            min_val -= margin
            max_val += margin

            if min_val < 0:
                max_val -= min_val
                min_val = 0
            if max_val > 2**16:
                min_val -= 2**16 - max_val
                max_val = 2**16 - 1
        return min_val, max_val

    for idx in range(image.shape[2]):
        band = image[:, :, idx]
        filtered_band = band[band > clipped_min_val]

        if filtered_band.any():
            min_val = np.percentile(filtered_band, min_percentile)
            max_val = np.percentile(filtered_band, max_percentile)
        else:
            min_val, max_val = 0, 255

        min_val, max_val = _scale_range(min_val, max_val)

        cvt_range = max_val - min_val
        band = (band - min_val) / cvt_range * new_max
        band = np.clip(band, 0, new_max)
        image[:, :, idx] = band

    return image


def imflip(img, direction='horizontal'):

    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             interpolation='bilinear',
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=cv2_interp_codes[interpolation],
        borderValue=border_value)
    return rotated
