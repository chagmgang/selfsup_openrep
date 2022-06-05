import cv2

from ..build import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:

    def __init__(
        self,
        to_float32=False,
        channel_order='bgr',
    ):

        self.to_float32 = to_float32
        self.channel_order = channel_order

    def __call__(self, results):

        filename = results['filename']
        img = cv2.imread(filename, cv2.IMREAD_COLOR)

        if self.channel_order == 'rgb':
            print(self.channel_order)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'channel_order={self.channel_order})')
        return repr_str
