import tensorflow as tf

from semsegmentation.datasets.cityscapes import CITYSCAPES_LABELS, create_cityscapes_label_colormap


class CitiscapesColormaps(tf.test.TestCase):

    def testColormaps(self):
        self.assertEqual(len(CITYSCAPES_LABELS), create_cityscapes_label_colormap().shape[0])
        self.assertEqual(3, create_cityscapes_label_colormap().shape[1])


if __name__ == '__main__':
    tf.test.main()
