import tensorflow as tf

from semsegmentation.datasets.cityscapes import Cityscapes


class CityscapesTest(tf.test.TestCase):

    def setUp(self):
        self.dataset = Cityscapes()

    def testColormaps(self):
        self.assertEqual(len(self.dataset.labels), self.dataset.colormap.shape[0])
        self.assertEqual(3, self.dataset.colormap.shape[1])


if __name__ == '__main__':
    tf.test.main()
