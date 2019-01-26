import os
import tarfile
import urllib.request
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    MODEL_URL = ('http://download.tensorflow.org/models/'
                 'deeplabv3_pascal_train_aug_2018_01_04.tar.gz')
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained DeepLab model.

        :param str tarball_path: path tarball with saved model
        """
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        :param PIL.Image image: raw input image
        :return: RGB image resized from original input image and Segmentation
            map of `resized_image`
        :rtype: tuple
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size,
                                                    Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def run_on_url(self, url):
        """Inferences DeepLab model and visualizes result.

        :param str url: URL of image
        :return:
        """
        try:
            f = urllib.request.urlopen(url)
            jpeg_str = f.read()
            original_im = Image.open(BytesIO(jpeg_str))
        except IOError:
            print('Cannot retrieve image. Please check url: ' + url)
            return
        print('running deeplab on image %s...' % url)
        resized_im, seg_map = self.run(original_im)
        return resized_im, seg_map
