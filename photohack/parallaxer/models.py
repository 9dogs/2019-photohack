"""Models."""
import os
import tarfile
import urllib.request
from collections import namedtuple
from io import BytesIO

import numpy as np
import scipy.misc
import tensorflow as tf
from PIL import Image

from photohack.parallaxer.consts import MONODEPTH_INPUT_SIZE
from photohack.parallaxer.utils import post_process_disparity

monodepth_parameters = namedtuple(
    'parameters', 'encoder height width batch_size num_threads '
                  'num_epochs do_stereo wrap_mode use_deconv '
                  'alpha_image_loss disp_gradient_loss_weight '
                  'lr_loss_weight full_summary')


class DeepLabModel:
    """DeepLab model and inference."""

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


class MonodepthModel:
    """MonoDepth model and inference.

    For details see: https://github.com/mrharicot/monodepth"""

    INPUT_TENSOR_NAME = 'input_image'
    OUTPUT_TENSOR_NAME = 'model/output_depth:0'
    INPUT_SIZE = MONODEPTH_INPUT_SIZE
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

    def _preprocess_image(self, image):
        """Resize the image to fit INPUT_SIZE and concat to its flipped
        version.

        :param PIL.Image image: image
        :return:
        """
        (self.original_height_, self.original_width_,
         self.num_channels_) = np.asarray(image).shape
        input_image = scipy.misc.imresize(
            image, [self.INPUT_SIZE[0], self.INPUT_SIZE[1]], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)
        return input_images

    def run(self, image):
        """Runs inference on a single image.

        :param PIL.Image image: raw input image
        :return: RGB image resized from original input image and Segmentation
            map of `resized_image`
        :rtype: tuple
        """
        images = self._preprocess_image(image)
        depth = self.sess.run(
            self.OUTPUT_TENSOR_NAME, feed_dict={'input_image:0': images})
        disp_pp = post_process_disparity(depth.squeeze()).astype(np.float32)
        disp_to_img = scipy.misc.imresize(
            disp_pp.squeeze(), [self.original_height_, self.original_width_])
        return disp_pp, disp_to_img
