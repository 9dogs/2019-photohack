"""Models."""
import os
import tarfile
import urllib.request
from collections import namedtuple
from io import BytesIO
from tarfile import ReadError

import numpy as np
import scipy.misc
import tensorflow as tf
from photohack.parallaxer.consts import (
    DEEPLAB_INPUT_SIZE, MONODEPTH_INPUT_SIZE)
from photohack.parallaxer.utils import post_process_disparity
from PIL import Image

monodepth_parameters = namedtuple(
    'parameters', 'encoder height width batch_size num_threads '
                  'num_epochs do_stereo wrap_mode use_deconv '
                  'alpha_image_loss disp_gradient_loss_weight '
                  'lr_loss_weight full_summary')


class DeeplabModel:
    """DeepLab model and inference."""

    MODEL_URL = ('http://download.tensorflow.org/models/'
                 'deeplabv3_pascal_train_aug_2018_01_04.tar.gz')
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = DEEPLAB_INPUT_SIZE
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained DeepLab model.

        :param str tarball_path: path tarball with saved model
        """
        (self.orig_w_, self.orig_h_, self.orig_ch_num_) = (None, None, None)
        self.graph = tf.Graph()
        file_handle, graph_def = None, None
        # Extract frozen graph from tar archive
        try:
            tar_file = tarfile.open(tarball_path)
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
        except ReadError:
            # Assume ordinary file
            file_handle = open(tarball_path, 'rb')
        if file_handle:
            graph_def = tf.GraphDef.FromString(file_handle.read())
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def resize(self, image, size, method='nearest'):
        """Resize image to input dimensions.

        :param PIL.Image image: image
        :param tuple size: size (h, w)
        :param method: method ('nearest', 'lanczos', 'bilinear', 'bicubic'
            or 'cubic')
        :return: resized image
        :rtype: np.ndarray
        """
        h, w = size
        resized_image = scipy.misc.imresize(
            np.asarray(image), size=(h, w), interp=method)
        return resized_image

    def preprocess_image(self, image):
        """Scale image to fit input dimensions

        :param PIL.image image: input image
        :return: processed image
        :rtype: PIL.image
        """
        h, w = self.INPUT_SIZE
        processed_image = self.resize(image, (h, w), method='lanczos')
        return processed_image

    def postprocess_result(self, result):
        """Resize segmentation map to fit original image.

        :param result:
        :return:
        """
        h, w = self.orig_h_, self.orig_w_
        resized = self.resize(result, (h, w), method='nearest')
        return resized

    def run(self, image):
        """Runs inference on a single image.

        :param PIL.Image image: raw input image
        :return: RGB image resized from original input image and Segmentation
            map of `resized_image`
        :rtype: tuple
        """
        (self.orig_w_, self.orig_h_, self.orig_ch_num_) = np.asarray(
            image).shape
        processed_image = self.preprocess_image(image)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(processed_image)]})
        seg_map = batch_seg_map[0]
        result = self.postprocess_result(seg_map)
        return result

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


class MonodepthModel(DeeplabModel):
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
        super().__init__(tarball_path)

    def preprocess_image(self, image):
        """Resize the image to fit INPUT_SIZE and concat to its flipped
        version.

        :param PIL.Image image: image
        :return:
        """
        h, w = self.INPUT_SIZE
        processed_image = self.resize(image, (h, w), method='lanczos')
        processed_image = processed_image.astype(np.float32) / 255
        input_images = np.stack(
            (processed_image, np.fliplr(processed_image)), 0)
        return input_images

    def run(self, image):
        """Runs inference on a single image.

        :param PIL.Image image: raw input image
        :return: RGB image resized from original input image and Segmentation
            map of `resized_image`
        :rtype: tuple
        """
        (self.orig_w_, self.orig_h_, self.orig_ch_num_) = np.asarray(
            image).shape
        images = self.preprocess_image(image)
        depth = self.sess.run(
            self.OUTPUT_TENSOR_NAME, feed_dict={'input_image:0': images})
        disp_pp = post_process_disparity(depth.squeeze()).astype(np.float32)
        disp_to_img = scipy.misc.imresize(
            disp_pp.squeeze(), [self.orig_h_, self.orig_w_])
        return disp_to_img
