"""Utilities."""
import logging
import math
import os
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from matplotlib import gridspec
from photohack.parallaxer.consts import MONODEPTH_INPUT_SIZE, MONODEPTH_MODEL


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    :return: A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    :param label: A 2D array with integer type, storing the segmentation label
    :return: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map
    :raises ValueError: If label is not of rank 2 or its value is larger than
        color map maximum entry
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]


def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view.

    :param PIL.Image image:
    :param seg_map:
    :return:
    """
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8),
        interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


#: DeepLab segmentation classes
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
#: PASCAL color map for classes
FULL_COLOR_MAP = label_to_color_image(
    np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1))


def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border',
                          name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):
            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]],
                            mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0, _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f, _width_f - 1 + 2 * _edge_size),
                         tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
                                   tf.linspace(0.0, _height_f - 1.0, _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed,
                tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch = tf.shape(input_images)[0]
        _height = tf.shape(input_images)[1]
        _width = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f = tf.cast(_width, tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def meta_graph_from_model(checkpoint_file, output_dir=None):
    """Convert MonoDepth model to MetaGraph.

    :param checkpoint_file:
    :param str output_dir: output directory name
    """

    class monodepth(object):
        """monodepth model"""

        def __init__(self, x, model):
            self.use_deconv = False
            self.output = self.build_model(x, model)

        def get_output(self):
            return self.output

        def scale_pyramid(self, img, num_scales):
            scaled_imgs = [img]
            s = tf.shape(img)  # img.get_shape().as_list()
            h = s[1]
            w = s[2]
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = h / ratio
                nw = w / ratio
                scaled_imgs.append(
                    tf.image.resize_area(img, [math.ceil(nh), math.ceil(nw)]))
            return scaled_imgs

        def conv(self, x, num_out_layers, kernel_size, stride,
                 activation_fn=tf.nn.elu):
            p = np.floor((kernel_size - 1) / 2).astype(np.int32)
            p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
            return slim.conv2d(p_x, num_out_layers, kernel_size, stride,
                               'VALID', activation_fn=activation_fn)

        def conv_block(self, x, num_out_layers, kernel_size):
            conv1 = self.conv(x, num_out_layers, kernel_size, 1)
            conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
            return conv2

        def upconv(self, x, num_out_layers, kernel_size, scale):
            upsample = self.upsample_nn(x, scale)
            conv = self.conv(upsample, num_out_layers, kernel_size, 1)
            return conv

        def deconv(self, x, num_out_layers, kernel_size, scale):
            p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size,
                                         scale, 'SAME')
            return conv[:, 3:-1, 3:-1, :]

        def get_disp(self, x):
            disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
            return disp

        def upsample_nn(self, x, ratio):
            s = tf.shape(x)
            h = s[1]
            w = s[2]
            return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

        def maxpool(self, x, kernel_size):
            p = np.floor((kernel_size - 1) / 2).astype(np.int32)
            p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
            return slim.max_pool2d(p_x, kernel_size)

        def resconv(self, x, num_layers, stride):
            do_proj = tf.shape(x)[3] != num_layers or stride == 2
            shortcut = []
            conv1 = self.conv(x, num_layers, 1, 1)
            conv2 = self.conv(conv1, num_layers, 3, stride)
            conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
            if do_proj:
                shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
            else:
                shortcut = x
            return tf.nn.elu(conv3 + shortcut)

        def resblock(self, x, num_layers, num_blocks):
            out = x
            for i in range(num_blocks - 1):
                out = self.resconv(out, num_layers, 1)
            out = self.resconv(out, num_layers, 2)
            return out

        def build_resnet50(self, x):
            # set convenience functions
            conv = self.conv
            if self.use_deconv:
                upconv = self.deconv
            else:
                upconv = self.upconv

            with tf.variable_scope('encoder'):
                conv1 = conv(x, 64, 7, 2)  # H/2  -   64D
                pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
                conv2 = self.resblock(pool1, 64, 3)  # H/8  -  256D
                conv3 = self.resblock(conv2, 128, 4)  # H/16 -  512D
                conv4 = self.resblock(conv3, 256, 6)  # H/32 - 1024D
                conv5 = self.resblock(conv4, 512, 3)  # H/64 - 2048D

            with tf.variable_scope('skips'):
                skip1 = conv1
                skip2 = pool1
                skip3 = conv2
                skip4 = conv3
                skip5 = conv4

            # DECODING
            with tf.variable_scope('decoder'):
                upconv6 = upconv(conv5, 512, 3, 2)  # H/32
                concat6 = tf.concat([upconv6, skip5], 3)
                iconv6 = conv(concat6, 512, 3, 1)

                upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
                concat5 = tf.concat([upconv5, skip4], 3)
                iconv5 = conv(concat5, 256, 3, 1)

                upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
                concat4 = tf.concat([upconv4, skip3], 3)
                iconv4 = conv(concat4, 128, 3, 1)
                disp4 = self.get_disp(iconv4)
                udisp4 = self.upsample_nn(disp4, 2)

                upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
                concat3 = tf.concat([upconv3, skip2, udisp4], 3)
                iconv3 = conv(concat3, 64, 3, 1)
                disp3 = self.get_disp(iconv3)
                udisp3 = self.upsample_nn(disp3, 2)

                upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
                concat2 = tf.concat([upconv2, skip1, udisp3], 3)
                iconv2 = conv(concat2, 32, 3, 1)
                disp2 = self.get_disp(iconv2)
                udisp2 = self.upsample_nn(disp2, 2)

                upconv1 = upconv(iconv2, 16, 3, 2)  # H
                concat1 = tf.concat([upconv1, udisp2], 3)
                iconv1 = conv(concat1, 16, 3, 1)
                disp1 = self.get_disp(iconv1)

            # Set name for output
            out = tf.expand_dims(disp1[:, :, :, 0], 3, name='output_depth')

            return out

        def build_vgg(self, x):
            # set convenience functions
            conv = self.conv
            if self.use_deconv:
                upconv = self.deconv
            else:
                upconv = self.upconv

            with tf.variable_scope('encoder'):
                conv1 = self.conv_block(x, 32, 7)  # H/2
                conv2 = self.conv_block(conv1, 64, 5)  # H/4
                conv3 = self.conv_block(conv2, 128, 3)  # H/8
                conv4 = self.conv_block(conv3, 256, 3)  # H/16
                conv5 = self.conv_block(conv4, 512, 3)  # H/32
                conv6 = self.conv_block(conv5, 512, 3)  # H/64
                conv7 = self.conv_block(conv6, 512, 3)  # H/128
                # conv_block(x, num_out_layers, kernel_size):

            with tf.variable_scope('skips'):
                skip1 = conv1
                skip2 = conv2
                skip3 = conv3
                skip4 = conv4
                skip5 = conv5
                skip6 = conv6

            with tf.variable_scope('decoder'):
                upconv7 = upconv(conv7, 512, 3, 2)  # H/64
                concat7 = tf.concat([upconv7, skip6], 3)
                iconv7 = conv(concat7, 512, 3, 1)

                upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
                concat6 = tf.concat([upconv6, skip5], 3)
                iconv6 = conv(concat6, 512, 3, 1)

                upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
                concat5 = tf.concat([upconv5, skip4], 3)
                iconv5 = conv(concat5, 256, 3, 1)

                upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
                concat4 = tf.concat([upconv4, skip3], 3)
                iconv4 = conv(concat4, 128, 3, 1)
                disp4 = self.get_disp(iconv4)
                udisp4 = self.upsample_nn(disp4, 2)

                upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
                concat3 = tf.concat([upconv3, skip2, udisp4], 3)
                iconv3 = conv(concat3, 64, 3, 1)
                disp3 = self.get_disp(iconv3)
                udisp3 = self.upsample_nn(disp3, 2)

                upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
                concat2 = tf.concat([upconv2, skip1, udisp3], 3)
                iconv2 = conv(concat2, 32, 3, 1)
                disp2 = self.get_disp(iconv2)
                udisp2 = self.upsample_nn(disp2, 2)

                upconv1 = upconv(iconv2, 16, 3, 2)  # H
                concat1 = tf.concat([upconv1, udisp2], 3)
                iconv1 = conv(concat1, 16, 3, 1)
                disp1 = self.get_disp(iconv1)
                # batch_size, height, width, channels = disp1.get_shape(
                # ).as_list()

            # Set name for output
            out = tf.expand_dims(disp1[:, :, :, 0], 3, name='output_depth')

            return out

        def build_model(self, x, model):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.elu):
                with tf.variable_scope('model', reuse=None):
                    # build model
                    if model == 'vgg':
                        return self.build_vgg(x)
                    elif model == 'resnet':
                        return self.build_resnet50(x)
                    else:
                        return None

    # Arguments
    Args = namedtuple('arg',
                      'batch_size encoder img_height img_width ckpt_file '
                      'output_dir graph')

    output_dir = output_dir or str(Path(checkpoint_file).parent.resolve())
    height, width = MONODEPTH_INPUT_SIZE
    arg = Args(2, MONODEPTH_MODEL, height, width, checkpoint_file,
               'models', os.path.join(output_dir, f'{checkpoint_file}.pb'))

    # Image placeholder
    x = tf.placeholder(
        tf.float32,
        shape=[arg.batch_size, arg.img_height, arg.img_width, 3],
        name='input_image')

    logging.info(f'Loading model from {checkpoint_file}')
    # Load model and get output (disparity)
    model = monodepth(x, arg.encoder)
    y = model.output
    logging.info(f'Model loaded.')

    # add pb extension if not present
    if not arg.graph.endswith(".pb"):
        arg.graph = arg.graph + ".pb"

    # initialise the saver
    saver = tf.train.Saver()

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # restore all variables from checkpoint
    saver.restore(sess, arg.ckpt_file)

    # node that are required output nodes
    output_node_names = ["model/output_depth"]

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        # The graph_def is used to retrieve the nodes
        output_node_names
        # The output node names are used to select the useful nodes
    )

    # convert variables to constants
    output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def)

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(arg.graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    logging.info('Frozen graph file {} created successfully'.format(arg.graph))
