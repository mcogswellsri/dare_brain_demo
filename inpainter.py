import os.path as pth
import uuid
import tempfile

import cv2
import skimage.transform
import skimage.util

import numpy as np
import tensorflow as tf
import neuralgym as ng

import nibabel as nib

import sys
sys.path.append('./generative_inpainting/')
from inpaint_model import InpaintCAModel

class Inpainter:
    def __init__(self, params):
        # configuration for the inpainting model
        self.FLAGS = ng.Config('generative_inpainting/inpaint.yml')
        # ng.get_gpus(1)
        #args, unknown = parser.parse_known_args()
        #sess_config = tf.ConfigProto(device_count={'GPU': 0})
        self.size = (256, 256)
        self.checkpoint_dir = params.checkpoint_dir

    def make_mask(self, image, x, y, w, h):
        mask = np.zeros(self.size + (3,))
        y1 = int(y * mask.shape[0])
        y2 = int((y+h) * mask.shape[0])
        x1 = int(x * mask.shape[1])
        x2 = int((x+w) * mask.shape[1])
        mask[y1:y2, x1:x2] = 255.0
        print('mask shape ', mask.shape)
        return mask


    def inpaint(self, image, mask, resize=True):
        #image = cv2.imread(args.image)
        #mask = cv2.imread(args.mask)
        # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

        tf.reset_default_graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess_config = sess_config

        self.model = InpaintCAModel()

        in_size = None
        if (image.shape != self.size) and resize:
            in_size = image.shape
            image = skimage.transform.resize(image, self.size)
            image = skimage.util.img_as_ubyte(image)
        else:
            raise Error(f'incorrect image size {image.shape}')
        assert mask.shape[:2] == self.size

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        with tf.Session(config=self.sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = self.model.build_server_graph(self.FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(self.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
            result = sess.run(output)

        if in_size is not None:
            result = skimage.transform.resize(result[0], in_size)
            result = skimage.util.img_as_ubyte(result)
            result = result[None]
        out_image = result[0][:, :, ::-1]
        return out_image, result
