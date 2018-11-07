#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
from kaffe import KaffeError, print_stderr
from kaffe.tensorflow import TensorFlowTransformer


def fatal_error(msg):
    print_stderr(msg)
    exit(-1)


def validate_arguments(args):
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')
    if (args.caffemodel is not None) and (args.data_output_path is None):
        fatal_error('No output data path provided.')
    if (args.code_output_path is None) and (args.data_output_path is None):
        fatal_error('No output path specified.')


def convert(def_path, caffemodel_path, data_output_path, code_output_path, phase):
    try:
        transformer = TensorFlowTransformer(def_path, caffemodel_path, phase=phase)
        print_stderr('Converting data...')
        if caffemodel_path is not None:
            data = transformer.transform_data()
            print_stderr('Saving data...')
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        if code_output_path:
            print_stderr('Saving source...')
            with open(code_output_path, 'wb') as src_out:
                b = bytes(transformer.transform_source(), encoding='utf-8')
                src_out.write(b)
        print_stderr('Done.')
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output-path', help='Converted data output path')
    parser.add_argument('--code-output-path', help='Save generated source to this path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    validate_arguments(args)
    convert(args.def_path, args.caffemodel, args.data_output_path, args.code_output_path,
            args.phase)


def my_main():
    def_path = 'models/reference_model/deploy_nodist.prototxt'
    caffe_model_path = 'models/reference_model/model.caffemodel'
    data_output_path = 'userguide_reference.npy'
    code_output_path = 'userguide_reference.py'

    convert(def_path, caffe_model_path, data_output_path, code_output_path, 'test')


if __name__ == '__main__':
    # main()
    my_main()