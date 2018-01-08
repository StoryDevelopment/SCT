import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from evaluator import AccuracyEvaluator
from graph_handler import GraphHandler
from model3 import get_multi_gpu_models
from trainer import MultiGPUTrainer
from read_data import read_data, update_config
from my.tensorflow import get_num_params


def main(config):
    set_dirs(config)
    with tf.device(config.device):
        # if config.mode == 'train':
        #     _train(config)
        if config.mode == 'test':
            _test(config)
        # elif config.mode == 'val':
        #     _val(config)
        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
    # create directories
    assert config.load or config.mode == 'train' or 'val', "config.load must be True if not training"
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    config.answer_dir = os.path.join(config.out_dir, "answer")
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.answer_dir):
        os.mkdir(config.answer_dir)
    if not os.path.exists(config.eval_dir):
        os.mkdir(config.eval_dir)


def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2



def _test(config):
    test_data = read_data(config, 'test', True)
    update_config(config, [test_data])

    _config_debug(config)

    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    evaluator = AccuracyEvaluator(config.test_num_can, config, model,
                                  tensor_dict=models[0].tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)
    num_steps = math.ceil(test_data.num_examples / (config.batch_size * config.num_gpus))

    e = None
    for i, multi_batch in enumerate(tqdm(
            test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps,
                                        cluster=config.cluster), total=num_steps)):

        ei = evaluator.get_evaluation(sess, multi_batch)
        e = ei if e is None else e + ei
        # if config.vis:
        #     eval_subdir = os.path.join(config.eval_dir,
        #                                "{}-{}".format(multi_batch[0][1].data_type, str(ei.global_step).zfill(6)))
        #     if not os.path.exists(eval_subdir):
        #         os.mkdir(eval_subdir)
        #     path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
        #     graph_handler.dump_eval(ei, path=path)

    print("test acc:{}".format(e.acc))

    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        main(config)


if __name__ == "__main__":
    _run()
