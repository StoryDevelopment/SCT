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
from model_c import get_multi_gpu_models
from trainer import MultiGPUTrainer
from read_data import read_data, update_config
from my.tensorflow import get_num_params


def main(config):
    set_dirs(config)
    with tf.device(config.device):
        if config.mode == 'train':
            _train(config)
        elif config.mode == 'test':
            _test(config)

        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
    # create directories
    assert config.load or config.mode == 'train', "config.load must be True if not training"
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


def _train(config):
    train_data = read_data(config, 'val_train', config.load)
    dev_data = read_data(config, 'val_val', True)
    # test = read_data(config, 'test', True)
    update_config(config, [train_data, dev_data])

    _config_debug(config)

    word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
    word2idx_dict = train_data.shared['word2idx']
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat

    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    trainer = MultiGPUTrainer(config, models)
    evaluator = AccuracyEvaluator(config.train_num_can, config, model,
                                  tensor_dict=model.tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config,
                                 model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)

    # Begin training
    num_steps = config.num_steps or int(
        math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
    global_step = 0
    best_dev=[0,0]

    for batches in tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus,
                                                     num_steps=num_steps, shuffle=False, cluster=config.cluster), total=num_steps):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary)

        if get_summary:
            graph_handler.add_summary(summary, global_step)


        if not config.eval:
            continue

        if global_step % config.eval_period == 0:

            num_steps_dev = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
            num_steps_train = math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))


            e_train = evaluator.get_evaluation_from_batches(
                sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps_train),
                           total=num_steps_train)
            )
            # graph_handler.add_summaries(e_test.summaries, global_step)
            e_dev = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps_dev),
                           total=num_steps_dev))
            # graph_handler.dump_eval(e)
            # graph_handler.add_summaries(e_dev.summaries, global_step)
            print('train step:{}  loss:{}  acc:{}'.format(global_step, e_train.loss, e_train.acc))
            print('val step:{}  loss:{}  acc:{}'.format(global_step, e_dev.loss, e_dev.acc))
            # print('w_s:{}'.format(w_s))
            if global_step > 700:
                 config.save_period = 50
                 config.eval_period = 50

            if best_dev[0] < e_dev.acc:
                best_dev=[e_dev.acc,global_step,e_train.acc]
                graph_handler.save(sess, global_step=global_step)



            # if config.dump_eval:
            #     graph_handler.dump_eval(e_dev)

    if global_step % config.save_period != 0:
        graph_handler.save(sess, global_step=global_step)
    print (best_dev)
    print ("you can test on test data set and set load setp is {}".format(best_dev[1]))



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
    tensor=[]
    for i, multi_batch in enumerate(tqdm(
            test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps,
                                        cluster=config.cluster), total=num_steps)):

        ei = evaluator.get_evaluation(sess, multi_batch)
        # outfinal=ei.tensor
        # tensor.extend(outfinal)

        e = ei if e is None else e + ei
        # if config.vis:
        #     eval_subdir = os.path.join(config.eval_dir,
        #                                "{}-{}".format(multi_batch[0][1].data_type, str(ei.global_step).zfill(6)))
        #     if not os.path.exists(eval_subdir):
        #         os.mkdir(eval_subdir)
        #     path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
        #     graph_handler.dump_eval(ei, path=path)

    print(e.acc)

    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e)
    if config.dump_answer:
        print("dumping answers ...")
        graph_handler.dump_answer(e)
        # import pickle
        # f =open(config.eval_dir+'/output_val.json','wb')
        # pickle.dump(tensor,f)
        # f.close()


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
