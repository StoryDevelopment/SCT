import numpy as np
import tensorflow as tf

from read_data import DataSet



class Evaluation(object):
    def __init__(self, data_type, global_step, idxs, loss, tensor_dict=None):
        self.data_type = data_type
        self.global_step = global_step
        self.idxs = idxs
        # self.prediction = correct
        # self.num_examples = len(correct)
        self.tensor_dict = None
        self.dict = {'data_type': data_type,
                     'global_step': global_step,
                     # 'prediction': correct,
                     # 'idxs': idxs,
                     }

        self.summaries = None

    def __repr__(self):
        return "{} step {}".format(self.data_type, self.global_step)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_idxs = self.idxs + other.idxs
        new_tensor_dict = None
        if self.tensor_dict is not None:
            new_tensor_dict = {key: val + other.tensor_dict[key] for key, val in self.tensor_dict.items()}
        return Evaluation(self.data_type, self.global_step,  self.correct, self.loss ,tensor_dict=new_tensor_dict)

    def __radd__(self, other):
        return self.__add__(other)


class LabeledEvaluation(Evaluation):
    def __init__(self, data_type, global_step, idxs, loss,tensor_dict=None):
        super(LabeledEvaluation, self).__init__(data_type, global_step, idxs, loss, tensor_dict=tensor_dict)


class AccuracyEvaluation(LabeledEvaluation):
    def __init__(self, data_type, global_step, idxs,acc,loss,num_examples,wrongs,rights,tensor,ans,wrongs_id,rights_id,tensor_dict=None):
        super(AccuracyEvaluation, self).__init__(data_type, global_step, idxs, loss, tensor_dict=tensor_dict)
        self.loss = loss
        self.rights=rights
        self.wrongs=wrongs
        self.tensor=tensor
        self.ans=ans
        self.rights_id=rights_id
        self.wrongs_id=wrongs_id

        self.acc = acc
        self.dict['loss'] = loss
        self.num_examples=num_examples
        self.dict['acc'] = self.acc
        self.dict['rights'] = self.rights
        self.dict['wrongs'] = self.wrongs

        self.dict['rights_id'] = self.rights_id
        self.dict['wrongs_id'] = self.wrongs_id
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=self.loss)])
        acc_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=self.acc)])
        self.summaries = [loss_summary, acc_summary]

    def __repr__(self):
        return "{} step {}: accuracy={}, loss={}".format(self.data_type, self.global_step, self.acc, self.loss)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        acc = (self.acc*self.num_examples+other.acc* other.num_examples) /(self.num_examples+other.num_examples)
        # new_correct = self.correct + other.correct
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / (self.num_examples+other.num_examples)
        # if self.tensor_dict is not None:
        num_examples=self.num_examples+other.num_examples
        self.wrongs.extend(other.wrongs)
        self.rights.extend(other.rights)
        self.wrongs_id.extend(other.wrongs_id)
        self.rights_id.extend(other.rights_id)
        self.ans.extend(other.ans)
        #     new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return AccuracyEvaluation(self.data_type, self.global_step, new_idxs,  acc, new_loss,num_examples, self.wrongs,self.rights,None,self.ans,self.wrongs_id,self.rights_id,tensor_dict=None)
class Evaluator(object):
    def __init__(self, config, model, tensor_dict=None):
        self.config = config
        self.model = model
        self.global_step = model.global_step
        # self.yp = model.pre
        self.tensor_dict = {} if tensor_dict is None else tensor_dict


    def get_evaluation_from_batches(self, sess, batches):
        e = sum(self.get_evaluation(sess, batch) for batch in batches)
        return e
class LabeledEvaluator(Evaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(LabeledEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        # self.y = model.y
        self.prediction= model.prediction


class AccuracyEvaluator(LabeledEvaluator):
    def __init__(self, num_candidate,config, model, tensor_dict=None):
        super(AccuracyEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.loss = model.loss
        self.tensor_dict=model.tensor_dict
        # self.correct=model.correct
        self.num_candidate=num_candidate
        self.prediction=model.prediction

    def _split_batch(self, batches):
        idxs_list, data_sets = zip(*batches)
        idxs = sum(idxs_list, ())
        data_set = sum(data_sets, data_sets[0].get_empty())
        return idxs, data_set
    def get_evaluation(self, sess, batch):
        (idxs, data_set) = self._split_batch(batch)


        # data_set = batch
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)

        global_step,loss, prediction,tensor = sess.run([self.global_step,self.loss, self.prediction,self.tensor_dict['outfinal']],
                                                 feed_dict=feed_dict)
        if data_set.data_type=='train':
            can_num=self.config.train_num_can
        else:
            can_num = self.config.test_num_can
        answers=np.array(data_set.data['answerss'])
        answers=np.reshape(answers,[-1,can_num])
        prediction = np.reshape(prediction, [-1, can_num])
        answers=np.argmax(answers, axis=1)
        prediction=np.argmax(prediction, axis=1)
        correct= answers==prediction[:len(answers)]
        wrongs=[]
        rights=[]
        wrongs_id=[]
        rights_id=[]
        ans=[]
        for i,c in enumerate(correct):
            ans.append([data_set.data['p'][i*2],answers[i]+1])

            if c==0:
                wrongs.append(idxs[i*2] /2+1)
                wrongs_id.append(data_set.data['p'][i*2])

            else:
                rights.append(idxs[i*2] /2+1)
                rights_id.append(data_set.data['p'][i*2])

        acc=sum(correct)/len(correct)



        e = AccuracyEvaluation(data_set.data_type, int(global_step), idxs, acc, float(loss), len(correct), wrongs,
                               rights, tensor,ans,wrongs_id,rights_id, tensor_dict=None)
        return e

    @staticmethod
    def compare(yi, ypi):
        for start, stop in yi:
            if start == int(np.argmax(ypi)):
                return True
        return False














