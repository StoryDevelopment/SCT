import argparse
import json
import os
import numpy as np
from collections import Counter
from tqdm import tqdm

negative=open('BingLiuList/negative-words.txt','r').read().split('\n')
positive=open('BingLiuList/positive-words.txt','r').read().split('\n')
negation=open('negation','r').read().split('\n')

pos_dic=json.load(open('pos_dic.json','r'))
def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("")
    source_dir = os.path.join('', "db")
    target_dir = os.path.join(home, "data/all_feature_hao_only3_100/")
    train_negative_num=0
    glove_vec_size=100
    glove_dir = os.path.join('', "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-n', "--train_negative_num",type=int, default=train_negative_num)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument("--train_name", default='train_data')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=glove_vec_size, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    prepro(args)

def prepro(args):
    # prepro_each(args, 'train', out_name='train')
    prepro_each(args, 'val', out_name='val')
    prepro_each(args, 'test', out_name='test')
    # prepro_together(args,['train','val'],out_name='train')

def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict

def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}_{}.json".format(data_type,args.glove_vec_size))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))
def prepro_together(args,data,out_name):
    q, cq, rx, rcx = [], [], [], []
    x, cx = [], []
    answerss = []
    p = []
    ai=0
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    for i, d in enumerate(data):
        qi, cqi, rxi, rcxi, word_counter, char_counter, lower_word_counter, answerssi, xi, cxi, aii, pi = prepro_return(
            args, d, word_counter, char_counter, lower_word_counter, ai)
        q.extend(qi)
        cq.extend(cqi)
        rx.extend(rxi)
        rcx.extend(rcxi)
        answerss.extend(answerssi)
        x.extend(xi)
        cx.extend(cxi)
        ai = ai + aii
    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)
    data = {'q': q, 'cq': cq, 'answerss': answerss, '*x': rx, '*cx': rcx, }
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}
    print("saving ...")
    save(args, data, shared, out_name)

def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    haoruopeng_feature=np.load('haoruopeng/{}_feature.npy'.format(data_type)).tolist()
    source_path=os.path.join(args.source_dir,'{}_data.json'.format(data_type))
    dataset=json.load(open(source_path, 'r'))
    if data_type == 'train':
        candidtate_num = args.train_negative_num + 1
        context_move=1
    else:
        candidtate_num = 2
        context_move = 0
    import nltk
    sent_tokenize = nltk.sent_tokenize

    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"').replace('-',' ') for token in nltk.word_tokenize(tokens)]
    q, cq,rx, rcx= [], [],[],[]
    q_pos, q_sem, x_pos, x_sem,q_neg,x_neg = [], [], [], [], [], []

    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()

    for aii, article in enumerate(tqdm(dataset)):

        # context = article[1+context_move]+" "+article[2+context_move]+" "+article[3+context_move]+" "+article[4+context_move]
        context = article[3 + context_move]
        context = context.replace("''", ' ')
        context = context.replace("``", ' ').replace('-',' ').replace('  ',' ')
        a=nltk.word_tokenize(context)
        x_semi=[]
        x_negi=[]
        for ai in a:
            if ai in negative:
                x_semi.append(1)
            elif ai in positive:
                x_semi.append(2)
            else:
                x_semi.append(0)
            if ai in negation:
                x_negi.append(1)
            else:
                x_negi.append(0)
        xi=[]
        xposi=[]

        x_neg_i=[]

        x_neg_i.append(x_negi)
        b = nltk.pos_tag(a)
        # pos = [bi[1] for bi in b]
        pos=[pos_dic[bi[1]] for bi in b]
        xposi.append(pos)
        xi.append(a)
        x_semii=[]
        x_semii.append(x_semi)
        # xi = [process_tokens(tokens) for tokens in xi]

        cxi = [[list(xijk) for xijk in xij] for xij in xi]
        x.append(xi)
        cx.append(cxi)
        p.append(context)
        for xij in xi:
            for xijk in xij:
                word_counter[xijk] += candidtate_num
                lower_word_counter[xijk.lower()] += candidtate_num
                for xijkl in xijk:
                    char_counter[xijkl] += candidtate_num
        rxi=[aii]
        if data_type == 'train':
            right_id=0
        else:
            right_id=int(article[7])-1

        for q_id in range(candidtate_num):
            ending = article[q_id+context_move+5]
            ending = ending.replace("''", ' ')
            ending = ending.replace("``", ' ').replace('-', ' ').replace('  ', ' ')
            qi=word_tokenize(ending)
            # q_posi = [p[1] for p in nltk.pos_tag(qi)]
            q_posi = [pos_dic[p[1]] for p in nltk.pos_tag(qi)]
            q_semi=[]
            q_negi=[]
            for ai in qi:
                if ai in negative:
                    q_semi.append(1)
                elif ai in positive:
                    q_semi.append(2)
                else:
                    q_semi.append(0)
                if ai in negation:
                    q_negi.append(1)
                else:
                    q_negi.append(0)
            # qi = process_tokens(qi)
            cqi = [list(qij) for qij in qi]

            # answer
            if q_id==right_id:
                answer=1
            else :answer=0
            answerss.append(answer)

            for qij in qi:
                word_counter[qij] += 1
                lower_word_counter[qij.lower()] += 1
                for qijk in qij:
                    char_counter[qijk] += 1

            q.append(qi)
            cq.append(cqi)
            rx.append(rxi)
            rcx.append(rxi)
            q_pos.append(q_posi)
            x_pos.append(xposi)
            x_sem.append(x_semii)
            q_sem.append(q_semi)
            x_neg.append(x_neg_i)
            q_neg.append(q_negi)
    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)
    data = {'q': q, 'cq': cq,  'answerss': answerss,'*x': rx, '*cx': rcx, 'q_pos':q_pos,\
            'q_neg':q_neg,'q_sem':q_sem,'x_pos':x_pos,'x_sem':x_sem,'x_neg':x_neg,'haoruopeng_feature':haoruopeng_feature}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}
    print("saving ...")
    save(args, data, shared, out_name)
def prepro_return(args, data_type,word_counter,char_counter,lower_word_counter,ai):
    source_path=os.path.join(args.source_dir,'{}_data.json'.format(data_type))
    dataset=json.load(open(source_path, 'r'))
    if data_type == 'train':
        candidtate_num = args.train_negative_num + 1
        context_move=1
    else:
        candidtate_num = 2
        context_move = 0
    import nltk
    sent_tokenize = nltk.sent_tokenize

    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    q, cq,rx, rcx= [], [],[],[]

    x, cx = [], []
    answerss = []
    p = []
    # word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    if len(dataset)>2000:
        dataset=dataset[:2000]
    for  article in tqdm(dataset):

        context = article[1+context_move]+" "+article[2+context_move]+" "+article[3+context_move]+" "+article[4+context_move]
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')
        a=nltk.word_tokenize(context)

        xi=[]
        # xi = list(map(word_tokenize, a))
        # xii = []
        xi.append(a)
        # xii=xi[0]
        # xii.extend(xi[1])
        # xii.append(xi[2])
        # xii.append(xi[3])
        #
        # xi = []
        # xi.append(xii)
        xi = [process_tokens(tokens) for tokens in xi]
        cxi = [[list(xijk) for xijk in xij] for xij in xi]
        x.append(xi)
        cx.append(cxi)
        p.append(context)
        for xij in xi:
            for xijk in xij:
                word_counter[xijk] += candidtate_num
                lower_word_counter[xijk.lower()] += candidtate_num
                for xijkl in xijk:
                    char_counter[xijkl] += candidtate_num
        rxi=[ai]
        if data_type == 'train':
            right_id=0
        else:
            right_id=int(article[7])-1

        for q_id in range(candidtate_num):
            qi=word_tokenize(article[q_id+context_move+5])
            qi = process_tokens(qi)
            cqi = [list(qij) for qij in qi]

            # answer
            if q_id==right_id:
                answer=1
            else :answer=0
            answerss.append(answer)

            for qij in qi:
                word_counter[qij] += 1
                lower_word_counter[qij.lower()] += 1
                for qijk in qij:
                    char_counter[qijk] += 1\

            q.append(qi)
            cq.append(cqi)
            rx.append(rxi)
            rcx.append(rxi)
        ai+=1
    return q,cq,rx,rcx,word_counter,char_counter,lower_word_counter,answerss,x,cx,ai,p



if __name__ == "__main__":
    main()