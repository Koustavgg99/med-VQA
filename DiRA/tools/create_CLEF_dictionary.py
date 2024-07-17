"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_CLEF import Dictionary

def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'All_train.csv'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = pd.read_csv(question_path, sep='|', names=['image_name', 'qid', 'questions', 'answers','question_type','answer_type'])
        for index, row in qs.iterrows():
            dictionary.tokenize(row['questions'], True)
    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r', encoding="utf8") as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    CLEF_dir = r'D:\PhD\Project\VQA_Dataset'
    d = create_dictionary(CLEF_dir)
    d.dump_to_file(os.path.join(CLEF_dir,'dictionary.pkl'))
    d = Dictionary.load_from_file(os.path.join(CLEF_dir, 'dictionary.pkl'))

    emb_dim = 300
    glove_file = os.path.join(CLEF_dir,'glove', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(os.path.join(CLEF_dir, 'glove6b_init_%dd.npy' % emb_dim), weights)

    #question_path = os.path.join(Slake_dir, 'train.json')
    #print(question_path)
    #qs = json.load(open(question_path, encoding="utf8"))
    #qid_list = []
    #for q in qs:
    #    if q['q_lang'] == 'en':
    #        print(q)
    #        break
    #        qid_list.append(q['qid'])
    #qid_list.sort()

    #print(qid_list)
    #print("len of qid list: ",len(qid_list))

    

    #d = create_dictionary(Slake_dir)

    #d = create_dictionary(Slake_dir)