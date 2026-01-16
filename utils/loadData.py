import numpy as np
from PIL import Image
import random, pdb
import pickle
import re
import json
from torch.utils.data import Dataset, DataLoader
from typing import *
from transformers import T5TokenizerFast,RobertaTokenizerFast
import torch
from .prompt import prompt_train, prompt_stage2_u, prompt_test, prompt_expl, prompt_stage2,prompt_roberta, prompt_dgm4, prompt_stage2_dgm4
from PIL import Image
from models.transforms import Compose,RandomResize,ToTensor,Normalize,Padding,Resize
from tqdm import tqdm
import torch.nn.functional as F
from .eval import giou_losses
import json
from utils.eval import loss_by_feat_single, bbox_cxcywh_to_xyxy, loss_by_feat_single_zc

loss_func = giou_losses


# Class To Load MSD data
wordPrefix = "data/mmsd2/extract/"
dataPrefix = "data/mmsd2/text/"
imagePrefix = "data/MSD/imageVector2/"

class TextItem():
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.label = label
        self.words = None

class TextIterator():
    def __init__(self, batchSize = 32, seqLen = 75):
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.textData = dict()
        self.trainNum = []
        self.validNum = []
        self.testNum = []
        dictExtractWords = self.getExtractDict()
        for i in range(3):
            self.readData(i, dictExtractWords)
        self.batchInd = 0
        self.validInd = 0
        self.testInd = 0
        self.epoch = 0
        self.threshold = int(len(self.trainNum) / self.batchSize)
        # print('train set:',len(self.trainNum),'vaild set:', len(self.validNum),'test set:', len(self.testNum))
        # print("rate: ", self.rate)

    def getExtractDict(self):
        file = open(wordPrefix+"extract_all")
        dic = {}
        for line in file:
            ls = eval(line)
            dic[int(ls[0])] = ls[1:]
        return dic

    def readData(self, i, dic):
        # print(i)
        p = n = 0
        if i == 0:
            file = open(dataPrefix+"train.txt")
            ls = self.trainNum
        elif i == 1:
            file = open(dataPrefix+"valid2.txt")
            ls = self.validNum
        else:
            file = open(dataPrefix+"test2.txt")
            ls = self.testNum
        for line in file:
            lineLS = eval(line)
            tmpLS = lineLS[1].split()
            if "sarcasm" in tmpLS:
                continue
            if "sarcastic" in tmpLS:
                continue
            if "reposting" in tmpLS:
                continue
            if "<url>" in tmpLS:
                continue
            if "joke" in tmpLS:
                continue
            if "humour" in tmpLS:
                continue
            if "humor" in tmpLS:
                continue
            if "jokes" in tmpLS:
                continue
            if "irony" in tmpLS:
                continue
            if "ironic" in tmpLS:
                continue
            if "exgag" in tmpLS:
                continue
            # assert int(lineLS[0]) not in self.textData
            ls.append(int(lineLS[0]))
            if i == 0:
                if lineLS[-1] == 1:
                    p += 1
                else:
                    n += 1
            # pdb.set_trace()
            # lineLS: ['840006160660983809', '<user> thanks for showing up for our appointment today . ', 1]
            self.textData[int(lineLS[0])] = TextItem(lineLS[1], int(lineLS[-1])) # sentence, label, word
            self.textData[int(lineLS[0])].words = dic[int(lineLS[0])]
        random.shuffle(ls)
        if i == 0:
            self.rate = float(n) / p

    def getTestData(self):
        text_data = []
        image_data = []
        id_list = []
        label_list = []
        for id in self.testNum:
            text = self.textData[id].sentence
            label = self.textData[id].label
            img_path = '/data1/zc/dataset/MMSD2.0/dataset_image/' + str(id) +'.jpg'
            text_data.append(text)
            image_data.append(img_path)
            id_list.append(id)
            label_list.append(label)
        return text_data, image_data, id_list, label_list

    def getTrainData(self):
        text_data = []
        image_data = []
        id_list = []
        label_list = []
        for id in self.trainNum:
            text = self.textData[id].sentence
            label = self.textData[id].label
            img_path = '/data1/zc/dataset/MMSD2.0/dataset_image/' + str(id) +'.jpg'
            text_data.append(text)
            image_data.append(img_path)
            id_list.append(id)
            label_list.append(label)
        return text_data, image_data, id_list, label_list

    def getValidData(self):
        text_data = []
        image_data = []
        id_list = []
        label_list = []
        for id in self.validNum:
            text = self.textData[id].sentence
            label = self.textData[id].label
            img_path = '/data1/zc/dataset/MMSD2.0/dataset_image/' + str(id) +'.jpg'
            text_data.append(text)
            image_data.append(img_path)
            id_list.append(id)
            label_list.append(label)
        return text_data, image_data, id_list, label_list

class TextIterator_dgm4():
    def __init__(self, batchSize = 32, seqLen = 75):
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.textData = dict()
        self.trainNum = []
        self.validNum = []
        self.testNum = []
        # for i in range(3):
        #     self.readData(i)
        self.batchInd = 0
        self.validInd = 0
        self.testInd = 0
        self.epoch = 0
        self.threshold = int(len(self.trainNum) / self.batchSize)


    def getTestData(self):
        text_data = []
        image_data = []
        id_list = []
        label_list = []
        Ffile = open('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/data/dgm4/test0.txt')
        Ffile = open('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/data/dgm4/train.txt')
        # '/media/disk/01drive/01congzhang/dataset/DGM4/metadata/'
        for line in Ffile:
            lineLS = eval(line)
            img_path = '/media/disk/01drive/01congzhang/dataset/DGM4/' + lineLS[0] +'.jpg'
            text_data.append(lineLS[1])
            image_data.append(img_path)
            id_list.append(lineLS[0])
            label_list.append(int(lineLS[-1]))
        return text_data, image_data, id_list, label_list

    def getTrainData(self):
        text_data = []
        image_data = []
        id_list = []
        label_list = []
        Ffile = open('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/data/dgm4/train.txt')
        for line in Ffile:
            lineLS = eval(line)
            img_path = '/media/disk/01drive/01congzhang/dataset/DGM4/' + lineLS[0] +'.jpg'
            text_data.append(lineLS[1])
            image_data.append(img_path)
            id_list.append(lineLS[0])
            label_list.append(int(lineLS[-1]))
        return text_data, image_data, id_list, label_list

    def getValidData(self):
        text_data = []
        image_data = []
        id_list = []
        label_list = []
        Ffile = open('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/data/dgm4/val.txt')
        for line in Ffile:
            lineLS = eval(line)
            img_path = '/media/disk/01drive/01congzhang/dataset/DGM4/' + lineLS[0] +'.jpg'
            text_data.append(lineLS[1])
            image_data.append(img_path)
            id_list.append(lineLS[0])
            label_list.append(int(lineLS[-1]))
        return text_data, image_data, id_list, label_list

def getScore(p, y):
    tp = fp = tn = fn = 0
    for i in range(len(p)):
        if y[i] == 1:
            if p[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if p[i] == 1:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn

def getF1(tp, fp, tn, fn):
    try:
        pre = float(tp) / (tp+fp)
        rec = float(tp) / (tp+fn)
        f1 = 2*pre*rec / (pre+rec)
    except:
        pre = rec = f1 = 0
    return pre, rec, f1

import os
def load_sentence(tweet_data_dir, tpye_path):
    """
    read the word from doc, and build sentence. every line contain a word and it's tag
    every sentence is split with a empty line. every sentence begain with an "IMGID:num"

    """

    IMAGEID='IMGID'
    img_id = []
    sentences = []
    sentence = []
    datasplit = []

    assert tpye_path in ['train', 'val', 'test']

    datasplit.append(len(img_id))
    with open(os.path.join(tweet_data_dir, tpye_path), 'r', encoding='utf-8') as file:
        last_line = ''
        for line in file:
            line = line.rstrip()
            if line == '':
                sentences.append(sentence)
                sentence = []
            else:
                if IMAGEID in line:
                    num = line[6:]
                    img_id.append(num)
                    if last_line != '':
                        print(num)
                else:
                    if len(line.split()) == 1:
                        print(line)
                    sentence.append(line.split())
            last_line = line

    targets = []
    out_sentences = []
    for sentence in sentences:
        target = []
        for word in sentence:
            try:
                if word[1] != 'O':
                    target.append(word[0])
            except IndexError:
                print(sentence)
        sen = ' '.join([word[0] for word in sentence])
        out_sentences.append(sen)
        targets.append(target)

    return out_sentences, targets, img_id, sentences


class T5DataSet(Dataset):
    def __init__(self, type_path, tokenizer = None, max_examples=-1,
                 max_src_len=512, max_tgt_len=200):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        self.max_examples = max_examples
        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len
        self.use_rationale = False      # aborted!! distil step by step setting.
        self.type_path = type_path
        self.text_indicator = None

        # self.texti = TextIterator_dgm4()
        self.texti = TextIterator()
        assert type_path in ['train','test','val']
        self._build()       # fill inputs, targets, max_lens
        print('done loading',type_path,'set with length of',len(self.img_data))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        src_text = self.source_data[index] # len=19816  'the most exciting part of being an author !'
        tgt_text = self.target_text[index] # 'Yes'
        image_id = self.img_data[index] # mmsd2/dataset_image/918159503866052608.jpg
        return {"source_text": src_text, "target_text": tgt_text, "image_ids": image_id}

    def _build(self):
        print("buliding dataset...")
        source_expl = []    # for rationale distillation
        target_expl = []
        self.target_text = []
        # target = []
        targetdict = {'0':'No','1':'Yes'}
        self.source_data = []
        if self.type_path == 'train':
            self.use_rationale = False
            self.text_data, self.img_data, self.id_list, self.label_data = self.texti.getTrainData()
            for idx in range(len(self.id_list)):
                text = self.text_data[idx]
                label = self.label_data[idx]
                text = prompt_dgm4.format(text)
                self.source_data.append(text)
                self.target_text.append(targetdict[str(self.label_data[idx])])
        else:
            if self.type_path == 'test':
                self.text_data, self.img_data, self.id_list, self.label_data  = self.texti.getTestData()
                for idx in range(len(self.id_list)):
                    label = self.label_data[idx] #self.texti.textData[self.id_list[idx]].label
                    text =  self.text_data[idx]
                    text = prompt_dgm4.format(text)
                    self.source_data.append(text)
                    self.target_text.append(targetdict[str(self.label_data[idx])])
            else:
                self.text_data, self.img_data, self.id_list, self.label_data = self.texti.getValidData()
                for idx in range(len(self.id_list)):
                    label = self.label_data[idx]#self.texti.textData[self.id_list[idx]].label
                    text =  self.text_data[idx]
                    text = prompt_dgm4.format(text)
                    self.source_data.append(text)
                    self.target_text.append(targetdict[str(self.label_data[idx])])
        self.source_expl,self.target_expl = [], []


        # for idx in range(len(self.id_list)):
        #     target.append()

        # self.target_text = target
        # pdb.set_trace()

class MSData(Dataset):
    def __init__(self,type,text_path = dataPrefix, img_path=imagePrefix) -> None:
        super().__init__()
        self.text_path = text_path
        self.img_path = img_path
        self.texti = TextIterator()
        if type == "train":
            self.text_data, self.img_data, self.id_list = self.texti.getTrainData()
        elif type == "test":
            self.text_data, self.img_data, self.id_list = self.texti.getTestData()
        else:
            self.text_data, self.img_data, self.id_list = self.texti.getValidData()


    def __getitem__(self, idx):
        img = self.img_data[idx]
        # text = self.text_data[idx]
        text = prompt_test.format(self.text_data[idx])
        label = self.texti.textData[self.id_list[idx]].label
        index = self.id_list[idx]
        return {'image':img, 'text':text, 'label':label,'index':index}
        # return img, text, label, index

    def __len__(self):
        return len(self.id_list)

class MSTIDataSet(Dataset):
    def __init__(self, type_path, tokenizer = None, max_examples=-1,
                 max_src_len=512, max_tgt_len=200):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        self.max_examples = max_examples
        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len

        print('loading',type_path,'set')

        # rationale_path = "{path_to_MSTI_rationale}.json"

        # import json
        # with open(rationale_path,'r') as f:
        #     self.rationale = json.load(f)

        self.type_path = type_path
        assert type_path in ['train','test','val']

        if tokenizer is None:
            self.tokenizer = T5TokenizerFast.from_pretrained('/data1/zc/models/flan-t5-base/')
        else:
            self.tokenizer  = tokenizer

        self._build()       # fill inputs, targets, max_lens
        print('done loading',type_path,'set with length of',len(self.img_data))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        src_text = self.source_data[index]
        # tgt_text = self.target_text[index]
        image_id = self.img_data[index]
        label = self.labels[index]
        tgt_text = []
        # fake_cls = self.fake_cls[index]
        # text_cls = self.text_cls[index]

        img_id = self.id_list[index]
        box_path = os.path.join('./data/msti/Visual target labels3', img_id + '.txt')
        box_f = open(box_path, 'r', encoding='utf-8')
        bboxes = []
        box_labels = []

        for line in box_f.readlines():
            splited = line.strip().split()
            # print(splited)
            num_boxes = len(splited) // 5
            # print(num_boxes)
            for i in range(num_boxes):
                x1 = float(splited[0 + 5 * i])
                y1 = float(splited[1 + 5 * i])
                x2 = float(splited[2 + 5 * i])
                y2 = float(splited[3 + 5 * i])
                c = int(splited[4 + 5 * i])

                x, y, w, h = x1, y1, x2, y2

                bboxes.append(torch.tensor([x, y, w, h]))
                box_labels.append(torch.tensor(c))

        max_len = 10
        out_bboxes = np.array(bboxes, dtype=float)
        out_bboxes1 = np.zeros([max_len, 4])
        out_bboxes1[:min(out_bboxes.shape[0], max_len)] = out_bboxes[:min(out_bboxes.shape[0], max_len)]

        out_labels = np.array(box_labels, dtype=float)
        out_labels1 = np.zeros([10])
        out_labels1[:min(out_labels.shape[0], max_len)] = out_labels[:min(out_labels.shape[0], max_len)]
        # pdb.set_trace()

        # labelS = label.split('],')
        for ith in range(len(label)):
            if label[ith][1] == 'O':
                tgt_text.append('T')
            else:
                tgt_text.append('O')
        if 'O' in tgt_text:
            text_cls = 'swap'
        else:
            text_cls = 'orig'
        if out_labels1.sum() != 0:
            fake_cls = 'swap'
        else:
            fake_cls = 'orig'
        tgt_text = ' '.join(tgt_text)


        return {"source_text": src_text,
                 "image_ids": image_id,
                 "bboxes": out_bboxes1,
                 "box_labels":out_labels1,
                 "text_cls":text_cls,
                 "fake_cls":fake_cls,
                 "target_text": tgt_text,
                 "label": str(label)}  # for now

    def _build(self):

        print("buliding ",self.type_path," dataset...")

        source_data = []
        targets = []
        # text_cls = []
        # fake_cls = []
        images = []
        dir = './data/msti/Textual target labels3'
        sentences, tgts, self.id_list,labels = load_sentence(dir,self.type_path)


        for text,target,idx,label in zip(sentences, tgts, self.id_list, labels):
            # rationale = self.rationale[idx]
            # text = prompt_stage2.format(text,
            #                     rationale)
            images.append(f"./data/msti/img/{idx}.jpg")
            if target == []:
                target = 'None'
            else:
                target = ' '.join(target)
            targets.append(target)
            source_data.append(text)

        self.img_data = images
        self.source_data = source_data
        self.target_text = targets
        self.labels = labels
        # self.fake_cls = fake_cls
        # self.text_cls = text_cls
        # print(target)
        assert len(self.img_data) == len(targets)

class MSTIDataSet_DGM4(Dataset):
    def __init__(self, type_path, tokenizer = None, max_examples=-1,
                 max_src_len=512, max_tgt_len=200):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        self.max_examples = max_examples
        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len
        print('loading',type_path,'set')
        self.type_path = type_path
        assert type_path in ['train','test','val']
        self._build()       # fill inputs, targets, max_lens
        print('done loading',type_path,'set with length of',len(self.img_data))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        src_text = self.source_data[index]
        tgt_text = self.target_text[index]
        image_id = self.img_data[index]
        out_bboxes1 = self.out_bboxes_list[index]
        out_labels1 = self.out_bboxes_label_list[index]
        fake_cls = self.fake_cls[index]
        text_cls = self.text_cls[index]
        fake_m_label = self.fake_m_label[index]

        return {"source_text": src_text, # 经过prompt以后的输入
                 "target_text": tgt_text, # 出现错误的toekn
                 "image_ids": image_id, # 图片的存储位置
                 "bboxes": out_bboxes1, # bbox.size=(10,) 没有bbox就是全零
                 "box_labels":out_labels1,
                 "text_cls":text_cls,
                 "fake_m_label": fake_m_label,
                 "fake_cls":fake_cls} # 有错是1，没错是0

    def get_multiLabel(self, fake_cls):
        if fake_cls == 'orig':
            multi_label = 0
        elif  fake_cls == 'face_swap':
            multi_label = 1
        elif  fake_cls == 'face_attribute':
            multi_label = 2
        elif  fake_cls == 'text_swap':
            multi_label = 3
        elif  fake_cls == 'text_attribute':
            multi_label = 4
        elif  fake_cls == 'face_swap&text_swap':
            multi_label = 5
        elif  fake_cls == 'face_swap&text_attribute':
            multi_label = 6
        elif  fake_cls == 'face_attribute&text_swap':
            multi_label = 7
        elif  fake_cls == 'face_attribute&text_attribute':
            multi_label = 8
        return multi_label


    def _build(self):

        print("buliding ",self.type_path," dataset...")
        dgm4_dir = '/data1/zc/dataset/'

        source_data = []
        targets = []
        images = []
        fake_cls = []
        out_bboxes_list = []
        out_bboxes_label_list = []
        text_cls = []
        fake_m_label = []
        loss_idx = []

        json_file = os.path.join(dgm4_dir+'DGM4/metedata/', self.type_path+'_raw.json')
        # json_file = os.path.join(dgm4_dir+'DGM4/metedata/', self.type_path+'.json')
        # if self.type_path == 'test':
        #     json_file = '/data1/zc/dataset/DGM4/metedata/test_1.json'
            # json_file = os.path.join(dgm4_dir+'DGM4/metedata/', self.type_path+'_raw_loss.json')
            # json_file = os.path.join(dgm4_dir+'DGM4/metadata/', self.type_path+'_loss.json')
            # json_file = '/media/disk/01drive/01congzhang/dataset/DGM4/metadata/test.json'
            

        with open(json_file,'r') as f:
            data = json.load(f)

        for line in data:
            # if self.type_path == 'train':
            #     loss_idx.append(line['loss'])
            images.append(os.path.join(dgm4_dir, line['image']))
            text = prompt_stage2_u.format(line['text'])
            source_data.append(text)
            fake_cls.append(line['fake_cls'])
            fake_m_label.append(self.get_multiLabel(line['fake_cls']))

            target = ['T'] * len(line['text'].split())
            if line['fake_text_pos'] != []:
                for text_index in line['fake_text_pos']:
                    target[text_index] = 'O'
            target = ' '.join(target)
            targets.append(target)

            max_len = 10
            out_bboxes1 = np.zeros([max_len, 4])
            out_labels1 = np.zeros([10])
            if len(line['fake_image_box']) != 0:
                for ith in range(int(len(line['fake_image_box'])/4)):
                    out_bboxes1[ith] = np.array(line['fake_image_box'])
                    # out_labels1[ith] = 1
                    if "face_swap" in line['fake_cls']:
                        out_labels1[ith] = 1
                    elif "face_attribute" in line['fake_cls']:
                        out_labels1[ith] = 2
            out_bboxes_list.append(out_bboxes1)
            out_bboxes_label_list.append(out_labels1)
            # pdb.set_trace()

            if "text_swap" in line['fake_cls']:
                text_cls.append("swap")
            elif  "text_attribute" in line['fake_cls']:
                text_cls.append("attribute")
            else:
                text_cls.append("orig")

        # # for CurriculumLearning
        # idx1 =[index for (index,value) in enumerate(loss_idx) if value <= 0.2]
        # idx2 =[index for (index,value) in enumerate(loss_idx) if 0.2 < value < 1.3]
        # idx3 =[index for (index,value) in enumerate(loss_idx) if value >= 1.3]        
        
        # random. shuffle(idx1)
        # random. shuffle(idx2)
        # random. shuffle(idx3)
        # idx = idx1 + idx2 + idx3
        # # pdb.set_trace()
        
        # if self.type_path == 'train':
        #     self.img_data =  [images[i] for i in idx]  #  [idx]
        #     self.source_data = [source_data[i] for i in idx] 
        #     self.target_text = [targets[i] for i in idx]  
        #     self.out_bboxes_list = [out_bboxes_list[i] for i in idx] 
        #     self.out_bboxes_label_list = [out_bboxes_label_list[i] for i in idx] 
        #     self.fake_cls = [fake_cls[i] for i in idx] 
        #     self.text_cls = [text_cls[i] for i in idx] 
        #     self.fake_m_label = [fake_m_label[i] for i in idx] 
        # else:
        if True:
            self.img_data = images # add dir, img path
            self.source_data = source_data # add prompt with provided text
            self.target_text = targets # txt tokens
            self.out_bboxes_list = out_bboxes_list
            self.out_bboxes_label_list = out_bboxes_label_list
            self.fake_cls = fake_cls
            self.text_cls = text_cls
            self.fake_m_label = fake_m_label
          
        assert len(self.img_data) == len(targets)




class MSTIDataSet_SC(Dataset):
    def __init__(self, type_path, tokenizer = None, max_examples=-1,
                 max_src_len=512, max_tgt_len=200):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        self.max_examples = max_examples
        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len
        print('loading',type_path,'set')
        self.type_path = type_path
        assert type_path in ['train','test','val']
        self._build()       # fill inputs, targets, max_lens
        print('done loading',type_path,'set with length of',len(self.img_data))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        src_text = self.source_data[index]
        tgt_text = self.target_text[index]
        image_id = self.img_data[index]
        out_bboxes1 = self.out_bboxes_list[index]
        out_labels1 = self.out_bboxes_label_list[index]
        fake_cls = self.fake_cls[index]
        text_cls = self.text_cls[index]

        return {"source_text": src_text, # 经过prompt以后的输入
                 "target_text": tgt_text, # 出现错误的toekn
                 "image_ids": image_id, # 图片的存储位置
                 "bboxes": out_bboxes1, # bbox.size=(10,) 没有bbox就是全零
                 "box_labels":out_labels1,
                 "text_cls":text_cls,
                 "fake_cls":fake_cls} # 有错是1，没错是0

    def _build(self):

        print("buliding ",self.type_path," dataset...")
        dgm4_dir = '/data1/zc/dataset/'

        source_data = []
        targets = []
        images = []
        fake_cls = []
        out_bboxes_list = []
        out_bboxes_label_list = []
        text_cls = []
        json_file = os.path.join(dgm4_dir+'DGM4/metedata/', self.type_path+'_sa.json')
        json_file  = '/data1/zc/dataset/DGM4/metedata/test_sa1.json'
        # if self.type_path == 'test':
        #     json_file = '/data1/zc/dataset/DGM4/metedata/test_sa1.json'

        with open(json_file,'r') as f:
            data = json.load(f)

        for line in data:
            images.append(os.path.join(dgm4_dir, line['image']))
            text = prompt_stage2_u.format(line['text'])
            source_data.append(text)
            fake_cls.append(line['fake_cls'])
            
        
            target = np.array(['T'] * len(line['text'].split()))
            # pdb.set_trace()
            target[np.array(line['fake_text_pos']) == 1] = 'O'
           
            # if line['fake_text_pos'] != []:
            #     for text_index in line['fake_text_pos']:
            #         target[text_index-1] = 'O'
            # pdb.set_trace()
            target = ' '.join(target.tolist())
            targets.append(target)

            max_len = 10
            out_bboxes1 = np.zeros([max_len, 4])
            out_labels1 = np.zeros([10])
            if len(line['fake_image_box']) != 0:
                box_iter = np.array(line['fake_image_box']).shape[0]
                if box_iter > max_len:
                    box_iter = 10
                for ith in range(box_iter):                    
                    out_bboxes1[ith] = np.array(line['fake_image_box'][ith])                    
                    out_labels1[ith] = 1
                   
            out_bboxes_list.append(out_bboxes1)
            out_bboxes_label_list.append(out_labels1)

            if 1 in fake_cls:
                text_cls.append(1)
            else:
                text_cls.append(0)
            
             
        self.img_data = images # add dir, img path
        self.source_data = source_data # add prompt with provided text
        self.target_text = targets # txt tokens
        self.out_bboxes_list = out_bboxes_list
        self.out_bboxes_label_list = out_bboxes_label_list
        self.fake_cls = fake_cls
        self.text_cls = text_cls
        
          
        assert len(self.img_data) == len(targets)

def load_images(image_path):
    image_input_ids = []
    image_size = 600
    transform = Compose([
        RandomResize([image_size], max_size=image_size),
        Padding(max_x=image_size,max_y=image_size),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for img_p in image_path:
        image = Image.open(img_p).convert("RGB") # load image
        # transform images
        image, _ = transform(image, None)
        image_input_ids.append(image)
    return image_input_ids

def load_images_with_source(image_path: list) -> Tuple[np.array, torch.Tensor]:
    image_input_ids = []    # keep a record of source image
    images = []

    transform = Compose(
        [
            # RandomResize([800], max_size=800),
            # Padding(max_x=800,max_y=800),
            RandomResize([600], max_size=600),
            Padding(max_x=600,max_y=600),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    for img_p in image_path:
        source_image = Image.open(img_p).convert("RGB") # load image
        image = np.asarray(source_image)
        image_transformed, _ = transform(source_image, None)
        image_input_ids.append(image_transformed)
        images.append(image)

    return images, image_input_ids

def load_image(image_path):
    # image_input_ids = []
    transform = Compose([
        Resize([800], max_size=1333),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB") # load image
    # transform images
    image, _ = transform(image, None)
    image_input_ids=image
    return image_input_ids


def bbox_transform(center_x, center_y, width, height):

    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    return x1,y1,x2,y2

def bbox_detransform(x1, y1, x2, y2):
    """
    xyxy to xywh
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    width = x2 - x1
    height = y2 - y1

    return center_x, center_y, width, height


def get_dataloaders(args, tokenizer, batch_size, num_train = -1, num_val = -1, num_workers = 4, shuffle_train=True,
                    shuffle_dev=False, SC=False) -> Tuple[DataLoader, DataLoader]:
    """
    Returns: Tuple[train_loader : DataLoader, dev_loader : DataLoader]
    # Note:
    # - we default to not shuffling the dev set

    """
    # todo: should pass max src and max tgt len in as arguments
    if args.stage == "stage1":
        if args.model_type == 't5': # zc marked
            train_data_set = T5DataSet("train", tokenizer, max_examples=num_train,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
            eval_data_set = T5DataSet("val", tokenizer, max_examples=num_val,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
            test_data_set = T5DataSet("test", tokenizer, max_examples=num_val,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    else:
        if SC:
            train_data_set = MSTIDataSet_SC('train')
            eval_data_set = MSTIDataSet_SC('val')
            test_data_set = MSTIDataSet_SC('test')
        else:
            train_data_set = MSTIDataSet_DGM4('train')
            eval_data_set = MSTIDataSet_DGM4('val')
            test_data_set = MSTIDataSet_DGM4('test')

        # train_data_set = MSTIDataSet('train')
        # eval_data_set = MSTIDataSet('val')
        # test_data_set = MSTIDataSet('test')

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_loader = DataLoader(eval_data_set, batch_size=batch_size, shuffle=shuffle_dev, num_workers=num_workers)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle_dev, num_workers=num_workers)
    # log.info(f'Datasets loaded with sizes: train: {len(train_data_set)}, dev: {len(eval_data_set)}')

    return train_loader, eval_loader, test_loader



def forward_u_ln(wrapped , tokenzier, device, batch, task_id):
    # tokenzier =

    src_out = tokenzier(batch["source_text"], max_length=128, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)


    # if model.model_type == 't5':
    tgt_out = tokenzier(batch["target_text"], max_length=200, padding="max_length", return_tensors="pt", truncation=True)
    tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
    tgt_ids[tgt_ids[: ,:] == 0 ] = -100



    # if model.args.stage == "stage2":
    source_img, img_ids = load_images_with_source(batch["image_ids"])
    img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
    tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

    for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
        h, w, _ = src_img.shape
        max_len = max(h,w)
        for j in range(len(boxes)):
            tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)
    # else:
    #     img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)

    label_ids = tgt_ids.to(device)
    # pdb.set_trace()
    out_dict, img_out = wrapped.forward_task(image = img_ids, input_ids=src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True, src_out = src_out, batch=batch, task_id=task_id)
        
    #     # task_id, batch["images"], batch["captions"])
    # model/
    
    # if model.args.stage == "stage2":
        # add bbox loss
        # resize pred bboxes back to original sizes for loss calculation
    pred_boxes = img_out['pred_boxes']#[:,:10,:]
    for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
        h, w, _ = src_img.shape
        max_len = max(h,w)
        for j in range(len(boxes)):
            pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
    img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)

    loss_cls, loss_bbox, loss_iou = loss_by_feat_single_zc(bbox_preds=img_out['pred_boxes'],       # [16,2,4]
                                                        cls_scores=img_out['pred_logits'][:,:1,:],       # [16,2,128] -> [16,1,128]? why 128
                                                        text_masks=src_mask,                       # [16,128]
                                                        box_labels=batch['box_labels'][:,:1],      # [16, 10] -> [16, 1]
                                                        target_boxes=batch['bboxes'][:,:1,:].cuda())       # we only take one target    # [8, 1, 4]
    # img_loss =  loss_cls*0.1 + loss_bbox*1e-3 + loss_iou*0.2       # to be adjusted
    # pdb.set_trace()
    img_loss =  loss_cls*0.1  + loss_iou*0.2       # to be adjusted
    if task_id == 0:
        out_dict['loss'] = out_dict['loss'] + img_loss*5 + out_dict['loss_txt'] + out_dict['loss_bi']
    else:
        out_dict['loss'] = out_dict['loss'] + img_loss*5 + out_dict['loss_txt'] + out_dict['loss_bi']+ out_dict['loss_multi']
    # pdb.set_trace()
      
    loss = out_dict['loss']

    return loss, None



def forward_u(model, device, batch, task_id):
    tokenzier = model.tokenizer

    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)


    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100



    if model.args.stage == "stage2":
        source_img, img_ids = load_images_with_source(batch["image_ids"])
        img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
        tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

        for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)
    else:
        img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)

    label_ids = tgt_ids.to(device)
    out_dict, img_out = model.forward_train_cls(image = img_ids, input_ids=src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True, src_out = src_out, batch=batch, task_id=task_id)
    
    
    if model.args.stage == "stage2":
        # add bbox loss
        # resize pred bboxes back to original sizes for loss calculation
        pred_boxes = img_out['pred_boxes']#[:,:10,:]
        for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
        img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)

        loss_cls, loss_bbox, loss_iou = loss_by_feat_single_zc(bbox_preds=img_out['pred_boxes'],       # [16,2,4]
                                                            cls_scores=img_out['pred_logits'][:,:1,:],       # [16,2,128] -> [16,1,128]? why 128
                                                            text_masks=src_mask,                       # [16,128]
                                                            box_labels=batch['box_labels'][:,:1],      # [16, 10] -> [16, 1]
                                                            target_boxes=batch['bboxes'][:,:1,:].cuda())       # we only take one target    # [8, 1, 4]
        # img_loss =  loss_cls*0.1 + loss_bbox*1e-3 + loss_iou*0.2       # to be adjusted
        img_loss =  loss_cls*0.1  + loss_iou*0.2       # to be adjusted
        if task_id == 0:
            out_dict['loss'] = out_dict['loss'] + img_loss*5 + out_dict['loss_txt'] + out_dict['loss_bi']
        else:
            out_dict['loss'] = out_dict['loss'] + img_loss*5 + out_dict['loss_txt'] + out_dict['loss_multi']+ out_dict['loss_bi']
        # pdb.set_trace()
      
    loss = out_dict['loss']

    return loss, None


def forward(model, device, batch):
    tokenzier = model.tokenizer

    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)


    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100



    if model.args.stage == "stage2":
        source_img, img_ids = load_images_with_source(batch["image_ids"])
        img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
        tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

        for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)
    else:
        img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)

    label_ids = tgt_ids.to(device)
    # label_cls_ids = tgt_cls_ids.to(device)
    # out_dict,img_out = model.forward_train(image = img_ids, input_ids=src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True,src_out = src_out)
    out_dict, img_out = model.forward_train_cls(image = img_ids, input_ids=src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True, src_out = src_out, batch=batch)
    # out_dict: 'loss', 'logits'      , 'past_key_values', 'encoder_last_hidden_state'
    #            26   ,[8, 200, 32128],  len=12          , [8, 512, 768]
    # img_out: 'pred_logits', 'pred_boxes'],

    
    if model.args.stage == "stage2":
        # add bbox loss
        # resize pred bboxes back to original sizes for loss calculation
        pred_boxes = img_out['pred_boxes']#[:,:10,:]
        for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
        img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)

        # pdb.set_trace()
        loss_cls, loss_bbox, loss_iou = loss_by_feat_single_zc(bbox_preds=img_out['pred_boxes'],       # [16,2,4]
                                                            cls_scores=img_out['pred_logits'][:,:1,:],       # [16,2,128] -> [16,1,128]? why 128
                                                            text_masks=src_mask,                       # [16,128]
                                                            box_labels=batch['box_labels'][:,:1],      # [16, 10] -> [16, 1]
                                                            target_boxes=batch['bboxes'][:,:1,:].cuda())       # we only take one target    # [8, 1, 4]
        img_loss =  loss_cls*0.1 + loss_bbox*1e-3 + loss_iou*0.2       # to be adjusted
        # pdb.set_trace()
        out_dict['loss'] = out_dict['loss'] + img_loss*5 + out_dict['loss_txt'] + out_dict['loss_multi']+ out_dict['loss_bi']
        # df
        # out_dict['loss'] = out_dict['loss']*0.5 + img_loss*5 + out_dict['loss_multi1']*10
      
    # # # st
    # out_dict['loss'] = out_dict['loss_txt'] + out_dict['loss_multi2']*10
    # # pdb.set_trace()
      
    loss = out_dict['loss']

    return loss, None

def forward_test(model,device, batch):
    tokenzier = model.tokenizer

    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)

    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100

    else:
        tgt_ids = batch["target_text"].to(device, dtype=torch.float)

    if model.args.stage == "stage2":
        source_img, img_ids = load_images_with_source(batch["image_ids"])
        img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
        tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

        for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)
        label_ids = tgt_ids.to(device)
        _,img_out, logits_txt, logits_bi = model(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out,labels=label_ids)
        img_out['pred_boxes'] = img_out['pred_boxes'][:,:1,:]

    else:
        img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)
        generated_ids,img_out = model(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out)

    return source_img,tgt_ids, img_out, logits_txt, logits_bi
    #     generated_ids,img_out = model(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out)

    #     # resize back to original sizes
    #     pred_boxes = img_out['pred_boxes']#[:,:10,:]
    #     for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
    #         h, w, _ = src_img.shape
    #         max_len = max(h,w)
    #         for j in range(len(boxes)):
    #             pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
    #     img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)         # model outputs xywh, xyxy needed for loss and val

    #     # if img_out['pred_logits'].shape[1]>1:
    #     #     img_out['pred_boxes'] = select_boxes(img_out)
    #     img_out['pred_boxes'] = img_out['pred_boxes'][:,:1,:]

    # else:
    #     img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)
    #     generated_ids,img_out = model(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out)

    # return generated_ids,tgt_ids, img_out


def forward_mmsd(model, device, batch):
    tokenzier = model.tokenizer
    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)


    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100

    img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)
    label_ids = tgt_ids.to(device)
    out_dict, img_out = model.forward_train_cls(image = img_ids, input_ids=src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True, src_out = src_out, batch=batch)
    # print(out_dict['loss'],out_dict['loss_bi'])
    # pdb.set_trace()
    loss = out_dict['loss']  + out_dict['loss_bi']*10
       
    return loss, None

def forward_test_cls(model, tokenzier,device, batch):
    
    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)

    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100

        # tgt_out_cls = tokenzier(batch["text_cls"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        # tgt_cls_ids = tgt_out_cls["input_ids"].to(device, dtype=torch.long)
        # tgt_cls_ids[tgt_cls_ids[: ,:] == 0 ] = -100
    else:
        tgt_ids = batch["target_text"].to(device, dtype=torch.float)

    # if model.args.stage == "stage2":
    source_img, img_ids = load_images_with_source(batch["image_ids"])
    img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
    # label_cls_ids = tgt_cls_ids.to(device)
    tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

    for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
        h, w, _ = src_img.shape
        max_len = max(h,w)
        for j in range(len(boxes)):
            tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)

    logits, img_out, logits_multi, logits_bi = model.forward_cls(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out, labels = tgt_ids.to(device), batch=batch)#), labels_cls=label_cls_ids)
  
    # resize back to original sizes
    pred_boxes = img_out['pred_boxes']#[:,:10,:]
    for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
        h, w, _ = src_img.shape
        max_len = max(h,w)
        for j in range(len(boxes)):
            pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
    img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)         # model outputs xywh, xyxy needed for loss and val

    # if img_out['pred_logits'].shape[1]>1:
    #     img_out['pred_boxes'] = select_boxes(img_out)
    # pdb.set_trace()
    img_out['pred_boxes'] = img_out['pred_boxes'][:,:1,:]

    return logits, tgt_ids, img_out, logits_multi, logits_bi


def forward_test_cls_mmsd(model, tokenzier,device, batch):
    
    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)

    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100        
    else:
        tgt_ids = batch["target_text"].to(device, dtype=torch.float)

    # if model.args.stage == "stage2":
    source_img, img_ids = load_images_with_source(batch["image_ids"])
    img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
  
   
    logits, img_out, logits_multi, logits_bi = model.forward_cls(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out, labels = tgt_ids.to(device), batch=batch)#), labels_cls=label_cls_ids)
  
  
    return logits, tgt_ids, img_out, logits_multi, logits_bi



def forward_test_cls_ln(wrapped, tokenzier,device, batch):
    
    src_out = tokenzier(batch["source_text"], max_length=128, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)

    # if model.model_type == 't5':
    tgt_out = tokenzier(batch["target_text"], max_length=200, padding="max_length", return_tensors="pt", truncation=True)
    tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
    tgt_ids[tgt_ids[: ,:] == 0 ] = -100
    
    # if model.args.stage == "stage2":
    source_img, img_ids = load_images_with_source(batch["image_ids"])
    img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
    # label_cls_ids = tgt_cls_ids.to(device)
    tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

    for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
        h, w, _ = src_img.shape
        max_len = max(h,w)
        for j in range(len(boxes)):
            tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)

    logits, img_out, logits_multi, logits_bi =  wrapped.forward_cls(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out, labels = tgt_ids.to(device), batch=batch)#), labels_cls=label_cls_ids)
  
    # resize back to original sizes
    pred_boxes = img_out['pred_boxes']#[:,:10,:]
    for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
        h, w, _ = src_img.shape
        max_len = max(h,w)
        for j in range(len(boxes)):
            pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
    img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)         # model outputs xywh, xyxy needed for loss and val

    # if img_out['pred_logits'].shape[1]>1:
    #     img_out['pred_boxes'] = select_boxes(img_out)
    # pdb.set_trace()
    img_out['pred_boxes'] = img_out['pred_boxes'][:,:1,:]

    return logits, tgt_ids, img_out, logits_multi, logits_bi


def forward_test_cls_1113(model,device, batch):
    tokenzier = model.tokenizer
    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)
    tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
    tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
    tgt_ids[tgt_ids[: ,:] == 0 ] = -100


    source_img, img_ids = load_images_with_source(batch["image_ids"])
    img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
    tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

    for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
        h, w, _ = src_img.shape
        max_len = max(h,w)
        for j in range(len(boxes)):
            tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)

    logits, img_out, text_output, _ = model.forward_cls(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out, labels = tgt_ids.to(device))
    # img_out: pred_logits=[8,2,512], pred_box=[8,1,4]
    # resize back to original sizes
    pred_boxes = img_out['pred_boxes']#[:,:10,:]
    for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
        h, w, _ = src_img.shape
        max_len = max(h,w)
        for j in range(len(boxes)):
            pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
    img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)         # model outputs xywh, xyxy needed for loss and val




    # loss_cls, loss_bbox, loss_iou = loss_by_feat_single(bbox_preds=img_out['pred_boxes'], cls_scores=img_out['pred_logits'], text_masks=src_mask,
                                                        # box_labels=batch['box_labels'][:,:1], target_boxes=batch['bboxes'][:,:1,:].cuda())
    # we only take one target   [8,2,4],  [8,2,512], [8,512], [8, 1], [8, 1, 4]


    img_out['pred_boxes'] = img_out['pred_boxes'][:,:1,:]


    # pred_img = img_out['pred_img'].argmax(-1)
    multi_label = torch.zeros((len(source_img), 4))
    for ith, text_cls in enumerate(zip(text_output)):
        if 'swap' in text_cls:
            multi_label[ith, 2] = 1
        elif 'attribute' in text_cls:
            multi_label[ith, 3] = 1
        # if pred_img[ith] != 0:
        #     multi_label[ith, pred_img[ith]-1] = 1

    # pdb.set_trace()
    return logits,tgt_ids, img_out, multi_label


def select_boxes(img_out):
    index = img_out['pred_logits'][:,:,0].argmax(dim=-1)
    pred_select = []
    for i,label in enumerate(index):
        t = img_out['pred_boxes'][i,label,:]
        pred_select.append(t)
    return torch.stack(pred_select).unsqueeze(dim=1)


def batch_post_process_dgm4(souce_txet, labels, labels_pred):
    preds = []
    gts = []
    ems = []
    accs = []
    for txt, lab, lab_pred in zip(souce_txet, labels, labels_pred):

        txt_list = np.array(txt.split(' '))
        pred = [False]*len(txt_list)
        gt = [False]*len(txt_list)


        # pred = (lab_pred.split(' ') != 'O')
        # gt = (lab.split(' ') != 'O')
        lab_split =lab.split()
        for ith in range(min(len(lab_split), len(txt_list))):
            if lab_split[ith] == 'O':
                gt[ith] = True

        lab_pred_split =lab_pred.split()
        for ith in range(min(len(lab_pred_split),len(txt_list))):
            if lab_pred_split[ith] == 'O':
                pred[ith] = True
        print(pred)
        pdb.set_trace()
        exact_match = [a == b for (a, b) in zip(pred, gt)]
        ems.append(sum(exact_match) // len(exact_match))
        accs += exact_match
        preds += pred
        gts += gt
    batch_acc = np.mean(accs)
    batch_EM = np.mean(ems)

    return preds, gts, batch_acc, batch_EM



def forward_s1(model, device, batch):
    tokenzier = model.tokenizer

    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)

    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100
    else:
        tgt_ids = batch["target_text"].to(device, dtype=torch.float)

    img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)

    label_ids = tgt_ids.to(device)
    out_dict,img_out = model.forward_train(image = img_ids, input_ids=src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True,src_out = src_out)

    loss, logits = out_dict['loss'], out_dict['logits']

  
    return loss, logits
