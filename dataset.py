import os
import re
import numpy as np
import gensim
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from collections import Counter
from tokenization import *

def clean_text(text, lang="vi"):
    text = text.lower()
    # Loại bỏ số và ký tự đặc biệt (giữ _)
    text = re.sub(r'[0-9!"#$%&\'()*+,-./:;<=>?@—，。：★、￥…【】（）《》？“”‘’！\[\\\]^`{|}~\u3000]+', ' ', text)
    # Loại bỏ tên riêng nước ngoài hoặc từ nhiễu
    text = re.sub(r'\b(carlos|hc|xiv|kashmir|sinner|duterte|francis|philippines|vatican)\b', '', text, flags=re.IGNORECASE)
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    return text

class DocDataset(Dataset):
    def __init__(self, taskname, txtPath=None, lang="vi", tokenizer=None, stopwords=None,
                 no_below=5, no_above=0.3, hasLabel=False, rebuild=False, use_tfidf=False):
        self.base_dir = "/content/Neural_Topic_Models/data"
        os.makedirs(self.base_dir, exist_ok=True)

        if txtPath is None:
            txtPath = os.path.join(self.base_dir, f'{taskname}.txt')

        tmpDir = os.path.join(self.base_dir, taskname)

        if not os.path.exists(txtPath):
            raise FileNotFoundError(f"❌ Không tìm thấy tệp: {txtPath}. Hãy kiểm tra đường dẫn!")

        print(f"📂 Đang mở tệp: {txtPath}")

        # Đọc và làm sạch dữ liệu
        with open(txtPath, 'r', encoding='utf-8') as f:
            self.txtLines = [clean_text(line.strip(), lang) for line in f if line.strip()]

        self.dictionary = None
        self.bows, self.docs = None, None
        self.use_tfidf = use_tfidf
        self.tfidf, self.tfidf_model = None, None

        if not os.path.exists(tmpDir):
            os.makedirs(tmpDir)

        if not rebuild and os.path.exists(os.path.join(tmpDir, 'corpus.mm')):
            self.bows = gensim.corpora.MmCorpus(os.path.join(tmpDir, 'corpus.mm'))
            if self.use_tfidf:
                self.tfidf = gensim.corpora.MmCorpus(os.path.join(tmpDir, 'tfidf.mm'))
            self.dictionary = Dictionary.load_from_text(os.path.join(tmpDir, 'dict.txt'))
            self.docs = pickle.load(open(os.path.join(tmpDir, 'docs.pkl'), 'rb'))
        else:
            if stopwords is None:
                stopwords_path = os.path.join(self.base_dir, 'stopwords.txt')
                if os.path.exists(stopwords_path):
                    stopwords = set([l.strip() for l in open(stopwords_path, 'r', encoding='utf-8')])
                else:
                    stopwords = set()

            print('Tokenizing ...')
            if tokenizer is None:
                tokenizer = globals()[LANG_CLS[lang]](stopwords=stopwords)
            self.docs = tokenizer.tokenize(self.txtLines)
            self.docs = [doc for doc in self.docs if doc]

            self.dictionary = Dictionary(self.docs)
            self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
            self.dictionary.compactify()

            self.bows = [self.dictionary.doc2bow(doc) for doc in self.docs if doc]
            if self.use_tfidf:
                self.tfidf_model = TfidfModel(self.bows)
                self.tfidf = [self.tfidf_model[bow] for bow in self.bows]

            gensim.corpora.MmCorpus.serialize(os.path.join(tmpDir, 'corpus.mm'), self.bows)
            self.dictionary.save_as_text(os.path.join(tmpDir, 'dict.txt'))
            pickle.dump(self.docs, open(os.path.join(tmpDir, 'docs.pkl'), 'wb'))

        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'✅ Đã xử lý {self.numDocs} tài liệu.')


    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt,bow

    def __len__(self):
        return self.numDocs

    def collate_fn(self,batch_data):
        texts,bows = list(zip(*batch_data))
        return texts,torch.stack(bows,dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def show_dfs_topk(self,topk=20):
        ndoc = len(self.docs)
        # print(self.dictionary.id2token[k])
        # dfs_topk = sorted([(self.dictionary.id2token[k],fq) for k,fq in self.dictionary.dfs.items()],key=lambda x: x[1],reverse=True)[:topk]
        dfs_topk = sorted([(self.dictionary.id2token.get(k, f"UNK_{k}"), fq) for k, fq in self.dictionary.dfs.items()],
                   key=lambda x: x[1], reverse=True)[:topk]
        for i,(word,freq) in enumerate(dfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ndoc} = {(1.0*freq/ndoc):>.13f}')
        return dfs_topk

    def show_cfs_topk(self,topk=20):
        ntokens = sum([v for k,v in self.dictionary.cfs.items()])
        cfs_topk = sorted([(self.dictionary.id2token[k],fq) for k,fq in self.dictionary.cfs.items()],key=lambda x: x[1],reverse=True)[:topk]
        for i,(word,freq) in enumerate(cfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ntokens} = {(1.0*freq/ntokens):>.13f}')

    def topk_dfs(self,topk=20):
        ndoc = len(self.docs)
        dfs_topk = self.show_dfs_topk(topk=topk)
        return 1.0*dfs_topk[-1][-1]/ndoc