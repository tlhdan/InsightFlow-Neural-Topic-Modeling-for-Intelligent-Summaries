import os
import re
import torch
import pickle
import logging
import time
from models import BATM
from utils import *
from dataset import DocDataset
from multiprocessing import cpu_count

# Dùng biến thay vì argparse để chạy trên Colab
class Args:
    taskname = 'EMNLP2020'
    no_below = 5
    no_above = 0.005
    num_epochs = 100
    n_topic = 20
    bkpt_continue = False
    use_tfidf = False
    rebuild = True
    dist = 'gmm_std'
    batch_size = 512
    criterion = 'cross_entropy'
    auto_adj = False
    lang = 'zh'

args = Args()

def main():
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    n_cpu = max(cpu_count() - 2, 2)
    bkpt_continue = args.bkpt_continue
    use_tfidf = args.use_tfidf
    rebuild = args.rebuild
    dist = args.dist
    batch_size = args.batch_size
    criterion = args.criterion
    auto_adj = args.auto_adj
    lang = args.lang

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # docSet = DocDataset(taskname, lang=lang, no_below=no_below, no_above=no_above, rebuild=rebuild, use_tfidf=use_tfidf)
    docSet = DocDataset('EMNLP2020', rebuild=True)

    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname, lang=lang, no_below=no_below, no_above=no_above, rebuild=rebuild, use_tfidf=False)

    voc_size = docSet.vocabsize
    model = BATM(bow_dim=voc_size, n_topic=n_topic, device=device, taskname=taskname)
    # print(BATM)
    model.train(train_data=docSet, batch_size=batch_size, test_data=docSet, num_epochs=num_epochs, log_every=10, n_critic=10)
    model.evaluate(test_data=docSet)

    save_name = f'./BATM_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
    torch.save({'generator': model.generator.state_dict(),
                'encoder': model.encoder.state_dict(),
                'discriminator': model.discriminator.state_dict()}, save_name)
    print(f'Model saved to {save_name}')

if __name__ == "__main__":
    main()