import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

import sys 
sys.path.append('/content/drive/MyDrive/인공지능/텍스트분류')
from simple_ntc.bert_trainer import BertTrainer as Trainer
from simple_ntc.bert_dataset import TextClassificationDataset, TextClassificationCollator
from simple_ntc.utils import read_text

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    # Recommended model list:
    # - kykim/bert-kor-base
    # - kykim/albert-kor-base
    # - beomi/kcbert-base
    # - beomi/kcbert-large
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base')
    p.add_argument('--use_albert', action='store_true')
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--lr', type=float, default=5e-5) # warmup이 끝났을때 lr이다.
    p.add_argument('--warmup_ratio', type=float, default=.2) # 트랜스포머가 학습이 까다로움/그냥 adam쓰면 성능이 잘 안나옴 
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    # If you want to use RAdam, I recommend to use LR=1e-4.
    # Also, you can set warmup_ratio=0.
    p.add_argument('--use_radam', action='store_true') # warmup안하고 하는 방법 연구 이것을 쓸대의 인자는 바로 위에 2개임임
    p.add_argument('--valid_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config


def get_loaders(fn, tokenizer, valid_ratio=.2):
    # Get list of labels and list of texts.
    labels, texts = read_text(fn)

    # Generate label to index map.
    unique_labels = list(set(labels)) # 유니크한 레이블로 만든다.
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels): # 유니크레이블을 돌면서 매핑
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    # 텍스트를 index로 변환해 나온 결과를 적용하면 interger의 리스트가 된다.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    # shuffle을 해서 train과 vali를 나눈다.
    shuffled = list(zip(texts, labels)) # zip해논 상태에서 shuffled 해야한다.
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    # Get dataloaders using given tokenizer as collate_fn.
    # 데이터로더가 나온다. train이니가 shuffle해야한다. val은 안한다.
    train_loader = DataLoader(
        TextClassificationDataset(texts[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader, index_to_label


def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW( # 웬만해서는 default값 사용
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer


def main(config):
    # Get pretrained tokenizer.
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label = get_loaders( # idnex_to_label은 추론할때 필요한 정보보
        config.train_fn,
        tokenizer,
        valid_ratio=config.valid_ratio
    )
    # 몇 개인지 확인인
    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    # warmup
    # adam은 고정 lr이다. 이렇게 하면 transformer가 학습이 잘안됨
    # 그래서 warmup을 한다. adam이 처음부터 잘동작한다.
    # 그러나 처음 들어오는 샘플들이 noise할 수 있다. 그걸로 모멘텀을 잘 못 배워서 날라가버리는 현상발생
    # 초반에 네트워크가 안정되기 전까지 많이 배우지 말고 warmup을 해라는 것이다.    
    n_total_iterations = len(train_loader) * config.n_epochs # 미니배치수 X epoch수로 iteration을 지정
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio) # 400의 20%면 80까지는 warmup한다.
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # 모델 선언
    # Get pretrained model with specified softmax layer.
    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name, # 사전 학습된 weight가 로딩이 됨,
        num_labels=len(index_to_label) # 다만 맨위에 있는 linear layer는 random 초기화되어 있다.
    )
    optimizer = get_optimizer(model, config)

    # By default, model returns a hidden representation before softmax func.
    # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    # 소프트맥스 직전의 hidden_referengentation...값을 loss에 집어넣으면 된다.
    # 그것을 logits이라고 한다. 
    # 그리고 linear 스케줄 warmup
    crit = nn.CrossEntropyLoss()    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    # gpu로 옮김김
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config, # 나중에 불러올 때 어떤 hp인지 알아야함함
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)

# 실행을 하면 여기로 간다.
# hyper parameter를 여기에 입력받게 된다.
if __name__ == '__main__':
    config = define_argparser()
    main(config)