import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config   


def read_text():
    '''
    Read text from standard input for inference.
    '''
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip()]

    return lines


def main(config):
    
    saved_data = torch.load( # 저장된 모델을 불러옴
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id # 원하는 디바이스에 로딩되도록록
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines = read_text()

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best) # fine-tuning한 파라미터를 로드한다.

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device # 모델의 첫번째 파라미터의 디바이스를 보면 어느 디바이스에 올랐는지 알 수 있음

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size): # 전체에 대해 batch_size만큼 점프하면서 인덱스를 맏아온다.
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size],#lines에서 indx부터 그 다음 batch_size까지 받아옴
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            # Take feed-forward
            # model(x, attention_mask=mask) : (n,1,|c|) or (n,|c|)
            #  F.softmax 확률값 구하기 위함
            # dim = -1를 해야지 |c|에 대해서 softmax를 구한다.
            # 같은 크기지만 각 미니배치별 샘플별 클래스가 들어잇는 확률을 구하게 된다.
            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1) 

            # y_hats에 쌓는다.
            y_hats += [y_hat]
        # Concatenate the mini-batch wise result
        # (n,|c|) X mini_batch 갯수
        # 이것을 다 합쳐야 된다.
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(lines), top_k)

        # 화면에 출력
        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (
                ' '.join([index_to_label[int(indice[i][j])] for j in range(config.top_k)]), 
                lines[i]
            ))


if __name__ == '__main__':
    config = define_argparser()
    
    main(config)