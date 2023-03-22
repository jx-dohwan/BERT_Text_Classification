import torch
import torch.nn.utils as torch_utils

from ignite.engine import Events

import sys
sys.path.append('/content/drive/MyDrive/인공지능/텍스트분류')
from simple_ntc.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

from simple_ntc.trainer import Trainer, MyEngine


class EngineForBert(MyEngine):

    def __init__(self, func, model, crit, optimizer, scheduler, config):
        self.scheduler = scheduler

        super().__init__(func, model, crit, optimizer, config)

    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train() # Because we assign model as class variable, we can easily access to it.
        engine.optimizer.zero_grad()

        x, y = mini_batch['input_ids'], mini_batch['labels']
        x, y = x.to(engine.device), y.to(engine.device) # gpu로 옮김김
        mask = mini_batch['attention_mask']
        mask = mask.to(engine.device) # gpu로 옮김김

        x = x[:, :engine.config.max_length] # n.l,1 : ㅣ차원에 대해서 잘라서 슬라이싱 한다.

        # Take feed-forward
        y_hat = engine.model(x, attention_mask=mask).logits # .logits==hidden state==softmax 넣기 직전값, linear layer통과해 차원축소함함
        # y_hat : (n,|c|)

        loss = engine.crit(y_hat, y) #crossentropy를 통과시키면 loss가 나온다.
        loss.backward() # loss를 미분해서 역전파함

        # Calculate accuracy only if 'y' is LongTensor,
        # which means that 'y' is one-hot representation.
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(get_parameter_norm(engine.model.parameters())) # parameter의 L2_norm
        g_norm = float(get_grad_norm(engine.model.parameters())) # gradient의 L2_norm

        # Take a step of gradient descent.
        engine.optimizer.step() # step을 먹여준다.gradient desent하여 한 스텝을 파라미터에 업데이트
        engine.scheduler.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad(): # grad계산할 필요가 없음/ 메모리를 작게 빠르게게
            x, y = mini_batch['input_ids'], mini_batch['labels']
            x, y = x.to(engine.device), y.to(engine.device)
            mask = mini_batch['attention_mask']
            mask = mask.to(engine.device)

            x = x[:, :engine.config.max_length]

            # Take feed-forward
            y_hat = engine.model(x, attention_mask=mask).logits

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }


class BertTrainer(Trainer):

    def __init__(self, config): # 학습을 위한 하이퍼파라미터라 들어있는 config를 가져옴
        self.config = config

    def train( # 학습할때 모델, loss함수, optimizer...를 받아온다.
        self,
        model, crit, optimizer, scheduler,
        train_loader, valid_loader,
    ):
        train_engine = EngineForBert(
            EngineForBert.train,
            model, crit, optimizer, scheduler, self.config
        )
        validation_engine = EngineForBert(
            EngineForBert.validate,
            model, crit, optimizer, scheduler, self.config
        )

        # trainer.py에 선언되어있음/ 현재상태 출력을 위한 것을 등록
        # train_engine과 validation_engine의 현재 상태를 출력력
        EngineForBert.attach( 
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # 학습이 끝나고 validation을 실행하도록 
        # 실행하는 함수를 만들고 train에 등록록
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        # best loss 여부체크 및 모델 저장장
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            EngineForBert.check_best, # function
        )

        # train engine 실행 train_loader를 넣고 몇 epoch를 돌릴것인지 지정
        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        # 다 끝나면 베스트 모델을 불러온다음에 return하면 학습이 종료된다.
        model.load_state_dict(validation_engine.best_model)

        return model