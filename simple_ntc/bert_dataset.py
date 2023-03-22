import torch
from torch.utils.data import Dataset

class TextClassificationCollator():
    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    # 매번 데이터로더가 미니배치 사이즈가 128이다. 그러면 128번을 데이터셋에 대해서 getitem 호출한 것을 받아왔다.
    # 받아온것을 concat하면 된다. / 그것을 지금 못하니 call_fn을 부른다.
    # samples에 데이터셋이 리턴한게 리스트로 들어있을 것이다.
    # 즉, 딕셔너리에 리스트가 들어있을 것이다.
    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        
        # 토크나이저를 사용한다.
        # __call__이 호출된다. 
        # 토큰갯수 기준으로 미니배치 사이즈는 가변적 대신 미니배치내의 토큰갯수만 바뀜 그러면 메모리는 고정
        # -> 구현 어려움 그래서 미니배치네 가장 긴 기준으로
        encoding = self.tokenizer(
            texts, # text
            padding=True, # 미니배치네 가장 긴 기준으로 패딩을 하기위해 max_length를 getitem에서 안쓰는 거다.
            truncation=True, # max_length 기준으로 잘라냄
            return_tensors="pt", # pytorch type으로 
            max_length=self.max_length 
        )

        return_value = {
            'input_ids' : encoding['input_ids'], # (x,l,1) -> 샘플, 타임스켑, 인덱스
            'attention_mask' : encoding['attention_mask'], # padding된 부분 학습하지 않기 위함
            'labels' : torch.tensor(labels, dtype=torch.long), # 리스트로 있던것을 torch.long 타입의 텐서로 바꿈          
        }
        if self.with_text: # 위 텍스트가 true인 경우에는 return_value['text]에 넣는다.
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels): # 전체 데이터셋(코퍼스), 각 샘플별 레이블을 리스트로 들고옴
        self.texts = texts
        self.labels = labels

    def __len__(self): # 전체 샘플이 몇개인지
        return len(self.texts)

    # 데이터셋을 데이터 데이터 로더에 넣을 건데 필요할때마다 미니배치를만들어서 메 iteration 리턴을 한다.
    # 미니배치가 128이면 128번을 데이터셋에 대해서 getitem을 호출한다.
    # 매번 호출할때마다 idx에 있는 아이템들을 리턴해주면 된다.
    # 문장의 길이가 다 다를것이기 때문에 미니 배치내에 가장 긴 문장을 기준으로 패딩을 채워서 리턴한다.
    def __getitem__(self, item): 
        text = str(self.texts[item])
        label = self.labels[item]

        return { # 길이가 모두 같다면 딕셔너리를 안쓰고 그냥 tensor로 리턴하면된다.
            'text' : text,
            'label' : label,
        }