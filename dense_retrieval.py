import os
import json
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import concatenate_datasets, load_from_disk
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)
from contextlib import contextmanager

import pyarrow as pa
import pyarrow.dataset as ds

import time
import numpy as np
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from datasets import Dataset


@contextmanager
def timer(name):
    t0 = time.time()

    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


class DenseRetrieval:
    def __init__(self, args, dataset, num_neg, tokenizer, p_encoder, q_encoder):

        """
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        """

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.p_embedding = None

        self.prepare_in_batch_negative(num_neg=num_neg)

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset["context"]])))
        p_with_neg = []

        for c in dataset["context"]:

            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            p_with_neg, padding="max_length", truncation=True, return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, num_neg + 1, max_len
        )

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size,
        )

        valid_seqs = tokenizer(
            dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
            valid_seqs["token_type_ids"],
        )
        self.passage_dataloader = DataLoader(
            passage_dataset, batch_size=self.args.per_device_train_batch_size
        )

    def get_dense_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고, Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        # Pickle을 저장합니다.
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join("./data", pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            p_embs = []
            with torch.no_grad():

                epoch_iterator = tqdm(
                    self.dataloader, desc="Iteration", position=0, leave=True
                )
                self.p_encoder.eval()

                for _, batch in enumerate(epoch_iterator):
                    batch = tuple(t.cuda() for t in batch)

                    p_inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }

                    outputs = self.p_encoder(**p_inputs).to("cpu").numpy()
                    p_embs.extend(outputs)

            self.p_embedding = p_embs

            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)

            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        assert self.p_embedding is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

        # if isinstance(query_or_dataset, str):
        #     doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
        #     print("[Search query]\n", query_or_dataset, "\n")

        #     for i in range(topk):
        #         print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
        #         print(self.contexts[doc_indices[i]])

        #     return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        # if isinstance(query_or_dataset, Dataset):

        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        with timer("query exhaustive search"):
            doc_indices = self.get_relevant_doc(query_or_dataset["question"], k=topk)
        for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": doc_indices[idx],
                "context": " ".join(
                    [self.contexts[pid] for pid in doc_indices.tolist()[idx]]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
        t_total = (
            len(self.train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    p_encoder.train()
                    q_encoder.train()

                    targets = torch.zeros(
                        batch_size
                    ).long()  # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(args.device),
                        "attention_mask": batch[1]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(args.device),
                        "token_type_ids": batch[2]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device),
                    }

                    p_outputs = self.p_encoder(
                        **p_inputs
                    )  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(
                        q_outputs, torch.transpose(p_outputs, 1, 2)
                    ).squeeze()  # (batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors="pt"
            ).to(args.device)
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                p_emb = p_encoder(**p_inputs).to("cpu")
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(
            len(self.passage_dataloader.dataset), -1
        )  # (num_passage, emb_dim)
        self.p_embedding = p_embs

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]


if __name__ == "__main__":

    # 메모리가 부족한 경우 일부만 사용하세요 !
    train_dataset = pd.read_csv("/opt/ml/input/data/aihub.csv")
    train_dataset = train_dataset.loc[
        :, ["id", "title", "context", "question", "answer"]
    ]

    train_dataset = Dataset(pa.Table.from_pandas(train_dataset))

    num_sample = 100
    sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
    train_dataset = train_dataset[sample_idx]
    # ontexts = list(
    #     dict.fromkeys([v["text"] for v in wiki.values()])
    # Test sparse
    # org_dataset = load_from_disk("/opt/ml/input/data/train_dataset")
    # full_ds = concatenate_datasets(
    #     [
    #         org_dataset["train"].flatten_indices(),
    #         org_dataset["validation"].flatten_indices(),
    #     ]
    # )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    # print("*" * 40, "query dataset", "*" * 40)
    # print(full_ds)

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
    )
    model_checkpoint = "klue/bert-base"

    # 혹시 위에서 사용한 encoder가 있다면 주석처리 후 진행해주세요 (CUDA ...)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)

    # Test sparse
    org_dataset = load_from_disk("/opt/ml/input/data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    retriever = DenseRetrieval(
        args=args,
        dataset=train_dataset,
        num_neg=5,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
    )
    retriever.train()
    retriever.get_dense_embedding()

    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds)
        df["correct"] = df["original_context"] == df["context"]
        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )

    # results = retriever.get_relevant_doc(query=f, k=5)

    # print(f"[Search Query] {query}\n")

    # indices = results.tolist()
    # for i, idx in enumerate(indices):
    #     print(f"Top-{i + 1}th Passage (Index {idx})")
    #     pprint(retriever.dataset["context"][idx])
