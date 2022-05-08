import logging
import os
import sys
from typing import NoReturn

from pyparsing import col

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # (study) Using HfArgumentParser we can turn this class into argparse arguments that can be specified on the command line
    # (study) TrainArguments를 제외한 나머지 Arguments 클래스들은 전부 arguments.py에서 만들어진 클래스이다
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    # (study) the dataclass instances in the same order as they were passed to the initializer.abspath
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)

    # (study) default = {model_args.model_name_or_path: "klue/bert-base"
    #                    data_args.dataset_name : "../data/train_dataset"}
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    # (study) 참고 : https://www.daleseo.com/python-logging/, https://www.delftstack.com/ko/howto/python/python-logging-stdout/
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    # (study) default = {training_args.seed: 42}
    set_seed(training_args.seed)

    # (study) 데이터셋을 로드합니다.
    # (study) default = {data_args.dataset_name : "../data/train_dataset"}
    # (study) load_from_disk 반환 타입 : ``datasets.Dataset`` or ``datasets.DatasetDict``
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.

    # (study) default = {model_args.config_name : None,
    #                    model_args.model_name_or_path : "klue/bert-base",
    #                    model_args.tokenizer_name : None }
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        # (study) 'from_tf' : (bool, optional, defaults to False) — Load the model weights from a TensorFlow checkpoint save file
        # (study) '.ckpt' : 텐서플로우에서 학습된 모델의 구조를 제외한 변수들을 담고 있는 파일입니다.
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    # (study) default = {training_args.do_train : False,
    #                    training_args.do_eval : None}
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    # (study) do_train or do_eval 둘 중 하나가 True의 형태이므로 if-else 문으로 구분된다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    # (study) column_names:  ['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__']
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.

    # (study) The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’]. Default value is picked from the class attribute of the same name.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    # (study) 모델의 정보와 헤당 모델이 받아들일 수 있는 데이터의 최대 길이를 올바르게 체크해서 반환해줌
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Train preprocessing / 전처리를 진행합니다.
    def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        # (study) pad_on_right가 True라면 qustion|context순서로 토크나이징이 진행됨. False면 반대로 진행됨.
        # (study) truncation을 하더라도 question이 아닌 context만을 자를 수 있도록 함.
        # (study) defalut = {data_args.doc_stride : 128}
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            # (study) return_overflowing_tokens : 길이가 넘어가는 토큰들을 반환할 것인지의 여부
            # (study) return_offsets_mapping : 각 토큰에 대해 (char_start, end_start) 정보를 반환할 것인지의 여부
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            # (study) default = {data_args.pad_to_max_length : False}
            # (study) question이랑 context를 합쳐서 넣다보니 max_seq_length보다 길이가 작은 경우가 있을까 싶음
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            # (study) input_ids의 타입은 리스트 형태임. index 메서드로 접근하면 왼쪽에서 가장 첫 번재로 만나는 값의 index 위치를 반환함
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            # (study) 두 개의 문장을 연결한 경우, 각각의 문장에 서로 다른 번호를 부여하는 경우에 해당함.
            # (study) token_type_ids 와 거의 유사함. special_token만 (None) 으로 처리되었을 뿐
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            # (study) max_seq_length를 넘어 하나의 example이 여러 개의 span으로 나뉘어지는 경우가 존재함.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                # (study) 정답이 여러 개가 존재하는 경우가 있음. 그럴 때는 가장 첫 번째의 것을 정답이라 하자.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # (study) 여기에서 하고자 하는 것은 context의 span 구간의 index 위치를 알고 싶은 것임.
                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                # (study) 정답의 위치가 context span 내부에 있는지 없는지에 따라 다르게 처리가 됨.
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # (study) default = {training_args.do_train : False}
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        # (study) 위에서 설정한 prepare_train_feature에 의해 feature들이 전처리가 진행됨.
        # (study) default = {data_args.preprocessing_num_worker : None,
        #                    data_args.overwrite_cache : False}
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        # (study) validation 같은 경우는 결국 평가를 해야 하는데, context가 둘로 쪼개져 있다면 이를 평가하는 데 방해가 될 수 있으니, 이를 대응되는 id에 mapping을 시켜줌
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            # (study) context 부분만 token position을 반환하게 하고 나머지 부분은 None으로 처리되게 함.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # (study) 위에서 tokenizer로 토크나이징을 할 때, padding="max_length" if data_args.pad_to_max_length else False 이걸로 기본값을 False로 넣어줌..
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    # (study) pad_to_multiple_of :  If set will pad the sequence to a multiple of the provided value
    # (study) batch 단위로 묶어서 모델의 입력으로 넣을 때 남는 부분을 pad로 채워주는.
    # (study) default = {trainging_args.fp16 : False}
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        # (study) 최종 결과를 얻고 싶을 때
        if training_args.do_predict:
            return formatted_predictions

        # (study) evaluation을 하는 경우는 단순히 retriever를 통해 불러온 context에서 answer을 찾아내는 과정임.
        # (study) retriever + reader의 성능을 체크할 때
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            # (study) EvalPredicion의 Parameter type : NameTuple
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    # (study) task에 다른 metric 정보도 라이브러리를 통해 불러올 수 있음.
    metric = load_metric("squad")

    # (study) post_processing_function을 통해 나온 반환값을 metric 계산을 실시함.
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    # (study) train_dataset, eval_dataset은 전처리가 진행된 데이터셋이고, dataset['valiation] 의 경우 전처리가 되지 않은 데이터셋이다.
    # (study) QuestionAnsweringTrainer의 경우, trainer_qa.py에서 만들어진 customed 된 trainer 클래스이다.
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        # (study) 참고 : https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.log_metrics
        trainer.log_metrics("train", metrics)
        # (study) 참고 : https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
