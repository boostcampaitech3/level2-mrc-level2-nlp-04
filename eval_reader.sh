# dev 데이터셋에서 단순히 Reader 로 답 추출하기 
# Reader 성능 검증용
# 직접 finetune 한 모델의 결과가 있다면 --model_name_or_path 에 불러주기

python train.py \
--model_name_or_path="./finetune/"