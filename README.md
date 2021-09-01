# Dependency
- Pytorch v1.7
- Huggingface transformers v4.3.0
- Huggingface accelerate v0.4.0
# Pretrain Bert
The scripts are in the folder of **pretrain_bert**.

First create the output_dir, where the trained model and log files will be saved.

Then run `bash train.sh`
```
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 \
main.py \
--output_dir /efs-storage/results/untitled/output \
--train_file ../alldata/train.json \
--validation_file ../alldata/valid.json \
--test_file ../alldata/test.json \
--model_name_or_path /efs-storage/bert_base_uncased/ \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--do_train \
--num_train_epochs 15 \
--learning_rate 2e-5 \
--logging_steps 100 \
--save_steps 10000 \
--dataloader_num_workers 4 \
--evaluation_strategy epoch \
--overwrite_output_dir \
--logging_dir /efs-storage/results/untitled/logs \
--output_file /efs-storage/results/untitled/models/ \
--max_seq_length 128 \
> >(tee -a /efs-storage/results/untitled/stdout.log) \
2> >(tee -a /efs-storage/results/untitled/stderr.log >&2)
```
# Data Preprocessing
The origin data is in the form of .json. There are three origin files: train.json, valid.json, and test.json

1. Generate history_embedding.npy, incoming_comment_embedding.npy, and accumulated_history_embedding.npy from pre-trained Bert for training, validation, and test, separately.
history_embedding.npy is Bert embedding for each sequence with pre-pend label.
incoming_comment_embedding.npy is Bert embedding for each sequence without label.
accumulated_history_embedding.npy initialization is mean of past activities Bert embedding.
The scripts are in the folder of **incremental_learning_preprocess** 
Run the following block three times, each time input train, valid, or test file.
```
python generate_embedding.py \
--model_path indicate_pretrained_bert_path \
--file_path train.json \
--save_path_incoming_comment indicate_path_save_incoming_comment_embedding_array_end_with.npy \
--save_path_history_embedding indicate_path_save_history_embedding_array_end_with.npy \
--save_path_accumulated_history_embedding indicate_path_save_accumulated_history_embedding_array_end_with.npy
```
2. Generate DataSet with (user_id, t, label) in order to load into DataLoader during training.
Run once to generate three files for training, validation, and test.
```
python generate_tuple.py \
--train_file_path indicate_original_train_json_file_path \
--valid_file_path indicate_original_valid_json_file_path \
--test_file_path indicate_original_test_json_file_path \
--train_save_path indicate_generated_train_file_path \
--valid_save_path indicate_generated_valid_file_path \
--test_save_path indicate_generated_test_file_path
```
# Training
The scripts are in the folder of incremental_learning.
Just crete the output_dir and then run `bash train.sh`.
```
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --use_env \
main.py \
--output_dir /efs-storage/results4/multi_step_changeloss/output \
--train_history_embedding ../data_chronological_v2/history_embedding_train.npy \
--train_user_profile ../data_chronological_v2/user_profile_train.npy \
--train_incoming_comment ../data_chronological_v2/incoming_comment_train.npy \
--train_tuple ../multi_step_data_v2/tuple_train.json \
--eval_history_embedding ../data_chronological_v2/history_embedding_valid.npy \
--eval_user_profile ../data_chronological_v2/user_profile_valid.npy \
--eval_incoming_comment ../data_chronological_v2/incoming_comment_valid.npy \
--eval_tuple ../multi_step_data_v2/tuple_valid.json \
--test_history_embedding ../data_chronological_v2/history_embedding_test.npy \
--test_user_profile ../data_chronological_v2/user_profile_test.npy \
--test_incoming_comment ../data_chronological_v2/incoming_comment_test.npy \
--test_tuple ../multi_step_data_v2/tuple_test.json \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--num_train_epochs 10 \
--learning_rate 2e-5 \
> >(tee -a /efs-storage/results4/multi_step_changeloss1/stdout.log) \
2> >(tee -a /efs-storage/results4/multi_step_changeloss1/stderr.log >&2)
```
