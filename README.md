# Dependency
- Pytorch v1.7
- Huggingface transformers v4.3.0
- Huggingface accelerate v0.4.0
# Pretrain Bert
The scripts are in the folder of **pretrain_bert**.

First create the output_dir, where the trained model and log files will be saved.
Indicate the train.json path, which contains 5M comments and their corresponding labels. Also indicate valid.json path, which will be used for validation.

Then run `bash train.sh`
```
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 \
main.py \
--output_dir $output_dir \
--train_file $train_json_dir \
--validation_file $valid_json_dir \
--test_file $test_json_dir \
--model_name_or_path $bert_dir \
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
--logging_dir /$logs_dir \
--output_file $save_model \
--max_seq_length 128 \
> >(tee -a $output_dir/stdout.log) \
2> >(tee -a $output_dir/stderr.log >&2)
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
--model_path $pretrained_bert_path \
--file_path ${train, valid, test}_json_dir \
--save_path_incoming_comment ${train, valid, test}_incoming_comment_embedding.npy \
--save_path_history_embedding ${train, valid, test}_history_embedding.npy \
--save_path_accumulated_history_embedding ${train, valid, test}_accumulated_history_embedding.npy
```
2. Generate DataSet in the form of (user_id, t, label) in order to load them into DataLoader during training.
Run once to generate three files for training, validation, and test.
The dataset than is sorted in choronogical order for each user.
```
python generate_tuple.py \
--train_file_path $train_json_dir \
--valid_file_path $valid_json_dir \
--test_file_path $test_json_dir \
--train_save_path $tuple_train_dir.json \
--valid_save_path $tuple_valid_dir.json \
--test_save_path $tuple_test_dir.json
```
# Training
The scripts are in the folder of incremental_learning.
Indicate the path of history_embedding.npy, incoming_comment_embedding.npy, accumulated_history_embedding.npy, tuple_file.json generated from previous two steps for training, validation, and test, separately.
Just crete the output_dir and then run `bash train.sh`.
```
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --use_env \
main.py \
--output_dir $output_dir \
--train_history_embedding $train_history_embedding.npy \
--train_user_profile $train_user_profile.npy \
--train_incoming_comment $train_incoming_comment.npy \
--train_tuple $train_tuple.json \
--eval_history_embedding $valid_history_embedding.npy \
--eval_user_profile $valid_user_profile.npy \
--eval_incoming_comment $valid_incoming_comment.npy \
--eval_tuple $valid_tuple.json \
--test_history_embedding $test_history_embedding.npy \
--test_user_profile $test_user_profile.npy \
--test_incoming_comment $test_incoming_comment.npy \
--test_tuple $test_tuple.json \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--num_train_epochs 10 \
--learning_rate 2e-5 \
> >(tee -a $output_dir/stdout.log) \
2> >(tee -a $output_dir/stderr.log >&2)
```
