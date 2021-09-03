# Dependency
- Pytorch v1.7
- Huggingface transformers v4.3.0
- Huggingface accelerate v0.4.0
# Pretrain Bert
The scripts are in the folder of **pretrain_bert**.
We sample a few tokens in each sequence for MLM training (with probability 15%). 
80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK]).
10% of the time, we replace masked input tokens with random word.
The rest of the time (10% of the time) we keep the masked input tokens unchanged.
The most important part is how we modify and write the collate function defined in the Pytorch DataLoader. 
Inside the `collate_fn`, we add mask to tokens in the way as we mentioned above.

To excute the scripts, first create the output_dir, where the trained model and log files will be saved.
Indicate the train.json path, which contains 5M comments and their corresponding labels. Also indicate valid.json path, which will be used for validation.
Define the hyparameters such as batch size, learning rate, epoch, max seq length, etc. in the `train.sh` file

Then run `bash train.sh`. A sample of this shell file is as below. We have to define the output_dir, input_dir, and model parameters.
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
--num_train_epochs 5 \
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

1. The goal of this section is to generate history_embedding.npy, incoming_comment_embedding.npy, and accumulated_history_embedding.npy from pre-trained Bert for training, validation, and test, separately.
If we do not want to incorporate history label information by pre-pending label in front of history sequence during encoding, then history_embedding.npy and incoming_comment_embedding.npy are the same thing.
history_embedding.npy is Bert embedding array of all sequence by pre-pending sequence label in front of it. It is in shape of n$\times$T*d.
incoming_comment_embedding.npy is Bert embedding array of all sequence without history label information. It is in shape of n*T*d.
accumulated_history_embedding.npy initialization is mean of past activities Bert embeddings. It is in shape of n*T*d.

Note: the sequences should be in the reverse chronological order before input into Bert for encoding because there is one following line inside the scirpt
```
np.flip(incoming_comment_embedding, axis=1)
```
This line flip the encoded array in the seconde dimension (the dimension on T) and thus make the array in chronological order. So, if the input is in the chronological order, then you have to comment the above line inside the script to make sure it is correct.

The scripts are in the folder of **incremental_learning_preprocess** 
Run the following block three times, each time input train, valid, or test file.
The last three path should end with .npy because they save the numpy arrays. 
```
python generate_embedding.py \
--model_path $pretrained_bert_path \
--file_path ${train, valid, test}_json_dir \
--save_path_incoming_comment ${train, valid, test}_incoming_comment_embedding.npy \
--save_path_history_embedding ${train, valid, test}_history_embedding.npy \
--save_path_accumulated_history_embedding ${train, valid, test}_accumulated_history_embedding.npy
```
After running this, we will have nine numpy arrays. Each one of training, validation, and test set has three corresponding numpy arrays. For example, there are train_history_embedding.npy, train_accumulated_history_embedding.npy, and train_incoming_comments.npy. These arrays will be called during training.

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
