# Dependency
- Pytorch v1.7
- Huggingface transformers v4.3.0
- Huggingface accelerate v0.4.0
# Pretrain Bert
The scripts are in the folder of **pretrain_bert**.
First create the output_dir, where the trained model and log files will be saved.
Then run `bash train.sh`
`python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 \
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
2> >(tee -a /efs-storage/results/untitled/stderr.log >&2)`
# Data Preprocessing
The origin data is in the form of .json. There are three origin files: train.json, valid.json, and test.json
