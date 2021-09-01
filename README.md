# Dependency
- Pytorch v1.7
- Huggingface transformers v4.3.0
- Huggingface accelerate v0.4.0
# Pretrain Bert
The scripts are in the folder of **pretrain_bert**.
First create the output_dir, where the trained model and log files will be saved.
Then run `bash train.sh`

# Data Preprocessing
The origin data is in the form of .json. There are three origin files: train.json, valid.json, and test.json
