do_train="True"
do_eval="True"
do_predict="False"
output_dir="output"
max_seq_length="384"
pad_to_max_length="False"
learning_rate="5e-05"
num_train_epochs="1"
per_device_train_batch_size="8"
per_device_eval_batch_size="8"
model_name_or_path="klue/bert-base"

python main.py \
    --do_eval ${do_eval} \
    --do_train ${do_train} \
    --do_predict ${do_predict} \
    --output_dir ${output_dir} \
    --max_seq_length ${max_seq_length} \
    --pad_to_max_length ${pad_to_max_length} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --model_name_or_path ${model_name_or_path}