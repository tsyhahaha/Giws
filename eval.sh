python /root/autodl-tmp/GTClassifier/scrips/eval.py \
    --model_or_path /root/autodl-tmp/GTClassifier/output/checkpoints/step_0.ckpt \
    --test_data_folder /root/autodl-tmp/DataSet/test \
    --test_batch_size 128 \
    --result_file eval.json
