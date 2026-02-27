# cd llm_finetuning
# ./pipline.sh
# cd ..

# Takes _APPS.json (2k lines) => .json (24k lines = original + 11 mutations,
# which start with <s>)
python perturb.py \
  --input_dir dataset/APPS/ \
  --input_file train_victim_APPS.json \
  --output_file train_victim.json

python perturb.py \
  --input_dir dataset/APPS/ \
  --input_file test_victim_APPS.json \
  --output_file test_victim.json

cd llm_inference
./infer.sh
cd ..

python classifier.py \
  --input_dir "dataset/APPS/" \
  --true_file "train_santacoder_victim_infer.txt" \
  --false_file "test_santacoder_victim_infer.txt" \
  --true_gt_file "train_victim.json" \
  --false_gt_file "test_victim.json" \
  --feature_path "feature.npz" \
  --n_samples_per_class 8 \
  --global_random_seed 140120031 \
  --random_state 676269283 \
  --random_state_test 212129145 \
  --dropout 0.1 \
  --batch_size 4 \
  --lr 1e-3 \
  --num_epochs 25 \
  --hidden_dims 512 512 512
