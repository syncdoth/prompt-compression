cuda_ids=$1

CUDA_VISIBLE_DEVICES=$cuda_ids python main.py \
    --model_name_or_path='facebook/blenderbot-3B' \
    --wandb_project='dialog-compression' \
    --wandb_runname='blenderbot-s100' \
    --checkpoint_dir='ckpt' \
    --seed=100 \
    --max_steps=250 \
    --epochs=3 \
    --lr=0.1 \
    --num_warmup_steps=0 \
    --weight_decay=0.01 \
    --batch_size=16 \
    --eval_batch_size=16 \
    --max_seq_length=128 \
    --prompt_length=8