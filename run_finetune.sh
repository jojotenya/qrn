#!/bin/bash
save_dir=`pwd`/saves/babi/finetune
echo ${save_dir}
num_trials=1
num_epochs=500
end_task=20
large=False

python -m babi.main --noload --save_dir ${save_dir} --task 1 --num_trials ${num_trials} --num_epochs ${num_epochs} --write_log=True --log_dir finetune --large=${large}
#python -m babi.main --train=False --save_dir ${save_dir} --task 1 --log_dir finetune --large ${large}

for ((i=2;i<=end_task;i++));
  do
    python -m babi.main --reset_epochs=True --save_dir ${save_dir} --task ${i} --num_trials ${num_trials} --num_epochs ${num_epochs} --write_log=True --log_dir finetune --large=${large}
    for ((j=1;j<=i-1;j++));
      do
        python -m babi.main --train=False --save_dir ${save_dir} --task ${j} --which_model ${i} --write_log=True --log_dir finetune --large=${large} 
      done
  done

python -m babi.main --reset_epochs=True --save_dir ${save_dir} --task 1 --num_trials ${num_trials} --num_epochs ${num_epochs} --write_log=True --log_dir finetune --large=${large}
for ((t=2;t<=end_task;t++));
  do
    python -m babi.main --train=False --save_dir ${save_dir} --task ${t} --which_model 1 --write_log=True --log_dir finetune --large=${large} 
  done
