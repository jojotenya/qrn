#!/bin/bash
lll_type=
name=finetune
save_dir=`pwd`/saves/babi/$name
log_dir=$name
echo ${save_dir}
num_trials=1
num_epochs=500
num_epochs=300
end_task=20
large=False
max_mem_size=50
mem_num_layers=2
max_mem_size=200
mem_num_layers=6
max_mem_size=300
mem_num_layers=8

rm -rf logs/babi/$name/*

python -m babi.main --noload --save_dir ${save_dir} --task 1 --num_trials ${num_trials} --num_epochs ${num_epochs} --write_log=True --log_dir ${log_dir} --large=${large} --lll_type=${lll_type}

# test before train tasks 2-20
for ((k=2;k<=end_task;k++));
  do
    python -m babi.main --train=False --save_dir ${save_dir} --task ${k} --which_model 1 --write_log=True --log_dir ${log_dir} --large=${large} --lll_type=${lll_type}
  done

for ((i=2;i<=end_task;i++));
  do
    python -m babi.main --reset_epochs=True --save_dir ${save_dir} --task ${i} --num_trials ${num_trials} --num_epochs ${num_epochs} --write_log=True --log_dir ${log_dir} --large=${large} --lll_type=${lll_type} 
    #for ((j=1;j<=i-1;j++));
    for ((j=1;j<=end_task;j++));
      do
        if [ $j != $i ]
        then 
          python -m babi.main --train=False --save_dir ${save_dir} --task ${j} --which_model ${i} --write_log=True --log_dir ${log_dir} --large=${large} --lll_type=${lll_type} 
        fi
      done
  done

python -m babi.main --reset_epochs=True --save_dir ${save_dir} --task 1 --num_trials ${num_trials} --num_epochs ${num_epochs} --write_log=True --log_dir ${log_dir} --large=${large} --lll_type=${lll_type}
for ((t=2;t<=end_task;t++));
  do
    python -m babi.main --train=False --save_dir ${save_dir} --task ${t} --which_model 1 --write_log=True --log_dir ${log_dir} --large=${large} --lll_type=${lll_type} 
  done
