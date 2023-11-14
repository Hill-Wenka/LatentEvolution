source /home/hew/anaconda3/bin/activate AggNet
echo 'using conda environment: AggNet'
export PYTHONPATH=$PYTHONPATH:/home/hew/python/AggNet/
echo 'PYTHONPATH' $PYTHONPATH
cd '/home/hew/python/AggNet/framework/config/'
python __init__.py command='task1' model=MLP model_hparams.activation='ReLU' model_hparams.max_seq_len=10
#python __init__.py /home/hew/python/AggNet/framework/debug/config/task_config.yaml
