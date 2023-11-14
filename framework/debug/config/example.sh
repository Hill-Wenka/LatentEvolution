source /home/hew/anaconda3/bin/activate AggNet
echo 'using conda environment: AggNet'
export PYTHONPATH=$PYTHONPATH:/home/hew/python/AggNet/
echo 'PYTHONPATH' $PYTHONPATH
cd '/home/hew/python/AggNet/framework/debug/config/'
python parse_config.py config_1.yaml config_2.yaml max_epochs=50 trainer_args.max_epochs=50 new_args.max_epochs=50
