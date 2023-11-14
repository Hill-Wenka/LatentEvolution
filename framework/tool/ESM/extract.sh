export MKL_THREADING_LAYER=GNU
source /home/hew/anaconda3/bin/activate AggNet
echo 'using conda environment: AggNet'
echo "executed file: $0"
echo "model: $1"
echo "fasta: $2"
echo "output: $3"
repr_layers=(${4//,/ })
echo "repr_layers: ${repr_layers[@]}"
inclue=(${5//,/ })
echo "inclue: ${inclue[@]}"
python /home/hew/python/AggNet/framework/module/esm/esm2/extract.py $1 $2 $3 --repr_layers ${repr_layers[@]} --include ${inclue[@]}
