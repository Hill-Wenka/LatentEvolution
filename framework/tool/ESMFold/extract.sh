export MKL_THREADING_LAYER=GNU
source /home/hew/anaconda3/bin/activate esmfold
echo 'using conda environment: esmfold'
echo "executed file: $0"
echo "-i: $1"
echo "-o: $2"
echo "--max-tokens-per-batch: $3"
echo "--num-recycles: $4"
echo "--cpu-only: $5"
echo "--cpu-offload: $6"

if [[ $5 == False && $6 == False ]]; then
  python /home/hew/python/AggNet/framework/tool/ESMFold/esmfold_inference.py -i $1 -o $2 --max-tokens-per-batch $3 --num-recycles $4
elif [[ $5 == True && $6 == False ]]; then
  python /home/hew/python/AggNet/framework/tool/ESMFold/esmfold_inference.py -i $1 -o $2 --max-tokens-per-batch $3 --num-recycles $4 --cpu-only
elif [[ $5 = False && $6 = True ]]; then
  python /home/hew/python/AggNet/framework/tool/ESMFold/esmfold_inference.py -i $1 -o $2 --max-tokens-per-batch $3 --num-recycles $4 --cpu-offload
else
  python /home/hew/python/AggNet/framework/tool/ESMFold/esmfold_inference.py -i $1 -o $2 --max-tokens-per-batch $3 --num-recycles $4 --cpu-only --cpu-offload
fi
