set -e
python generate_sample.py --zmode mumu --n 500000 --seed 11 --out samples/zh240_mumu_A
python generate_sample.py --zmode mumu --n 500000 --seed 12 --out samples/zh240_mumu_B
python generate_sample.py --zmode ee   --n 250000 --seed 21 --out samples/zh240_ee_A
python generate_sample.py --zmode ee   --n 250000 --seed 22 --out samples/zh240_ee_B
python generate_sample.py --zmode qq   --n 250000 --seed 31 --out samples/zh240_qq_A
python generate_sample.py --zmode qq   --n 250000 --seed 32 --out samples/zh240_qq_B
python generate_sample.py --sqrts 250 --no-beam-effects --zmode mumu --n 20000 --seed 99 --out samples/zh250_mumu_check
echo ALLDONE
