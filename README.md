# KaoKore-VQ-VAE2

## Installation

Install required python packages using `pip3 -r install kaokore/requirements.txt`.

## Usage

Navigate to the kaokore directory to run the code (`cd kaokore/`).

Run VQ-VAE with:

```
python3 train_vae.py --dataset kaokore --model vqcvae --data-dir ../data --epochs 100 --batch-size 64
```

Run VQ-VAE2 with:

```
python3 train_vqvae2.py ../data --epoch 100 --batchsize 64 --lr 1e-4
```