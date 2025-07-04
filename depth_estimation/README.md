# Depth Estimation

The implementation is based on [Marigold](https://github.com/prs-eth/Marigold).

## Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate marigold
```

## Training

1. Install additional dependencies:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

2. Set environment variables:

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>
export BASE_CKPT_DIR=<YOUR_CHECKPOINT_DIR>
```

Download [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and store it in `${BASE_DATA_DIR}`. Please refer to [README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.

Download the pre-trained [Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2) checkpoint into `${BASE_CKPT_DIR}`.

3. Run the training script:

```bash
# Set environment parameters for dataset and checkpoint
export BASE_DATA_DIR=<YOUR_DATA_DIR>
export BASE_CKPT_DIR=<YOUR_CHECKPOINT_DIR>

python train.py --config config/train_marigold.yaml --do_not_copy_data
```

## Evaluation

1. Install additional dependencies:

```bash
pip install -r requirements+.txt -r requirements.txt
```

2. Download [evaluation datasets](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset):

```base
# Set environment variable
export BASE_DATA_DIR=<YOUR_DATA_DIR>

wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR} https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
```

3. Run inference and evaluation scripts:

```bash
# Run inference
bash script/eval/11_infer_nyu.sh

# Evaluate predictions
bash script/eval/12_eval_nyu.sh
```

## Checkpoints

Our checkpoints can be found in this [link](https://drive.google.com/drive/folders/18BxEsdkk6zf8pYW9ZH-eXdcxMMR0KPLE?usp=sharing).
