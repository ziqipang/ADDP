# Referring Segmentation

The implementaiton is based on [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix).

## Setup

To set up the environment:

```bash
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate ip2p

# Download the pre-trained checkpoint
bash scripts/download_checkpoints.sh
```

## Dataset

We follow the dataset setup from [CRIS](https://github.com/DerrickWang005/CRIS.pytorch). Please refer to their repository for detailed dataset preparation instructions.

## Training

To start training,

```bash
python main.py --name "<exp-name>" --base configs/refcoco.yaml --train --gpus 0,1
```

> You will need to modify the dataset directory path in `configs/refcoco.yaml` to match your local setup.

## Evaluation

To generate predicted images, run

```bash
python generate.py --dataset-root "<datasets/RefCOCO>" --model-ckpt "<model.ckpt>" --save-dir "<result-dir>"
```

To evaluate the results, run

```bash
python evaluate.py --dataset-root "<datasets/RefCOCO>" --result-dir "<result-dir>"
```

## Checkpoints

Our checkpoints can be found in this [link](https://drive.google.com/drive/folders/18BxEsdkk6zf8pYW9ZH-eXdcxMMR0KPLE?usp=sharing).
