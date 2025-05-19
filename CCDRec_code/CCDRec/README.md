# Curriculum Conditioned Diffusion for Multimodal Recommendation

## Environment

- cuda 10.2
- python 3.8.10
- pytorch 1.12.0
- numpy 1.21.2

## Usage

### Data

The experimental data are in './data' folder, including Amazon-Baby. Sports and Clothing dataset are too large for uploading, whereas these datasets will be released with the code upon acceptance.

### Training

#### CCDRec_FREEDOM

```
cd ./src
python main.py -m ccdrec -d baby
```
