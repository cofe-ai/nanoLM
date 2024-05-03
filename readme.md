# nanoLM

## ✨ Introduction

We present an approach (namely **μScaling**) to predict the pre-training loss, based on our observations that Maximal Update Parametrization (μP) enables accurate fitting of scaling laws close to common loss basins in hyperparameter space. With μScaling, different model designs can be compared on large scales by training only their smaller counterparts. Further, we introduce **nanoLM**: an affordable LLM pre-training benchmark that facilitates this new research paradigm. We will continue to develop it, and also welcome feedbacks and ideas from the community.


## 🛠️ Environment Setup

```
conda env create -f environment.yml
```

## 📚 Data Preparation

The format of the pre-training data is the same with C4 and mC4. We downloaded it from [huggingface C4 data](https://huggingface.co/datasets/c4),[huggingface mC4 data](https://huggingface.co/datasets/mc4). eg.,
```python
from datasets import load_dataset
dataset = load_dataset("c4")
```
You can develop your own pretraining dataset following this format.
## 🏗️ Train your model with μScaling

nanoLM offers support for decoder-only structures (eg., GPT,Llama), encoder-only structures (eg., BERT), and encoder-decoder structures (eg., T5).

### decoder-only structures 

```
bash scripts/pretrain_gpt.sh
bash scripts/pretrain_llama.sh
```

### encoder-only structures

```
bash scripts/pretrain_bert.sh
```

### encoder-decoder structures 

```
bash scripts/pretrain_t5.sh
```

The training log will saved in ./logs, include time step and loss. 

## 🌈 Fit scaling laws and Loss prediction

Use training loss under different width to fit scaling laws  for loss prediction.

```
python fit_loss.py
```

The coefficients a,b,c obtained by fitting the power-law function $L=aC_t^b+c$, according to the loss will be printed on the terminal. 


# 🙏🏻 Acknowledgements

This project incorporates and modifies code from the following open-source repositories   [nanoGPT](https://github.com/karpathy/nanoGPT)  and [MuTransformers](https://github.com/microsoft/mutransformers). We extend our gratitude to the original authors for their innovative work and for making it available to the community.

## Citation
```
@misc{yao2024nanolm,
      title={nanoLM: an Affordable LLM Pre-training Benchmark via Accurate Loss Prediction across Scales}, 
      author={Yiqun Yao and Siqi fan and Xiusheng Huang and Xuezhi Fang and Xiang Li and Ziyi Ni and Xin Jiang and Xuying Meng and Peng Han and Shuo Shang and Kang Liu and Aixin Sun and Yequan Wang},
      year={2024},
      eprint={2304.06875},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```