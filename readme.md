# `braai` \[Bogus/Real Adversarial AI\]
## Real-bogus classification for the Zwicky Transient Facility using deep learning

Efficient automated detection of flux-transient, reoccurring flux-variable, and moving objects 
is increasingly important for large-scale astronomical surveys. `braai` is a convolutional-neural-network, 
deep-learning real/bogus classifier designed to separate genuine astrophysical events and objects 
from false positive, or bogus, detections in the data of the [Zwicky Transient Facilty (ZTF)](https://ztf.caltech.edu), 
a new robotic time-domain survey currently in operation at the Palomar Observatory in California, USA.
`braai` demonstrates a state-of-the-art performance as quantified by 
its low false negative and false positive rates.

For details, please see [Duev et al. 2019, MNRAS, 489 (3), 3582-3590](https://academic.oup.com/mnras/article/489/3/3582/5554758).

[arXiv:1907.11259](https://arxiv.org/pdf/1907.11259.pdf)

### `braai` architecture

![](doc/fig-braai.png)

### Dataset

todo: plots  

### Classifier performance

![](doc/fig-perf_d6_m7.png)

### Use `braai`

See [this jupyter notebook](https://github.com/dmitryduev/braai/blob/master/nb/braai_run.ipynb)

#### Edge TPU

### Transfer learning with `braai`

#### Jupyter/Colab

See [this jupyter notebook](https://github.com/dmitryduev/braai/blob/master/nb/braai_tl.ipynb), or 
[![Open In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmitryduev/braai/blob/master/nb/braai_tl.ipynb)

### Train your own `braai`

#### Jupyter/Colab

See [this jupyter notebook](https://github.com/dmitryduev/braai/blob/master/nb/braai_train.ipynb), or 
[![Open In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmitryduev/braai/blob/master/nb/braai_train.ipynb)

#### Docker

Build and launch the app container:
```bash
# without GPU support:
docker build --rm -t braai:cpu -f Dockerfile .
# with GPU support (requires nvidia-docker):
docker build --rm -t braai:gpu -f gpu.Dockerfile .

# run:
# without GPU support:
docker run -it --rm --name braai -v /path/to/store/data:/data braai:cpu
# with GPU support (requires nvidia-docker) exposing the first GPU:
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -it --rm --name braai -v /path/to/store/data:/data braai:gpu

```

---

Train `braai`:

```bash
python /app/braai.py --t_stamp 20190614_003916 --model VGG6 --epochs 200 --patience 50 --batch_size 64 --verbose
```