# `braai` \[Bogus/Real Adversarial AI\]
## Real-bogus classification for the Zwicky Transient Facility using deep learning

Efficient automated detection of flux-transient, reoccurring flux-variable, and moving objects 
is increasingly important for large-scale astronomical surveys. `braai` is a convolutional-neural-network, 
deep-learning real/bogus classifier designed to separate genuine astrophysical events and objects 
from false positive, or bogus, detections in the data of the [Zwicky Transient Facilty (ZTF)](https://ztf.caltech.edu), 
a new robotic time-domain survey currently in operation at the Palomar Observatory in California, USA.
`braai` demonstrates a state-of-the-art performance as quantified by 
its low false negative and false positive rates.

### `braai` architecture

![](doc/fig-braai.png)

### Dataset

todo: plots  

### Classifier performance

![](doc/fig-perf_d6_m7.png)

### Use `braai`

todo: jupyter notebook: find a nice SN on Kowalski, grab it from mars.lco.global, make triplet, run braai

#### Edge TPU

### Train your own `braai`

#### Colab

todo:

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