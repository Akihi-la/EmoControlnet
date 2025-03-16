# Emo-Control

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) by Lvmin Zhang and Maneesh Agrawala.

This is based on the [training example in the original ControlNet repository](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md). It trains a ControlNet to show emo using a [emo dataset](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?rlkey=7f3oyjkr6zyndf0gau7t140rv&e=1&dl=0).

## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
cd diffusers/examples/controlnet
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

## Emo dataset

The original dataset is hosted in the [ControlNet repo](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?rlkey=7f3oyjkr6zyndf0gau7t140rv&e=1&dl=0). We re-uploaded it to be compatible with `datasets` [here](https://drive.google.com/file/d/1rkTWdEOH3QxoSmZPe_fY0sguZLejsrkE/view?usp=sharing). Note that `datasets` handles data loading within the training script.

Our training examples use [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) as the original set of ControlNet models were trained from it. Or you can use [this]()

## Training

```bash
#!/bin/bash

python train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5" \
  --output_dir="res" \
  --dataset_name="EmoSet/emoset.py" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --validation_image "EmoSet/source/amusement_00000.jpg" "EmoSet/source/sadness_12982.jpg" \
  --validation_prompt "there are two children standing next to a woman on the beach" "people are sitting at a table with candles and flowers" \
  --train_batch_size=2 \
  --num_train_epochs=5 \
  --checkpointing_steps=1000
```

