# Image Classification by CVT based on Vit from scratch

This is a simplified PyTorch implementation of the mc-Vit (modified compact version of ViT) based on c-Vit. Reference to the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) and [Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704v4). 

The goal of this project is to implement a simple and easy-to-understand mc-Vit model to deal with fast image classification tasks. We evaluated the performance between the baseline model c-Vit (with CLS token) and mc-Vit(modified compact ViT) with sequence pooling on CIFAR-10 and CIFAR-100 datasets. 

![image](https://github.com/user-attachments/assets/f082ecb6-8bdd-44d0-b68b-1bc0cba348c2)

We further implement a fast multi-head attention module where all heads are processed simultaneously with merged query, key, and value.

![image](https://github.com/user-attachments/assets/25fdb3d8-de1a-4344-9b3e-8258e924847f)

ALL FROM SCRATCH!!

## Usage

Dependencies:
- PyTorch 1.13.1 ([install instructions](https://pytorch.org/get-started/locally/))
- torchvision 0.14.1 ([install instructions](https://pytorch.org/get-started/locally/))
- matplotlib 3.7.1 to generate plots for model inspection

Dependencies installation
```bash
pip install -r requirements.txt
```

The `vit.py` file shows how we implement the embeddings, 2 ways of multi-head attention, the transformer encoder, and the vit for classification. 

In `train.py`, you can experiment with different hyperparameters by editing config dictionary. Training parameters can also be passed using the command line.

For example, to train the model for 100 epochs with a batch size of 256(the default, in order to demonstrate), you can run:

```bash
python train.py --exp-name vit-with-10-epochs --epochs 100 --batch-size 256
```

## Results

The model was trained on the CIFAR-10 dataset / CIFAR-100 dataset for 100 epochs with a batch size of 256. The learning rate was set to 0.01 and no learning rate schedule was used. The model config is as below:

```python
config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
```

The model is much smaller than the original ViT models from the paper (which has at least 12 layers and hidden size of 768) as I just want to illustrate how the model works rather than achieving state-of-the-art performance.

Here below is our model size comparison:

![image](https://github.com/user-attachments/assets/537d9af8-be89-4a0f-ab00-1839ed30d8a4)

These are some results of the model:

CIFAR-10:

![image](https://github.com/user-attachments/assets/9b1ca33d-9e34-429c-8fb0-8a24309f0146)

CIFAR-100:

![image](https://github.com/user-attachments/assets/1a94ee9b-6732-4b72-9c1e-e1b9d9f0db71)

*Train loss, test loss and accuracy of the c-Vit and mc-Vit model trained on CIFAR-10, CIFAR-100.*


The mc-Vit model was able to achieve 77.25% acc for CIFAR-10 and 47.21% acc for CIFAR-10 on the test set after 100 epochs of training.


