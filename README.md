# Opinin_Free_BIQA
This is the repository contains the official pytorch implementation of the paper [Toward a blind image quality evaluator in the wild by learning beyond human opinion scores](https://www.sciencedirect.com/science/article/pii/S0031320322007750), Zhihua Wang, Zhiri Tang, Jianguo Zhang, and Yuming Fang, Pattern Recognition, 2023.

You can download the pre-trained weights adapting to [KonIQ-10k](https://drive.google.com/drive/folders/1KIIwMplZbWSZzmtMlTjCHGaYmM0th8ay?usp=sharing) and [SPAQ](https://drive.google.com/drive/folders/1KIIwMplZbWSZzmtMlTjCHGaYmM0th8ay?usp=sharing).

# The generation of pseudo-labeled dataset:
1. You need to download Waterloo Exploration Database (https://ece.uwaterloo.ca/~k29ma/exploration/) first, and then leverage the [distortion generation codes](https://github.com/wangzhihua520/OF_BIQA/tree/main/imgs_generator_and_pseudo_labels) to simulate distorted images.
2. The [FR-IQA models](https://github.com/wangzhihua520/OF_BIQA/tree/main/FR-IQAs) for pseudo-label predictions includes FSIMc, SR-SIM, NLPD, VSI, MDSI and GMSD, released by respectively authors.
3. Randomly sample image pairs and assign binary pseudo-labels.

# Train & Test
You can run the Main.py for training and the test_SPAQ.py and test_KonIQ.py for testing

# Citation
If you find the repository helpful in your resarch, please cite the following papers.
```sh
@article{wang2023toward,
title = "Toward a blind image quality evaluator in the wild by learning beyond human opinion scores", 
author = "Zhihua Wang and Zhi-Ri Tang and Jianguo Zhang and Yuming Fang",  
year = "2023",  
volume = "137",  
journal = "Pattern Recognition"}





