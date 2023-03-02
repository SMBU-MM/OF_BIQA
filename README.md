# Opinin_Free_BIQA
This is the repository of paper [Toward a blind image quality evaluator in the wild by learning beyond human opinion scores](https://www.sciencedirect.com/science/article/pii/S0031320322007750)

You can download the pre-trained weights adapting to [KonIQ-10k](https://drive.google.com/drive/folders/1KIIwMplZbWSZzmtMlTjCHGaYmM0th8ay?usp=sharing) and [SPAQ](https://drive.google.com/drive/folders/1KIIwMplZbWSZzmtMlTjCHGaYmM0th8ay?usp=sharing).

# The generation of pseudo-labeled dataset:
1. You need to download Waterloo Exploration Database (https://ece.uwaterloo.ca/~k29ma/exploration/) first, and then leverage the matlab codes to simulate distortions  
2. The FR-IQA models for pseudo-label predictions are FSIMc, SR-SIM, NLPD, VSI, MDSI and GMSD, released by respectively authors.
3. Randomly sample pairs from the pseudo-labeled images.

# Train & Test
You can run the Main.py for training

