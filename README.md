# LatentDiffusionModel

For additional information please contact the authors (Oscar and Marcus) or Claes Nøhr Ladefoged to get your hands on our thesis where the logic is thorougly explained. 

This repository shows the fundamentals for setting up a Latent Diffusion Model for Style-Transfer in PET imaging using a patch based approach.

TorchIO (https://torchio.readthedocs.io/) has been used for extracting the 3D patches of the volumes and the framework is as follows:
![image](https://github.com/user-attachments/assets/52a8979e-4d58-42e8-985d-9b8b85dbf5c3)



The whole pipeline for the Latent Diffusion model is:
![image](https://github.com/user-attachments/assets/7b81fd72-8e87-40bb-9b5b-ce03cbec31ab)



The logic will be split into two stages:
Stage 1: Vector Quantized Variational Autoencoder.
Stage 2: Diffusion Model.

Stage 1 model architecture:
![image](https://github.com/user-attachments/assets/f201768f-ea4a-4ce3-84ea-7575c4b91080)


Stage 1 model training:
![VQVAEtrainingJANUARY7 (1)](https://github.com/user-attachments/assets/a61b4c0f-2029-4d0d-b2ff-589fd10c2749)



Stage 2 model architecture:
![image](https://github.com/user-attachments/assets/479569ab-0eff-48c8-891e-4cd91e0edfcb)


Stage 2 model training:
![LDMTRAININGFINAL_RESIZED (1)](https://github.com/user-attachments/assets/81c7edae-b917-47ba-acef-693c40a3d6de)
![LDMINFERENCEFINAL_RESIZED (1)](https://github.com/user-attachments/assets/cef3e387-2b5f-4d3c-8c76-7e2aa3a06f0d)














