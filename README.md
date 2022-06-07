# Fashion-MNIST-classifier-pytorch-CNN-MLFlow

# STEPS

## Step 01 - Create a repository by using template repository
## step 02 - Clone the new repository
## step 03 - Create a conda environment after opening the repository in VSCODE
```
conda create --prefix ./env python=3.7 -y
```
## activate environment
```
conda activate ./env
```
### or
```
source activate ./env
```

## STEP 04- install the requirements
```
pip install -r requirements.txt
```
## step 05- install pytorch 11.3

```
--------------USE ANY ONE---------------------------------
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch    <<--- for cuda toolkit (GPU) , using conda 
conda install pytorch torchvision torchaudio cpuonly -c pytorch             <<--- for cpu , , using conda   
pip3 install torch torchvision torchaudio                                   <<--- pip installation
```
## or use init_setup.sh if not want run step 01 to step 05
### in bash terminal use below command
```
bash init_setup.sh
```
## step 06- install setup.py
```
pip install -e .
``` 

==================== # Explaination ==============================

Classifies between ants and bees

We use MLops pipeline for this project to smoothen the process and seprate each stage from each other 

This project we divided in 5 stages. we use tenserflow to define our architecture

1) stage for downloading data as zip file , extract zip in defined folder in their respective folder of ants and bees

2) Use vgg-16 CNN Archeitecture layer as our model and create a base model and dump in our defined directory

3) Here we preprocess our images data before passing to our vgg-16 layer and do some data augmentation to generalize our model

4) We train our VGG-16 base model using Stochastic gradient descent as our optimizer and dump our model in defined model directory

5) Stage for prediction 

we create utils directory for all types of function such as some common , data management and for evaluating model. 
We also consider the situation if we stuck anywhere by creating log files for each and every steps for each stage
