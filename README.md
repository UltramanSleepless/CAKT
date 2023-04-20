# CAKT: Coupling contrastive learning with attention networks for interpretable knowledge tracing

This repository contains source code for the paper "Coupling contrastive learning with attention networks for interpretable knowledge tracing" to be presentated at IJCNN 2023.

## Requirements

```
python==3.7.12
PyTorch==1.11.0
pandas==1.3.5
numpy=1.21.5
```
## Usage 

### cloning the reposity

```
git clone 
cd CAKT
```

### Runing 

We evaluate our method on datasets **ASSIST2009**, **ASSIST2015**, **ASSISTs2017**, and **Span**.

```
python main.py --dataset assist2009_pid --model cakt_pid 
```
The results (AUC scores) will be saved in file **experiment_log**.

## Acknowledgments

### Reference
