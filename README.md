# DynProfiler
This is the repository of the code for DynProfiler: A Python package to analyze and interpret the entire signaling dynamics leveraged by deep learn-ing techniques

## Requirements
- Python (version not specified)
- Pytorch (version not specified)
- Captum (version not specified)

## Usage
```bash
git clone http://
```
Please set the path so that this package can be imported.  

```python
import sys
sys.path.append( )
```
### 1. Embed the dynamics
Please define the model configuration and prepare input data. After running `dynpro.embed()`, you can check the embedding result file as `npy`.
```python
import dynprofiler as dynpro

## Please configure the model by refferring to data/inputs
import yaml
train_params = yaml.safe_load(open("data/inputs/train_params.yml"))
model_params = yaml.safe_load(open("data/inputs/model_params.yml"))
dataset_params = yaml.safe_load(open("data/inputs/dataset_params.yml"))
outdir = "data/outputs"

## If you train the model using random sampling, please specify the mean and std.
import numpy as np
inp_mean = np.load("data/inputs/input_mean.npy")
inp_std = np.load("data/inputs/input_std.npy")

## Run
dynpro.embed({"mean": inp_mean, "std": inp_std},
            outdir,
            **train_params, **model_params, **dataset_params)
```
### 2. Extracting important dynamics
Please define the model configuration and prepare input data and labels.   
You can run Step2 alone without having executed the self-supervised pre-training in Step1.  
After running `dynpro.interpret()`, you can check the resulting `npy` file that represent the time-dependent attributions of each variable.
```python
import dynprofiler as dynpro

## Please configure the model by refferring to data/inputs
import yaml
train_params = yaml.safe_load(open("data/inputs/train_params.yml"))
model_params = yaml.safe_load(open("data/inputs/model_params.yml"))
dataset_params = yaml.safe_load(open("data/inputs/dataset_params.yml"))
outdir = "data/outputs"

## Load Data
import numpy as np
inp_mean = np.load("data/inputs/input_mean.npy")
labels = np.load("data/inputs/labels.npy")

## Run
dynpro.interpret({"mean": inp_mean}, outdir, labels,
                **train_params, **model_params, **dataset_params)
```

## Fig2
- Reproduction codes for Fig. 2
- Large files, such as input simulated dynamics and model weights, are not stored here. If needed, please contact the author.