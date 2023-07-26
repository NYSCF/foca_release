[<img src="https://img.shields.io/badge/license-Business%20Source%20License%201.1-red?style=flat-square&logo=appveyor"></img>](https://github.com/NYSCF/foca_release/blob/main/LICENSE)
# FocA: A deep learning tool for reliable, near-real-time imaging focus analysis in automated cell assay pipelines
<!-- > Deep learning-based tool for classifying focal quality of microscopy images in highly automated cell production pipelines -->
<!-- > Live demo [_here_](https://www.example.com). If you have the project hosted somewhere, include the link here. -->
### The New York Stem Cell Foundation
<img src="https://github.com/NYSCF/FocA_release/blob/main/assets/figures/fig1_workflow.png"></img>
The increasing use of automation in cellular assays and cell culture presents significant opportunities to enhance the scale and throughput of imaging assays, but to do so, reliable data quality and consistency are critical. Realizing the full potential of automation will thus require the design of robust analysis pipelines that span the entire workflow in question. Here we present FocA, a deep learning tool that, in near real-time, identifies in-focus and out-of-focus images generated on a fully automated cell biology research platform, the NYSCF Global Stem Cell Array®. The tool is trained on small patches of downsampled images to maximize computational efficiency without compromising accuracy, and optimized to make sure no sub-quality images are stored and used in downstream analyses. The tool automatically generates balanced and maximally diverse training sets to avoid bias. The resulting model correctly identifies 100% of out-of-focus and 98% of in-focus images in under 4 seconds per 96-well plate, and achieves this result even in heavily downsampled data (~30 times smaller than native resolution). Integrating the tool into automated workflows minimizes the need for human verification as well as the collection and usage of low-quality data. FocA thus offers a solution to ensure reliable image data hygiene and improve the efficiency of automated imaging workflows using minimal computational resources.

Read the preprint [here](https://www.biorxiv.org/content/10.1101/2023.07.20.549929v1):

Winchell, J., Comolet, G., Buckley-Herd, G., Hutson, D., Bose, N., Paull, D., & Migliori, B. (2023). **FocA: A deep learning tool for reliable, near-real-time imaging focus analysis in automated cell assay pipelines**. Preprint.
## Table of Contents
- [FocA: A deep learning tool for reliable, near-real-time imaging focus analysis in automated cell assay pipelines](#foca-a-deep-learning-tool-for-reliable-near-real-time-imaging-focus-analysis-in-automated-cell-assay-pipelines)
		- [The New York Stem Cell Foundation](#the-new-york-stem-cell-foundation)
	- [Table of Contents](#table-of-contents)
	- [Technologies Used](#technologies-used)
	- [Features](#features)
- [Using the Tool](#using-the-tool)
	- [Getting Started](#getting-started)
	- [Train a model on your own data](#train-a-model-on-your-own-data)
	- [Use a pre-trained network](#use-a-pre-trained-network)
	- [Deploy FocA](#deploy-foca)
		- [Data Format](#data-format)
	- [More Help](#more-help)



<!-- - Provide general information about your project here.
- What problem does it (intend to) solve?
- What is the purpose of your project?
- Why did you undertake it? -->
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python 3.8.10
- Tensorflow 2.8.0
- CUDA 11.5 + cuDNN

NOTE: Tested on Ubuntu 22.04, you may encounter issues when using operating systems...
## Features
- Slack integration
- Automated train/test set generation with class imbalance compensation



# Using the Tool
First clone the main branch of the repository and navigate into the main directory:
```
git clone -b main git@github.com:NYSCF/foca_release.git
cd foca_release/
```
## Getting Started
FocA depends on the use of Anaconda which can be downloaded [here](https://www.anaconda.com/products/distribution)
1. Create and activate a conda environment with a modern python version:
	```
	conda create -n foca python=3.8.10 pip
	conda activate foca
	```
2. Install the required libraries:
	```
	pip install -r requirements.txt
	```
3. Update parameters in [`input/config.py`](https://github.com/NYSCF/foca_release/blob/main/input/config.py)

4. Run tests in project root directory to ensure the parameters entered are valid:
	```
	pytest				# test everything
	pytest tests/test_deployment.py	# test only deployment code
	pytest tests/test_train.py	# test only model training code
	```
	NOTE: Only deployment tests need to pass, if you're using our pre-trained weights
	
## Train a model on your own data

After updating the parameters in [`input/config.py`](https://github.com/NYSCF/foca_release/blob/main/input/config.py), run
```
python train/train_model.py
```
This command will train a model and save the final weights to a subdirectory called `foca_weights/` (this can be changed using the `-n` flag described below) of the `WEIGHT_DIR` path specified in [`input/config.py`](https://github.com/NYSCF/foca_release/blob/main/input/config.py).

You can add optional flags:
- `-n model_name`: name of model directory; if exists, you will be asked if you want to overwrite it
- `-y yaml_name`: name of YAML file containing model training parameters; default is `default.yaml`
- `-d device`: choose to train using `CPU` or `GPU`, default is `GPU` but will compute on `CPU` if not available
- `--verbose`: add this if you want more verbose output during training, there is no argument for this flag
- `--generic`: add this if your data is not formatted like the example, it will train the model without optimizing plate variety but will still attempt to balance the two classes
- `-h`: print descriptions of what different flags mean
## Use a pre-trained network
After downloading the weights [here](https://storage.googleapis.com/foca2023/pretrained_weights/foca_weights.zip), extract the zip file and specify the folder's location as the `MODEL_WEIGHTS` parameter in [`input/config.py`](https://github.com/NYSCF/foca_release/blob/main/input/config.py).

## Deploy FocA
After updating the parameters in [`input/config.py`](https://github.com/NYSCF/foca_release/blob/main/input/config.py) and either training your own model or downloading our weights, run
```
python deployment/foca_main.py 
```
This command will search subdirectories of `IMAGE_DIR/` (specified in [`input/config.py`](https://github.com/NYSCF/foca_release/blob/main/input/config.py)) for new scans of plates, analyze any previously un-analyzed plate scans, and write the results to the CSV file `CSV_OUTPUT` (also specified in [`input/config.py`](https://github.com/NYSCF/foca_release/blob/main/input/config.py)).

You can add optional flags:
- `-r run_name`: process all plates in a single run
- `-p plate_name`: process a single plate
- `--overwrite_scans`: overwrite existing scan data
- `-h`: print descriptions of what different flags mean

### Data Format
Data should be structured according to this pattern:
```
<path_to_runs/>/RUN-NAME/RUN-NAME_PLATE-NAME/TIFF<scale-in-micron-meters-per-pixel>/RUN-NAME_PLATE-NAME_SCAN-DATETIME/Well_ROWCOL_Ch<channel_number>_<scale-in-micron-meters-per-pixel>um.tiff
```
for example:
```
/network_drive/DeadTotalImages/RUN0020/RUN0020_PLATE_201/TIFF06/RUN0020_PLATE_201_11-23-2022-2-10-44-PM/Well_B10_Ch2_6um.tiff
```
<!-- ## Usage
How does one go about using it?
Provide various use cases and code examples here.

`write-your-code-here` -->

## More Help
For more information on either training a model or deploying the tool, feel free to [open an issue](https://github.com/NYSCF/foca_release/issues).

## Citation
If you find this repository useful for your research, please consider giving us a star ⭐ and cite our [bioRxiv paper](https://www.biorxiv.org/content/10.1101/2023.07.20.549929v1):

```
FocA: A deep learning tool for reliable, near-real-time imaging focus analysis in automated cell assay pipelines
Jeff Winchell, Gabriel Comolet, Geoff Buckley-Herd, Dillion Hutson, Neeloy Bose, Daniel Paull, Bianca Migliori 
bioRxiv 2023.07.20.549929; doi: https://doi.org/10.1101/2023.07.20.549929
```
FocA<sup>SM</sup> &copy; 2023 by NYSCF is licensed under Business Source License 1.1.
