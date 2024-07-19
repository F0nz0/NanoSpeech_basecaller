# NanoSpeech basecaller

A new Basecaller for the direct detection of modified bases in a single-step during basecalling procedures for direct-RNA runs from Oxford NanoPore sequencers. NanoSpeech is based on a Transformer model with an expanded dictionary with respect to canonical-bases only. It provides also utilities for the mapping of these modified bases onto a reference genome or transcriptome providing prediction on both per-read and genome-space level (aggregate data).

## **Required Softwares**:
NanoSpeech uses internally (and not) some software that should be installed preferably into a new conda enviroment. \
After the activation of the conda enviroment install the following softwares:
1) Python >= 3.7
2) Samtools >= 1.3.1
3) Minimap2 == 2.24

## **Installation**:
1) Download the source code from GitHub repository at the url:
        
    https://github.com/F0nz0/NanoSpeech_basecaller

2) Create a new virtual environment (it's suggested to create, use and activate a base conda environment with all the required software):

		# create a new conda environment
        conda create --name NanoSpeech python=3.8

		# activate the conda env
		conda activate NanoSpeech

		# install samtools
		conda install -c bioconda samtools >= 1.3.1

		# install minimap2
		conda install -c bioconda minimap2 == 2.24

		# create virtual environment inside Conda NanoSpeech env
		python3 -m venv NanoSpeech_basecaller

4) Activate the venv:
	
	    source NanoSpeech_basecaller/bin/activate

5) Upgrade pip version:
	
	    python3 -m pip install --upgrade pip

6) Install wheel package via pip:
	
	    pip install wheel

7) Install required Python packages using the requirements.txt file:

        python -m pip install -r requirements.txt

## **Basic Usage**:
The NanoSpeech basecaller is easy to use and need to be feed with a directory containing the fast5 to be basecalled.

		