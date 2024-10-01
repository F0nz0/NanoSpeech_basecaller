# NanoSpeech basecaller

<p align="center">
<img src="https://github.com/F0nz0/NanoSpeech_basecaller/blob/master/OIG4.LacWFMr4ivi0OKPCIuqq.jpg" width="300" height="300" align="center">
</p>
<sub><sup>*image created in Copilot by Designer using DALL·E 3 technology</sub></sup><br><br>



A new Basecaller software for the single-step direct detection of modified bases during basecalling procedures for direct-RNA runs produced via Oxford NanoPore sequencers. NanoSpeech is based on a Transformer model with an expanded dictionary with respect to canonical-bases only (A,C,G,U,I=Inosine). It provides also utilities for the mapping of these modified bases onto a reference genome or transcriptome providing predictions on both a per-read and a genome-space level (aggregated data).

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
The NanoSpeech basecaller is easy to use and need to be feed with a directory containing the fast5 to be basecalled and converted into fasta/fastq files.

1) General Usage:

	Print the help message to visualize all the avaiable options:
		
		python NanoSpeech.py -h

		usage: NanoSpeech.py [-h] -d FAST5_FOLDERPATH -o OUT_FILEPATH -m MODEL_WEIGTHS [-t THREADS_N]
                     [-c CLIP_OUTLIERS] [-u PRINT_GPU_MEMORY] [-r PRINT_READ_NAME] [-nr N_READS_TO_PROCESS]
                     [-fl FAST5LIST_FILEPATH] [-rl READSLIST_FILEPATH] [-cl CHUNKS_LEN]
                     [-idxs PRINT_CHUNKS_IDXS]

		NanoSpeech basecaller.py v. 0.0.1

		optional arguments:
		-h, --help            show this help message and exit
		-d FAST5_FOLDERPATH, --fast5_folderpath FAST5_FOLDERPATH
								--fast5_folderpath: a <str> with the fullpath for the input fast5 folderpath.
		-o OUT_FILEPATH, --out_filepath OUT_FILEPATH
								--out_filepath: a <str> with the fullpath for the output fasta/fastq file generated
								during the basecalling.
		-m MODEL_WEIGTHS, --model_weigths MODEL_WEIGTHS
								--model_weigths: a <str> with the fullpaht for the h5 file containing the weights to
								inizialize pretrained transformer model.
		-t THREADS_N, --threads_n THREADS_N
								--threads_n: a <int> indicating the number of basecaller workers.
		-c CLIP_OUTLIERS, --clip_outliers CLIP_OUTLIERS
								--clip_outliers: a <str> indicating the min-max currents value to be clipped with
								mean. [30-175]
		-u PRINT_GPU_MEMORY, --print_gpu_memory PRINT_GPU_MEMORY
								--print_gpu_memory: <str> Set to True to print gpu usage for every worker starting a
								new read (experimental). [False]
		-r PRINT_READ_NAME, --print_read_name PRINT_READ_NAME
								--print_read_name: <bool> Set to True to let producer to print reads names added to
								queue. [False]
		-nr N_READS_TO_PROCESS, --n_reads_to_process N_READS_TO_PROCESS
								--n_reads_to_process: <int> Numer of reads to limit basecalling. [None]
		-fl FAST5LIST_FILEPATH, --fast5list_filepath FAST5LIST_FILEPATH
								--fast5list_filepath: <str> Fullpath for a file with list of paths to fast5 files to
								limit basecalling on. [None]
		-rl READSLIST_FILEPATH, --readslist_filepath READSLIST_FILEPATH
								--readslist_filepath: <str> Fullpath for a file with list of reads ids to limit
								basecalling on. [None]
		-cl CHUNKS_LEN, --chunks_len CHUNKS_LEN
								--chunks_len: <int> Chunks lenght the generator will be output from raw signals.
								[2800]
		-idxs PRINT_CHUNKS_IDXS, --print_chunks_idxs PRINT_CHUNKS_IDXS
								--print_chunks_idxs: <bool> Set to True to print 0-based index for the end of chunks
								in the + line in fastq output [None]

2) Basecalling of fast5 files. 

	It's important to consider at least 3 GB of GPU memory for each parallel thread. For an NVIDIA-A100 40GPU 15 threads shuold be a good amount of parallel models. Here a simple commando to perform inosine-aware basecalling of fast5 from dRNA (002) runs:

		python3 NanoSpeech.py -d {fast5_folder_path} -o {fasta/fastq output file} -m {model_weigths in ./models/} -t {number of threads}

	NanoSpeech will produce either a fasta or fastq file where a list of indexes of adenosines with higher probability to be inonsines is added in the header of each read.

3) Alignments of fasta/fastq files.

	Mapping of the basecalled reads via minimap2 using specific presets to store metadata related each inosine reference positions. The alignment step can be executed using either a reference genome or a reference transcriptome. 
	For a reference genome we suggest to use this command:
		
		minimap2 -ax splice -uf -k 14 --secondary=no --MD {ref_path} {out_filepath} > {out_filepath}.sam

	For a reference transcriptome insteat use this alternative command:

		minimap2 -ax map-ont -k 14 --secondary=no --MD {ref_path} {out_filepath} > {out_filepath}.sam

	Then filter the produced SAM files, converto to sorted and indexed BAM files. The plain-text alignment file can be finally removed:

		samtools view -b {out_filepath}.sam | samtools sort -O BAM > {out_filepath}.bam
		samtools index {out_filepath}.bam
		rm {out_filepath}.sam

4) After the mapping procedures via minimap2, it's time to detect at a per-read level and then, on aggregated genome-space, the inosines mapped on the reference sequences. NanoSpeech provides an accessory script to automatize this crucial step. The script will use BAM files and inosines indexes in the fasta/q headers for each read to detect the reference coordinates.

		python inosine_detector.py -b {bam_file_path} -f {fasta/q file_path} -q {minimum inosine quality}

5) The script will produce two main output files:
	
	A) per-read predictions: it will be named as {bam_file_path}.per_read.bed. This is a bed-like file with 7 columns where each row being a mapped inosine with information about: region, start, stop, read-id, inosine quality, strand, and the reference base. Below few example rows:
		
		gBlock1	26.0	27.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	3	+	A
		gBlock1	34.0	35.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	4	+	A
		gBlock1	49.0	50.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	6	+	A
		gBlock1	59.0	60.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	7	+	A
		gBlock1	70.0	71.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	7	+	A
		gBlock1	81.0	82.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	8	+	A
		gBlock1	92.0	93.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	9	+	A
		gBlock1	105.0	106.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	10	+	A
		gBlock1	125.0	126.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	4	+	A
		gBlock1	136.0	137.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	5	+	A
		gBlock1	147.0	148.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	4	+	A
		gBlock1	158.0	159.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	2	+	A
		gBlock1	169.0	170.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	8	+	A
		gBlock1	180.0	181.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	7	+	A
		gBlock1	193.0	194.0	09cb072c-1680-4ed3-a3f9-110a7b07f83a	7	+	A

	B) genome-space (or transcriptome space) aggregated predictions: it will be named as {bam_file_path}.genome_space.tsv. It is a tabular file, the per-read prediction are grouped by stranded genomic position and the number and frequency of mapped inosines where stored. Each row, representing a given reference coordinate, has 6 columns: region, start (0-based), strand, inosine_count, depth, inosine_frequency. Here an exemple table:

		region	start	strand	I_count	depth	I_freq
		gBlock2	140.0	+	2	21	0.09523809523809523
		gBlock2	147.0	+	1	21	0.047619047619047616
		gBlock2	148.0	+	19	24	0.7916666666666666
		gBlock2	150.0	+	2	23	0.08695652173913043
		gBlock2	154.0	+	1	23	0.043478260869565216
		gBlock2	156.0	+	1	23	0.043478260869565216
		gBlock2	157.0	+	18	23	0.782608695652174
		gBlock2	159.0	+	1	23	0.043478260869565216

## **Basic Usage**:
All the provided models work only for the Nanopore libraries produced by ONT SQK-RNA001 and SQK-RNA002 kits. Even if hundreds if not thousands of dRNA runs with these old chemistry are available and can be re-basecalled using NanoSpeech, we are committed to make the upgrade with the newest 004 pore.

Research Purpose Only.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
