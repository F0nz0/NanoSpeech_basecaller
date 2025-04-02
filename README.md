# NanoSpeech basecaller

<p align="center">
<img src="https://github.com/F0nz0/NanoSpeech_basecaller/blob/master/OIG4.LacWFMr4ivi0OKPCIuqq.jpg" width="300" height="300" align="center">
</p>
<p align="center"><sub><sup>*image created in Copilot by Designer using DALL·E 3 technology</sub></sup><br><br></p>



A new Basecaller software for the single-step detection of modified bases during basecalling procedures of direct-RNA raw data. NanoSpeech is based on a Transformer model with expanded dictionaries with respect to canonical-nucleotides only (A,C,G,U,I=Inosine or other additional modifications). It provides also utilities for the mapping of these modified onto a reference genome or transcriptome, providing predictions on both a per-read and a genome-space level (aggregated data).

## **Required Softwares**:
NanoSpeech has been tested on a CentOS-7 system with an NVIDIA A100-PCIE-40GB GPU (Compute Capability 8.0), CUDA 11.8, cuDNN 8.6 and TensorFlow v.2.7.0. This tool uses internally (and not) some software that should be installed preferably into a new conda enviroment. \
After the activation of the conda enviroment install the following softwares:
1) Python >= 3.7
2) Samtools >= 1.3.1
3) Minimap2 == 2.24

## **Installation**:
1) Download the source code from GitHub repository at the url https://github.com/F0nz0/NanoSpeech_basecaller:

		git clone https://github.com/F0nz0/NanoSpeech_basecaller.git

2) Create a new virtual environment (it's suggested to create, use and activate a base conda environment with all the required software):

		# create a new conda environment
        conda create --name NanoSpeech python=3.8

		# activate the conda env
		conda activate NanoSpeech

		# install samtools
		conda install -c bioconda samtools==1.3.1

		# install minimap2
		conda install -c conda-forge minimap2==2.24

		# create virtual environment inside Conda NanoSpeech env
		python3 -m venv NanoSpeech_basecaller

4) Activate the venv:
	
	    source NanoSpeech_basecaller/bin/activate

5) Upgrade pip version:
	
	    python3 -m pip install --upgrade pip

6) Install wheel package via pip:
	
	    pip install wheel

7) Install required Python packages using the requirements.txt file:

        python -m pip install -r NanoSpeech_basecaller/requirements.txt

## **Basic Usage**:
### **Single Modification version (Inosine-only)**
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

	It's important to consider at least 3 GB of GPU memory for each parallel thread. For an NVIDIA-A100 40GPU 15 threads shuold be a good amount of parallel models. Here a simple command to perform inosine-aware basecalling of fast5 from dRNA (002) runs:

		python3 NanoSpeech.py -d {fast5_folder_path} -o {fasta/fastq output file} -m {model_weigths in ./models/} -t {number of threads}

	We stronlgy suggest to use the multi-species model named *NanoSpeech_Inosine_m23M_e71_d19Mrec_SINGLE_MOD_VER.h5* within the ./models directory of this repository. NanoSpeech will produce either a fasta or fastq file where a list of indexes of adenosines with higher probability to be inonsines is added in the header of each read. For the sake of example, below a NanoSpeech basecalled read in FASTQ format:

		@dc4a56dd-b899-49c4-9fc2-4bffcbe63463 2,6,9,13,14,17,20,34,37,38,40,41,43,54,58,61,65,66,68,70,74,75,77,78,82,84,85,89,92,104,105,113,118,121,122,129,134,137,138,153,156,158,162,170,172,174,180,187,191,196,199,201,204,205,207,213,218,222,231,239,241,245,246,250,252,253,254,256,264,270,271,272,273,275,277,281,284,289,301,302,303,307,309,313,316,317,322,327,330,331,333,340,343,345,348,357,361,363,364,366,367,372,373,376,385,389,391,392,394,395,400,402,403,404,409,413,417,424,425,431,442,445,446,447,453,455,461,463,466,467,471,472,479,485,486,491,497,502,504,508,511,512,517,522,525,526,528,535,538,540,543,552,556,558,559,561,562,567,569,570,571,576,580,584,591,594,595,596,602,604,610,612,615,616,620,621,632,641
		ACAATTACCACCCAAATAACAAAACACCCACCCTACTAATAATACACAATATTAACCTACCACCAAACAAATTTAACAATCCATAAATCAACACATTATTCACTAAAACTATTAATCCACAAAACTATTAATCCACAAACACATCCTTTCTTTACTAAAATCACCCATTTACACATCTCCACTCACTATTTAATTCACCATAATAATAAAATAATCCAATATATTCCTTTCAAATAAACATACATAACATACAAAAATCTCTCAATATCAAAAACACAAACACTACAATAATTTTTCTATTAAAATTAAACTTAAAAACACCAATACACTAACATATACCAATACACAATATCAACAACTTACAACAATTACAACAATATCAACAACTTACAACAATTACACAAACACTAATTAATTACTATCCAAATATCACTAAAAACATAACAAACCATTATAATATTACACCAAATCAAAAAACCAATCCCAATCCTACATTTAAATTAAACTTAAAAACACCAATACACTAACATATACCAATACACAATATCAACAACTTACAACAATTACACAAACACTAATTAATTACTATCTAACAAACCATTATAATATTACACCAAATCAAAGCCTCATCCATTCCTACCAACCCTCCTA
		+
		#)*++,+++++++++++++++++++++++++++++++*+++++++++++++++++++++++++++++++++++++++++++++++++*+++++++++***++*')(#%&#'+**()+*++('$"%+(+,++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++,($"&+%+,++++++++++++++++++++++++++++++++++++++++++*+++++*++++++++++)*+++++*++++*++++++++++++++++*+++*+)*+**(*++*+(&)**++&*)(')+(*''&'%'+*+(+("$#%+*+++++++++++++++++++++++++++++++++++++++++***+++*)*++++*+++*))**+*)*+*)*+++++++(*+**+++*++*))+++++))+++)++(++'+++*+()+)*)"##(%#+++++++++++*+++++++++++++++++*++++++*++++++++*+*++++***++++++')++++(((*++)*+++++**+)&""#$(**+++++++++++++**++*$'*+())")('"##%"#$%%)*$%)(&#'#%##$$&

3) Alignments of fasta/fastq files.

	Mapping of the basecalled reads via minimap2 using specific presets to store metadata related each inosine reference positions. The alignment step can be executed using either a reference genome or a reference transcriptome. 
	For a reference genome we suggest to use this command:
		
		minimap2 -ax splice -uf -k 14 --secondary=no --MD {ref_path} {out_filepath} > {out_filepath}.sam

	For a reference transcriptome insteat use this alternative command:

		minimap2 -ax map-ont -k 14 --secondary=no --MD {ref_path} {out_filepath} > {out_filepath}.sam

	Then filter the produced SAM files, converto to sorted and indexed BAM files. The plain-text alignment file can be finally removed:

		samtools view -b -F 2304 {out_filepath}.sam | samtools sort -O BAM > {out_filepath}.bam
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

### **Multi Modification version**
Additionally, a more complex set of scripts is released in order to handle NanoSpeech models trained on multiple modifications. For every model in h5 format, a configuration file (models/*.h5.cfg) is coupled which contains parameters to initialize the transformer model. The main script is NanoSpeech_multi_mod.py and it accepts the following options:

		python3 NanoSpeech_multi_mod.py \
		usage: NanoSpeech_multi_mod.py [-h] -d FAST5_FOLDERPATH -o OUT_FILEPATH -m MODEL_WEIGTHS [-t THREADS_N] [-c CLIP_OUTLIERS] [-u PRINT_GPU_MEMORY] [-r PRINT_READ_NAME] [-nr N_READS_TO_PROCESS]
							[-fl FAST5LIST_FILEPATH] [-rl READSLIST_FILEPATH] [-cl CHUNKS_LEN] [-idxs PRINT_CHUNKS_IDXS]

		NanoSpeech basecaller.py v. 0.0.1

		optional arguments:
		-h, --help            show this help message and exit
		-d FAST5_FOLDERPATH, --fast5_folderpath FAST5_FOLDERPATH
								--fast5_folderpath: a <str> with the fullpath for the input fast5 folderpath.
		-o OUT_FILEPATH, --out_filepath OUT_FILEPATH
								--out_filepath: a <str> with the fullpath for the output fasta/fastq file generated during the basecalling.
		-m MODEL_WEIGTHS, --model_weigths MODEL_WEIGTHS
								--model_weigths: a <str> with the fullpaht for the h5 file containing the weights to inizialize pretrained transformer model.
		-t THREADS_N, --threads_n THREADS_N
								--threads_n: a <int> indicating the number of basecaller workers.
		-c CLIP_OUTLIERS, --clip_outliers CLIP_OUTLIERS
								--clip_outliers: a <str> indicating the min-max currents value to be clipped with mean. [None]
		-u PRINT_GPU_MEMORY, --print_gpu_memory PRINT_GPU_MEMORY
								--print_gpu_memory: <str> Set to True to print gpu usage for every worker starting a new read (experimental). [False]
		-r PRINT_READ_NAME, --print_read_name PRINT_READ_NAME
								--print_read_name: <bool> Set to True to let producer to print reads names added to queue. [False]
		-nr N_READS_TO_PROCESS, --n_reads_to_process N_READS_TO_PROCESS
								--n_reads_to_process: <int> Numer of reads to limit basecalling. [None]
		-fl FAST5LIST_FILEPATH, --fast5list_filepath FAST5LIST_FILEPATH
								--fast5list_filepath: <str> Fullpath for a file with list of paths to fast5 files to limit basecalling on. [None]
		-rl READSLIST_FILEPATH, --readslist_filepath READSLIST_FILEPATH
								--readslist_filepath: <str> Fullpath for a file with list of reads ids to limit basecalling on. [None]
		-cl CHUNKS_LEN, --chunks_len CHUNKS_LEN
								--chunks_len: <int> Chunks lenght the generator will be output from raw signals. [None]
		-idxs PRINT_CHUNKS_IDXS, --print_chunks_idxs PRINT_CHUNKS_IDXS
								--print_chunks_idxs: <bool> Set to True to print 0-based index for the end of chunks in the + line in fastq output [None]
 
 Here an example command to lauch the basecalling of fast5 reads (pod5 reads have to be converted into fast5 files, see https://pod5-file-format.readthedocs.io/en/0.1.21/docs/tools.html#pod5-convert-fast5) using the NanoSpeech multi-mod version. The configurations will be automatically loaded from the corresponding *.h5.cfg file:

	python3 NanoSpeech_multi_mod.py \
		-m {h5 model-path} \ ### please select a model coupled with a *.h5.cfg file ### for the multi-species only inosines use *NanoSpeech_Inosine_m23M_e71_d19Mrec.h5*
		-d {fast5 folder-path} \
		-t {number of parallel models} \
		-o {fasta/fastq output file}

Also with this version, fasta/q files will contain indexes for all the modified bases within the head of each read. In this case a list of modified bases for each read will be provided if NanoSpeech will detect at least one modified nucleotide. Here a small example for a model calling at the same time inosine (I, stored as I_A, AlternativeBase_CanonicalBase) and m6A (M, M_A):

	@7e4f945e-a1e3-497c-80a6-098d026708ac I_A=2531,2540,2549,2562,2571,2580,2589 M_A=47,53,64,65,72,75,77,88,94,97,99,110,117,119,126,130,133,140,142,207,211,212,213,219,223,224,229,230,233,247,250,251,254,257,258,264,265,266,267,268,269,270,273,276,285,295,297,304,308,309,311,312,318,319,324,325,326,340,341,342,345,357,368,375,376,379,381,385,386,387,388,390,392,393,395,399,400,410,412,425,426,433,438,443,451,456,457,465,472,474,480,483,484,485,486,490,492,494,521,531,533,544,548,549,552,554,555,557,558,575,583,584,588,590,591,596,598,599,600,601,602,603,611,613,614,618,619,620,626,645,647,648,652,658,675,676,681,684,689,692,693,697,698,701,702,703,704,705,708,710,713,714,715,716,717,719,735,736,741,744,745,748,749,750,751,752,759,764,770,772,773,776,777,780,783,784,796,799,801,802,803,807,819,822,830,837,838,841,843,849,856,859,861,864,867,868,871,872,875,886,887,893,897,898,899,902,909,912,913,914,916,917,924,925,933,935,936,937,939,940,945,946,948,960,965,967,971,972,974,978,979,982,989,990,1002,1009,1010,1021,1025,1029,1030,1032,1035,1041,1048,1051,1056,1058,1064,1065,1067,1068,1070,1075,1077,1079,1081,1094,1095,1097,1104,1109,1110,1111,1121,1124,1130,1131,1132,1133,1138,1141,1144,1149,1173,1174,1176,1185,1186,1188,1190,1195,1200,1203,1205,1233,1238,1240,1243,1245,1253,1257,1259,1262,1271,1272,1273,1281,1287,1290,1291,1296,1297,1299,1300,1301,1302,1309,1311,1312,1313,1314,1328,1332,1334,1335,1348,1356,1362,1367,1374,1378,1381,1382,1426,1443,1444,1450,1451,1452,1455,1460,1463,1467,1472,1473,1474,1480,1485,1490,1500,1512,1514,1518,1525,1527,1534,1537,1539,1545,1547,1548,1549,1562,1572,1575,1576,1582,1583,1587,1589,1592,1594,1596,1600,1601,1607,1609,1610,1612,1615,1618,1619,1631,1635,1643,1648,1653,1654,1660,1664,1672,1677,1682,1683,1685,1693,1694,1695,1704,1706,1710,1713,1714,1716,1724,1725,1726,1728,1735,1741,1743,1747,1750,1754,1756,1759,1761,1764,1767,1770,1771,1773,1775,1777,1783,1785,1791,1792,1797,1798,1803,1806,1809,1812,1816,1824,1831,1841,1844,1847,1856,1858,1872,1876,1878,1879,1885,1891,1893,1895,1898,1904,1910,1911,1914,1915,1916,1918,1921,1925,1932,1933,1934,1939,1941,1943,1948,1950,1951,1954,1956,1959,1960,1963,1975,1982,1986,1987,1996,2004,2005,2007,2013,2014,2016,2018,2021,2022,2034,2036,2043,2046,2053,2054,2055,2059,2060,2061,2064,2068,2071,2077,2082,2085,2087,2088,2089,2095,2100,2103,2108,2110,2111,2112,2118,2131,2132,2133,2135,2149,2151,2153,2154,2156,2161,2162,2169,2171,2178,2179,2181,2183,2186,2187,2190,2192,2194,2229,2239,2242,2243,2246,2252,2258,2268,2271,2275,2284,2342,2344,2349,2351,2352,2353,2370,2375,2379,2381,2386,2392,2395,2398,2402,2403,2406,2408,2411,2414,2416,2418,2422,2423,2427,2435,2442,2447,2452,2454,2456,2458,2459,2460,2465,2466,2470,2481,2594,2600,2603
	AGCGGATTATGGGGGCGCCATTATCGCGCGCGTGCTACCCTGTGTCCACGTCGAGCTGTTGCGTAAGTCGTCAGCATACGTCCCTTGGACCCTCAGCATACGTCCCTTGGACCGTGCATATTCCTCAGTCACTACTTTCTAGACAGTAACTCGCTGTCTCGCTCTGTGAGAGCAATAGCGCCTCGAGCTAACTAAATCTGTGCCTCGAGCTAAATGCCGACGCAAGGGGAATCATGGCTGCCCCGTTAGGAAGCAGCAAGCCCTAAAAAAACGAGCATGGCTGTCATCGCTTGGGAGATTCGGGACTTAACAACGGCTAAGGGTAAAGGTTGGCCGTGTTAAACCAGTTGCTCGGTCATGTCTCCGCTACCTGTGAATCATATCGAAAACATAATAGCTAACTTGCCCGCAGATGGGCCGGTGTGAAGCTGGTAGCCGACCCCACGGCCTTACCGGAACGCCCGGACCCCCGACACTCGTAGTAAAATTGATACATCGCGTTCGTGGGGCTGGCGCGTGCGACGGTTCTTGACATGTGTGGGTTATGTAAGTATAACAACCCTTGCGGCGTTGGCACGGCCCTAAGCTATAAGCTGAGAAAAAAGGTGCTCATAACTCAAACTTCGACGTTCCCCGTGGGTGGCGATAAGGCATTGGTATGGTTTGGTCTTCTGCAAGTCGAGGATGTGATCAATGTAACGAAAAATGATAGTAAAAAGATCCGGCTGTCTGTGGAAGTCGATGAATCAAAAATCTTGGACCCTATCTTCACAACGAACGAGTAATCCTGCGTGGGACTACAAAGTCATTTTTTCGTGGACGAGCCCTCGACCTCCCAAGCATATGGGGACGTCGTAGTAGACTATCAACTAACTATTGTGGCTCGAAGCGCTACCCAAATCAGGTTCGAGCAAATAATGTGCTAACGTCTCGAGAAACAACCCGAACACCCTGTCGTTCATTGGATACGTAACATTCAATCACCTTGCAACTGCTCGCGGTATGGTTCAACGTGGTTGTTAGGCAGCGAACACCACCCGGAGCGTTTATCACGCCATATCCTTAACAATAGCCTAGAGAGAGTCTGTTGGTCTAACACTGGCCACTCGAAAGGCCTGCCGACGATGCGTAAAAGTGGATTAGCACCGCATCCCCCTCTCCGTTCTGGCCGTTAATATGTGTCGGAACAGACCGGATTTCATCACACGGCCTTTCCGTGTCCGTGCTGGGGTGAGGGCATACGAGATGCCTGGAGGTATACTACCTTGGGTAAATGCTGCGACCCGTAGGAAGCGGAATAAAATGTCCCATAAAATGTCCCCGGTCTCATGTACAATGTTCGCGCGCGAGTGCTCTATGCTCAGCTCATCCGTCACTGACTAAGGCCGTGGTCCTACGCGTAGCCAAAGGTACATATACAGCGTCTAGTGTGCGCTTCTCCTGAATTGTCAAATGAGCCGACGATGTATTCCAAAGGCGGAGGCCACTCCACGTTCCTGTACGTTGGGGTGGATATGCATTTCTTAGACCTGCTATTATACGCGTATAAACGCCCGTCGTCGAGGTCGCGTTACTAATGGTGAAGTGATACCACACACCTAATTTGTAGAACATCATTAACTCCTCTGGTCAGGGATGCTTCCACTGGACTCTAAGGGTCAGGGATGCTTCCACTGGACTCTAAGACTGTTTGAAAGGTGGCCTATAGTTACGAATAGCTCTGTAAAGAGCTTCGATGGCCAGATTTATTAGGTACACCATAGGACCAGTAACATATACCGCGACATCGGTAAGTGTAATGCCACGATCACGATCTAGGGTGTGAGCGCCTAGGTGCTTTCATCACCAGGTCTGGGAGAGCCTGCCCTTCGTAGTCAGAACCCTCATCGGCAGACACTAGTCCGATCGCCAAGTAAACATGACGTAGCCCCGAAACGTTATATAGGCTAGAAGTATATGAAGGAGGGTTCCCGGCACTGCCGAGTCAATTCCTCCTATTTTTTCAAGAGGCTGAAGACAGGAACCTGGTTTTGTAGATGGTCGACTAGCTTTTAAACTGAAACCAGCTAGGAGGTTTAGCGTACGATAAATTTCGATTCCAGGACTCGATAAATTTCGATTCCGGTTTCTCAAACAGTGGTGCCTCGGGACACAATACGGCAATCTGTGACAGGGTCCAACACAGCAACCAGACATGGCTAGTAAGATTAAATATATCAGAGCGGGGCCAGCCTTCCCCAGTAACGAGTTTCAGGCCCACTTCGCTTTAGTACGGAGTCGTCGTACGCGTGGGCGTGAAAATCATCGAGGATGTGTGGATCTAGTTGGGACGTCCAGGGGCTACAGCCCAGAAATGGGTGGGGCGGCCCGACTTGAGGCACATGTTATCGCTATGAGTACCGAAGGAGAGGATGACATACTTAATTGAGTGGTTGACGGGCCATGGTAGGTCATAGACAAATTCTAATTGAGTGGGTTCTCAGCATGGTAGGTCAAAGCCAAACAGGATGCCGGACAGACAACCGAAACCTATAATTAATATCACAAACACAATACAAACCTATAATTAATATCACAAACACAATATACATATCATGTCTATTA
	+
	#%$%$%+%+-))*1.'.,*,+.'(0*--+-)-*-+(,/.%&#$*%#'%&(&+-*+)*).(+**)+,+&**(,*))$&)((,&##&&,$($$"#$#&+',,&&,+/,(+)(*+)+,++**+*,**,+,*++))-,++)&/.*,(%"")(+'-)*',&-*&+'***%(&*-')+++*)-&&%)(+%+,((%)("#*("#%#%*"#$'11'+-),,+*,+.-+-+,**/*,-))+-'*"###""'$$&$&()$**',))+$&"$&#")**)*+$)(&-,+++-+&(&'%###'&($+$)/&,'+-,)**'&+)/*)("#$*&)0,-)+-(-*)(++,/--)'))&%'.*--),).--)+-,+,*+,,),/+,'(*+',$#(''))))+,.-*),++,,+++**+++,,,*(,,+,)+-+--(%$'%"$*%,.&)'**+--,+,--,,-+***)-*,'+#$#0%'0,,')/+,/*,+-)+*-,)*+(+++*),,,*+,++*,('$##,%-**+.)**/(()+*,,%(.*)*+*),*+)**))$&#&$*,),&-,-'**,+,,-)++(+*,*,,,++),-)+,.,.*,('%)*"#&*)(+%+&,)**,&,((,,+,+*+,)(',,(,)'+)+*,,+/,-)+.*+'(-(/'&&&#)(*),+-**',,-+-+2,--,,(++,+,*,,,#/#*-+&(*++,*('***++**)%%$))0-1*--.--,,,**,++#'##,**+,)+-,*+*+++++)+,,,,(-+)..+**+++$%##$%-,,,+*+,*,+.*+,+,*+,,*,,*+.++-,,--+,(&%&)&%'*)),')(+-,,*,,+)-,*-...+,,+-,**+,,)+,(()"#%&##$*0+&+*(),/#+,*)*),+-*-.*)(,#+1*%.-.*,,(+*(-+)++--+,***++*,&,(%'#&#)*,(,**+,*++*+,***+**++**+++*+**+,,.*,-,*.,*+%($#+$((%,-(.+),,+(++*+)+,+'+-)*-)*+++)-)*,-).,**))+',)*&%$#%'##&*+&)++,)*+,**..-.,,,,)+,++,*)($#"%#$"$""+&+*++0+.+&%(()%,(,%)'$/#)%"%("!$%&'&###$#$%$####$,),,,-)*--,+)++,+*+,,*,*,+,*+*+*)+$%##%**).+.(+,),,'.,**,**,*,+--(-*,+,++,)*/*)**',&(/(''%##%+()-,(-,--*,&+--*+,*++*,+,--,+%))&"&&&''$&+'*/.-)**,,+.,-,,+*+,++-+,++*+/+,,)*)*)#%)#%%&$#**+,-.+**+****,*,),*)*)+*,+(*)**,)'"%&--(*'(+/(*,+*+,**++(++**.+')**,&&$"$'$#)#%)*#"%"#)"#"$"'(&$&""%$("."%##$"*$)##$'*)*+*)(+-++++-*,++-,/-++,-,,,)+'%%.2+$(0+*,+*,),,+,+*-.++,,*,,--*.++))**)++.))+'#&#*$#%''")&#,),+(**.)).(*(*-,,++**+(+*+&)(+)&#%,$%/-+*).&.,)*(*('*+)*-(/**)**+-+).**)***'%%"#%-$&"$*#'$#"'"(#&&+'$,&&,*)&)(*(()"(&$)'*%((%&'#&'-&#%+"&&(&$+,+**+(++,-*++)+,*,+*+,+-*$#(+$"+.,-,,,,,+,*,,,*)+,*-)+*+-*)*),,+****'$$/'-,,,,*,,,,*,,+,+*+,).-*+)+*+)')$*#,,**(*,+-*,)*+*)+++**+)**++)),-+(+)$#$*+'0(*,---+-+,-+*,*-+*,+(+*+''%$"##)))++(')*,,+-*-+,,,--)+*,-+)*).+,,**,,')#$&')+*++'),+,+),*-*)),).+(,,,+,),)***)+**(,)#%#')..',-**-)+,,+,+,++)(*+,**,,("#-0).(-,),-,,--++-*))++*+*(+*'#%%)'$%%)#'((*--+++,-++++,-+*+-++,)#,,-+%,+,*.,-++-,,+.+*++,++-,,*%#,%*-(',.+-+.-+.+-/,,**%-(*)+')(+&&'&+%#&%%#%($&&+))'('+,)**++)&,*+,,*%'#$,"(').+'.),-,,*++*,+/-/-*+)+*+0(,&"%&)+/'))-,-,&,/'+,,-,--*,(/#'$&.()+.,,,++*++*+))-)++,)*'%'$()+.%,*.---**-)**$"$$)),,++,,+,+,*+,**+,++++,*+,++-)+,(+)"#$'$%#(+&'$*'&(-('',,(,+(*(+&((&'*',*(&#"'$)*##"$#,'#+#&+')(,*-))++,,-*(++*+,.+****('$$,+&,*+,-*)*-,)+,.,.+.,.+')+***,#(.$),*+*+-,,,+(*,*-+.*,,&$'&&($+*+*+++-,()-)+)**+*+,(+(%'')%("$&'*#&*'#("#"""#"##))(&(#($""%%-)%,)%,+%#1&#%-$))0%&&$&$(,%"##("--&).+,),)*'$)-)'*),)&+""#"%#%((-'*)*,(,&'*&(,&%*&)-%/$*#)"''"$**#$(*($)
	@c39c218d-e965-42ae-8350-e1dc20718bce I_A=2733,2742,2751,2762,2771 M_A=0,4,16,23,24,27,28,29,30,72,74,80,89,99,107,108,116,119,126,128,131,133,137,138,142,143,147,148,149,150,153,155,158,159,160,163,165,166,168,173,177,183,185,186,188,193,204,205,208,213,226,242,243,244,247,250,254,266,269,272,276,281,283,288,289,295,297,298,299,304,307,312,316,317,323,327,366,375,381,383,384,439,444,445,446,449,457,462,482,483,487,498,505,507,509,511,516,522,524,526,527,529,535,536,566,570,573,587,588,593,597,598,599,607,608,652,661,665,666,668,676,681,685,688,689,692,693,699,700,701,704,706,723,731,732,735,736,737,741,744,748,751,755,763,765,766,768,778,780,786,793,794,795,797,798,799,801,804,808,809,811,815,816,825,828,832,835,845,851,852,857,859,863,868,875,886,896,898,901,904,906,907,908,909,910,911,924,964,968,980,982,993,997,1002,1008,1009,1015,1023,1040,1044,1046,1050,1055,1063,1070,1071,1072,1073,1074,1076,1077,1081,1084,1087,1094,1101,1103,1110,1112,1113,1117,1118,1124,1128,1135,1141,1145,1149,1157,1159,1165,1211,1212,1213,1214,1217,1221,1224,1226,1230,1237,1243,1248,1253,1258,1263,1266,1268,1271,1272,1274,1311,1312,1317,1318,1320,1321,1340,1342,1343,1344,1347,1353,1354,1355,1360,1361,1365,1369,1378,1379,1381,1382,1394,1395,1398,1402,1412,1414,1419,1428,1429,1430,1433,1437,1445,1449,1455,1457,1458,1463,1464,1465,1469,1472,1474,1477,1479,1481,1482,1486,1487,1491,1515,1528,1542,1546,1552,1557,1562,1563,1568,1573,1576,1579,1582,1586,1590,1592,1603,1608,1611,1616,1620,1634,1641,1647,1648,1649,1655,1662,1663,1666,1667,1674,1677,1690,1691,1735,1737,1739,1741,1745,1749,1751,1754,1755,1756,1757,1766,1779,1780,1787,1823,1836,1839,1842,1850,1853,1854,1859,1864,1870,1874,1879,1939,1952,1961,1968,1974,1980,1982,1987,1995,1998,1999,2001,2010,2011,2014,2016,2018,2031,2034,2035,2039,2055,2062,2063,2066,2070,2073,2076,2082,2083,2085,2092,2093,2095,2097,2099,2100,2102,2106,2107,2111,2116,2117,2118,2143,2161,2163,2164,2172,2186,2187,2189,2190,2200,2205,2212,2222,2224,2225,2232,2233,2238,2240,2245,2247,2249,2258,2266,2268,2269,2270,2273,2274,2275,2278,2283,2287,2307,2308,2309,2323,2333,2334,2337,2418,2419,2421,2423,2428,2431,2436,2439,2440,2468,2469,2471,2472,2474,2480,2485,2488,2490,2495,2496,2498,2500,2501,2502,2503,2506,2510,2521,2523,2528,2530,2538,2539,2540,2545,2549,2570,2574,2579,2580,2583,2585,2595,2597,2598,2600,2602,2603,2604,2609,2610,2611,2623,2624,2626,2629,2631,2633,2637,2638,2645,2648,2650,2651,2655,2659,2687,2782,2786,2791,2795,2797,2799,2803,2808,2810,2811,2813,2815,2816,2817
	AGTTAGTGGCGCTGGCATGTGGTAACCAAAATCAAAACGGTCGTGATGATGCGTTCTCTTCCGCATGTCTCCATATGCTCATCCTCCCGATGTCTCCTCACCTGCGTAAGTTGCGGAGCACGTTGCACACTAGATTGAATTCAATGCAAAAGCAGACGAAAGGATAAGAGTGGAGGGATCTCTATAACACTGCACGGTTTCTCGAATGATCGTAGCCGGCCGCGCCAGTTGTCTCTCCTGCCAAACGATTATGTAGGGCTCTCGGCAGGATCACTTATCTGAGACGCGAAGTTCCATAAATTTTAGGATTCGACGCAATTGGTACGCACGGGGTTTTTGAATGTAACTGTCCGGACTGGTGGAGTGAGTTGCGTCAGTCGCACAAGGGCGCGCGGACCTTTTTGTTTCCTTTAACGGCGTCTACTATGGCGTGACCCCTAGCCTAAACCATGCGCCCATGTCAGCGCTCTTGTGGTGTGGGCAACGGACGCCGCCTCTACGCCTTACATACAGTTCAGCCGTACATAACAGCTGGAACGGGTTTAAGTAGTAGCTATCCAGGACTTATCCAGGACTTTCGCCTGTGCAAGCGCATTGAAACTTGTCGAACGAACTTCAGATAACCCCGTGGCAGTGCCGGGAACTGGCGGCGACTGCGCGCACTCAACATTGTTGTACTGCACGTACCAATGAACCGTGAAAGTACAGTCCCCAGTTCTCCCTACGCTTGTAAGGAAACCGACCACGGATCATCGAGTTCTTGATAATACTGTGTTCTAGAGGTGTATTCTTCAAAGAAAGACCATCCAATAGGGAAGCTTGCTTATGACTCAGTACTCCGCGGGATCTGGAACCCTACATGGATGTGACTTCCTAGCGCGTGTCCACCGGCGGTTAGAGCATTAGAAAAAACTCGTGCGGGCGATTTGCAGTCACATCTGCTTGGTTATTCGGCGCAGTTTTTACTCACGGTTCTGCGGATAGTCGGTTGCCATCTATCTCATCTCCAAGGCCGATTGCGGCATTCGTCCTGGGCGTTGATGGAGACCCATCGCATGGCTGCAGCGCTTAAAAACAACGGAGGATGATGCCGTAGGCTGTATAGGCTGTAGAATGCAAGCGCGAGCGATCGTTGATCGCTACCGATCCAGTCTTCCATAGTTGGATTCTGCTAGTAGCGTAGCGGATAATATGCGGCACGCGTGGGCGTGAAAATCATCGAGGATATCTAGTTGGGACGTTGACTCCAGCTTACTCCAGCTTATGAGAGGAATATAGGAATATTAGTAGCAATAAAGCGCGCGCGCGGGCAACCGGAACAAGGCCAATTGCTCGGTCTCATAAAGGACTGGCAAAGGGTAAGGGACTGAGTGCGGTTAAGAACTGTCGTTCTTAACCACGTATCTCCTCCTAGATCCTATTTTTTGTAAATTAGTGATTGCGCTACGGACCCTTACAACCTGAAATTGAGGATATGAGATAATCGAACTGATCTCGGGCCTCGGGCGTCCCGCTAGCCCGGTTTCGTACCGGGTTTGGTTGACTTATGTCCATTGTACTTCAACGGTATCCCACCACTACGACGTAGCGACATCTGTTTGGTATGGCATCACTTGATCCAGGGGTTCTTGGGTATGCTCGACTCCGAAATGTTGACTTGGGAATGAATGGTTTAGTAGGTTTGTCTGCGAACAGATGGCTTGCAAGCACTGCGCAGCCGAAGACAATGTACCGTAGACAGACTGACGCACACCAAAATTTTGCTCACTCGTGGCTTTGAATTCGCGAGGGTATTCTGGTACTTACTAAACGAAAAGACAGGGACGGGCCCTGTCGACGACGAGCTGCCGACTAAGCTCACTTTACTTTCAGGGACTCGAAATGGTTCCCTTAAGCGTCTAAGTAGCCCGTACAGTCTGCGCCCTTGCGCCCTTGCTCCACTTGGTTTCTGGATGCCCTTTACGCGGGACGTCCAGGTCCAGATGCCACTCGGTGACGAAGAGTTTGGCGAATGACATACGTTGCGTTTTGATTAAGGCATCGCTCGCTTGTCTCAGTGCTGAAGGACCTACCACCACCCTCAAGATCTCCTAATACACAAGACTCAACTCAGCGCAAAGGTCCGGTTGGCTGCTCCCCTTTCACCCGGTCCCCTCGTGGCATAAGTTGTCGACCGCCCTTGGTTTAATAATTGGTCGCCACGTGAGCTTCTACCTTCCCCGAGAATGGGCTAACCCTATAGCTGATACATCGCTGGGAGCGGCCGAGAAACTAAATCAGGGTATTTAGCAGCCCTCAATGTTTTTGAAATCCGTGCTCCCTTAGCCTGCGGGAACGACCCGGTAATGGCCGTTCTGGAGTTAGTTAACCAGACATCAGCCATGTCGCACTAATTTGCTCGCCGTTGACATTATGCAGAACAGAGGGCAGTACGCCAGGAATCGAGCAATAATATGCCTACCTTTGACAATAATATGCCTACCTTATTATAGTCTAACAGAAAAGTACCTAGCGTGGTGGCAGACGCGATAGGCCTGTAAAGGCCATGGAGGGCCGATGGTAGGGGCGCAATCTACGGGAAGTATACGTTCGCGGAGAATACAAATCTTAAATTTTCGTGGGGAATATCACACATGGAACGTGTCACCATAAGCCATGGAGGCCGTTTCGTGGGCCGGTTACTAACAAAAAAGCCAAACAGGATGCCGGACAGACAACCGAACAAGACAACCTATAATTAATATCACAAACACAATAATTACATATCATCAAACCTCATGTAAGCCATGGTAGGCATAGAGCGAGCGCAGAAGATAAAG
	+
	%*&&$.11)++(+,**+&+*).-+)''(%$#"#$**%.-+,+/+)+-.,-*.--+++-*.**###&#$."#$#%-#$#'-&"##$#*#$%#%*'$#%%#%###,$&%%$)''*%+"&""#"$-5.,&)+-1.,+**-'&-7+.,++,/,,++,*,***+,*--+,,))))-"##)$$&%"%""#&&#&&&$""%","&"#+#&$*'#(($$$#&#($/#,&)(&-*().-*&,,+)+%0',0*,,+.*,-,--*+--),/.,.++*+)+*'$$&)')/.+-,,,**+,*,*,+-*,+)*#+1,,%,,+,,+*,+.,+-,-+/)+--)+),,,)%$$,-'*.&&%(%*,(*())*+&#)('(,')$#%-1(--)*.+/,++),,+*+*)*$##$-(,0(*,*,,.)+,+-+++++,,++-*,),***(,,--+*,--/(##-(++)('***&*,),+*+-(+***)*,,,+)%$&#'**.((+#,(*--*+-*+*+,*+**,,$%"#+)##&+$$+(,-&(,**++**,,,),,*,+)()*"%$0%),*'-')+,%%%*,*&+(.(%%%+)-*,+,,--*-+,-,+),*+++--*++**,+)+++,*)($"*+0+-,-,-*+--,+,),+-+**,,-,+-,-,+-)'%&#$'**)+.,*-,.++*,,,-+*,*+++,*+'#+(/-,,-(--,+,,**)+*++++)++-.*(,%"#%"$##$)#%$***-,('*+.*))*)**+**,-(,++'"#&+./+/)-/-,.*+*,-*-/*.,,---+,+,,.,()&(,+-,,+*++,,+)+,+,*+**%*'$%$%',+.*,//+++*,,+,-,))++,,-.*(,)'(%#.+*0)+*)(+),*)++,,)*--,)*-**)+(-,*,)-))($#'++/&).(+++,++,,*,**(+-()*+++*)$%'.(+*.,-,).--*+***+*++*-++),)*,'#&,($(+*/-.*(+*),)****+-+,,)+-,.*--+)-)%#+',,),.*,--.+++-,+*),*)+*,-++*$$((*0+,-'')..-++*)+,.*-+,,*,+*,)-%&0,+*(,++(,.,(-,++,*,*++,))+'%&&$#(%$&.+.0%**.,+**---,.),)+/+(+*)'%$)$,0,**+-+.,,,,,,*+),,,+-,,,*./*+&+*,#"'.*+0#+),-*+,+++*)++0,*+,'&&)"&+)$-++,$+)+)--*+**,+&#$()%,)+&-,,+*)+*,+*()(,+,+*(&&*)))#&))/*,)*%)#$%$)"+#-*+,++,-.,,-++-+++,,-(*#'#"-)'-++(-*-,*),,,*-*,""$%.#$,("%#$#$#%($)'&/&)'-($"(%#$%$$'&.**(,-)**+-+-**-+*-+*,*,+#+++,-+),,-++++*)*-*)*,++*+*)*'$#%,%0(*-,--.,*-+,--,+,)*+,-+.,,+,,++,+,*(()$##+&#"$$)'(*())+*(**)&)**)')(%)&)%$$+*,--+,,,+/+(&*),+,+,,,+,+)****+*)&'&),$#(,+++--)++-++*+*+++-+*)-)++*,*#$##&&),,**),),*-**+**+*+-,,+**,-,*++-+*(*)+(+**+$)$)#(-#()#%).+(')'))-+*0*+,*,&)*)*##.,1'*-,)+/*$,,-,,++/-+-*,+.+-''*'$$(-3).(--),)**)++(,*)-+*-,,+**)-,*,,&&%"$%*-+--*'*+,*-(.)-+*+++$#$)0().-,&*+.)+.-.***,,,+,++,+)#"&%)"#""*#%#&(-,#(%.&$(+,(%+),'."##%)-0'0-+-.++,---0*,,-,+,,,+0)+)$'%'%&'++++*+)*,,++***),+#%+)),,+.+--+,,+-(+)*++,++-,,-),)).,+,+-.++)&$%)$-'+-*'(*-,*+,,*,*+),*++++,,++*+,-+*0,,-(),$$)*+*&*4-)--)1..-*,(-).*-,*)#%"(20+/*((-+.+,,+-+*+--+++&&+$(+0.--+.*,-*,,-+,,,,++*-,+)*+-),,*,+*(,,)%'&+"-,-).-/+++++),*-++,-,*-.*++*$$#(1&)+-*))**+,+-,-+$%()++(**,*)++,++*+/++-++*,***+)+**)*))$$#-),+)/)-*)+*-+,+,,***&'$(-+')+(+&)-*)*,+)),+-*)-++-*++,+*+*-,*****)*+-&"%-++(*)*)/)*)*&..'&%$#%%#-*')*)/(++,+*+++,'"&"##+&"#"'%#&#(%#"&$,,/(+,,(*.+..,+*+,-,,,+**-+*-**--*+,&%%"$&%+.$.,(')-+*,*-/,.,-+-,,)+*""#"#)$$$#$#&0+$1*/+/---),++,***+++-+,,--,)-)$#'&())*)),-,+**+++,-)(#####$'$##$%""-#%$-.'&'###"()#.*-+),.)+-*+,)+-)+,,()+*++++()$%$),(,//,*(+*-+*,,.(**.+&'%1-,+,,,+****)+,-*)((($"'.%"#$#)#$&%"####"#%'+,.)+++--.+*,*,-+*++++**++))(*&#)(,**,.,+,+*+),+,,+***''&#"%*+0*+)+)+,+/+*,*-,,)****,(&"$&'&("")%##%('&#&$$$%###%'$"""#')'*$**+*)"#"&+%+++$',()*(+$+&%$$&$#"##(($++-+.,+%+&*)#)+(&*$&&$$#""$%$',,#&#,&&#'$$$#''"$"%$&'%$#%#$##"%%&$$&$"&$"""""##"$(#"#
	@70548ef6-af7d-4f4a-b8c1-39bf24d9fc71 I_A=1616,1625,1634,1967,1976,1989 M_A=3,22,24,31,48,49,57,61,66,69,70,74,75,78,79,80,81,84,85,88,90,93,94,95,96,97,99,111,112,113,118,121,122,125,126,127,128,129,135,140,145,147,148,151,152,155,158,159,170,173,175,176,177,181,191,194,202,209,210,213,215,221,230,231,232,236,238,241,244,246,249,252,253,256,257,260,271,272,304,306,307,314,322,324,325,326,328,329,334,335,337,350,355,357,361,362,364,367,368,371,411,415,419,420,423,429,436,439,444,446,452,453,455,456,463,465,467,481,482,484,489,491,498,502,506,507,508,509,512,516,523,524,525,526,531,534,537,542,569,570,572,581,582,584,586,591,595,598,600,622,628,631,633,641,646,648,651,654,657,665,666,667,675,681,684,685,690,691,693,694,695,696,698,699,700,704,706,707,711,712,716,733,735,743,749,754,761,765,768,769,770,771,776,779,780,781,785,792,797,800,809,811,812,819,820,822,840,841,847,848,849,852,857,860,864,868,873,874,875,881,886,891,902,912,918,921,925,932,934,941,944,946,952,962,972,975,976,982,983,987,989,990,993,995,997,1001,1002,1008,1011,1014,1018,1020,1022,1024,1025,1027,1028,1030,1035,1037,1038,1042,1045,1046,1058,1061,1069,1074,1079,1080,1082,1090,1093,1094,1102,1104,1108,1111,1112,1114,1122,1123,1124,1126,1127,1135,1141,1143,1147,1150,1154,1156,1159,1161,1164,1186,1188,1190,1196,1198,1204,1205,1210,1211,1216,1219,1222,1226,1234,1241,1251,1254,1256,1265,1267,1281,1315,1316,1319,1320,1321,1323,1326,1330,1342,1343,1344,1349,1351,1353,1358,1360,1361,1364,1366,1369,1370,1375,1376,1377,1378,1390,1397,1401,1402,1411,1418,1419,1421,1427,1428,1430,1432,1433,1436,1437,1449,1451,1458,1461,1468,1469,1470,1474,1475,1478,1479,1482,1485,1489,1492,1498,1502,1505,1507,1508,1509,1515,1528,1529,1534,1536,1541,1551,1553,1565,1567,1569,1570,1572,1577,1578,1585,1588,1590,1651,1654,1655,1657,1660,1661,1662,1664,1666,1669,1671,1681,1684,1688,1698,1701,1702,1705,1713,1714,1717,1724,1730,1740,1743,1747,1753,1760,1761,1762,1763,1765,1767,1770,1775,1776,1780,1781,1782,1787,1790,1791,1793,1807,1809,1840,1845,1849,1851,1853,1858,1864,1867,1870,1874,1875,1877,1880,1882,1884,1887,1890,1892,1894,1898,1899,1903,1940,1942,1943,1948,1952,2023
	TCGACGTTCCCCGTGGGTGGCGATAGGTGGTATGGTTTGGTCTTCTGCAAGTCGCCCAGGGATGTGATCAATGTAACGAAAAGGAATGATAGTAAAAAGATCCGGCTGTGGAAAGTCGATGAATCAAAAATCTTGACCCTATCTCACAACGAACGAGTAATCCTGCGTGGACTACAAAGTCATTTTCGTGGACGAGCCCTCGACCTCCCAAGCATATGGGGACGTCGCTCAAACCCACAGTAGTAGACTATCAACTAACTATTGTGGCTCGAATGACGCTACCCAAATCAGGTTCGAGCAAATAATAATGTGCTACGTCTCGAGAAACAACCCGAACACCCCTGTCGTTCATTGGATACGTAACATCAATCACCTTGCTGCCGAGGTCAAGTGGATACTGAACTGAGTGTTAGGCAGCGAACCACCCGGAGCGTTTATCACGCCATATCCTTAACAAGCCTCCAGAGAGTCTGTTTGGTCTAACACTGTATAGCGGCGAGCGACGGAAAAGGAGGTAGTGCGTAAAAGTGGATTAGCACCGCATCCCCCCTCTCCGTTTTCTGGCCGTTAATATGTGTCGGAACAGACCGGATTCATCACACGGCCTTTCCGTGCTGGGGTGAGGGCTACGAGATGCCTGGAGGGTATACTACTACTACTTGGGTAAATGCTGCGACCCGTAGGAAGCGGAATAAAATAAATGGATAATCGAACTGATCTCGGGCGTCCCGCTAGAGTGCTCTATGCTCAGCTCATCCGTCACTGACTAAAACCTGACTAAAGCCATTGCTGAGTCCATGAGCGCTCCGACAAGTCCCTAATAGTGTGCGCGTTCTCCTGAATTGTCAAATGAGCCGACGATGGATGTATTCCAAAGGCGGAGGCCACTCCACCGTTCCTGTACGTTGGGGGATTGGGATTATGCATTTCTTAGACCTGCTATTATACGCGTATGCCCGTCGAGGTCGCGTTACTAATGGTGAAGTGATAACCACACACCTAACGTCCAGCAGTATTTAGATACAAGAATATTTGAGAACTCATTAACTCCTCTGGTCAGGATGCTTCCACTGGACTCTAAGACTGTTTGATCAAGGGGCCTATAGTTACGAATAGCTCTGTAAAGAAGCTTCGGATGGCCAGATTTATTAGGTACACCATAGGACCCATAGGACCAGTGAACATAATATACCGCGACATCGGTAAGTGTAATGCCACGATGATCTAGGGTGTGAGCGCCTAGGTGCTTTCACCAGAGGTCTGGGAGAGCCTGCCCTTCGTAGTAGTCAGAACCCTCAGACTTAGTCCGATCGCCAAGTAAACATGACGTAGCCCCGCCCCGAAACGTTATATAGGCTAGAAGTATATGAAGGTTAAAAGGGTTCCCGGCACTGCCGAGTCAATTCCTCCTATTTTTCAAGAGGCTGAAGACAAGGAACCTGGTTTGTTAGATGGTCGACTAGCTTTTAAACTGAATGAACTACCAGCTAGGAGGTTTAGCGACGATAAATTTCGATTCCGGTTTCTCAAGTTGAGAGTTGAGCTCCCCTGACAGTGGTGCCTCGACACAATACGGCAATCTGTGATTACAGGGCCAACACAGCAACCACTATTAAACCAAACACATAATACCCAACCACCTTTGGTCTCTAGTAAGATTAAATATATCAGAGCGGGGGGCAGCAGCCAGCCTTCCCCAGTAACGACGTTTGTAACGACGTTTCAGGCCCACTTCGCTTTAGTACGGAGTCGTACTTCTGAAAATAGAGTAGGCGAATCGAAATCCCACCAAGACGTGCCCGGGGCTACAAGCCCAGAAATGGGTGGGGCGGCCCGACTTACTTGAGGCACACATGTTATCGCTATGAGTACCGAAGATGAGAGAGGATGACATACTTAATTGAGTGGGTTGGTTGACGGGCCATGGTAGGTCATAGACCATAATTTTAGGTACACAAACCTGACCTATAATTAATATCACAAAAACCTATAATTAAACGTATCATCAGCCCTAACGAACTACATGAGCCAATGAT
	+
	$&,**')/--/-++,+)+-+,+)'*,)#".*&-0,*,).+,*+*++,,+++,***)),,)#$%.*)*+.-(+--*+*)+)('*+%$)*+0-*+,.-.,*,-,+,*,%++'((&#&#&'&)*+.*1-,-,,%+*##%,)'-).,+*-)+,-++--*+,,*),+++,+-.*$',)-)+-.,(,+,(+*,)++*,-,*+-+***++)'%+/0'+),,*+(+,,-++-,-,,+,,+,'*++),(**'$%*,2/,(),&,,-,*..,/&*+++*+*'&")**,*0.++0+&&*)*).-*-+,,.+*&%'##&()+&*))**-++,,++*,(*.*,-,)+)'%%#$((&*+,,*).+*.,++-*+-+++(+*)***(*,++*$""%*"(($"#-%$##&'%%*%&$#)#$)"&$#&/(-)*,+)-,,+$-)(*-,,+),%##$+.),+,+,*+++*,*,+*+),,*.+)+-+++.,+*'&'$-1*-+),,-/,,-*-++*-,-,*+,+(()(%$$$'#$%)/(,+*,,,*,,++',-*-,,++++,*+)'%#&''$$/'-+'+..,**,,+,+)**,++).,.,,,-++,+-*+,%*&#%$)))),+-)*,,)+-(,**+))+-,+(,+,-+(#,++--*-,.,**-+++++***%'(%$(&*&+()+,-++,,-+--,)++*,(*.+--**,-')*%#&)%##(,+#("''+#"*&(%#&**&((*$%'%%+&(&&('$#%,,-.(-*,**-+*-+,+,,,+,,+,+***+*,+)&#$""%'$%,'+'(+*-*+**',,*.)+/+'%$$%'($"&),*-+&+)))*-,+()$+$$#"###(.#)*--0+*+0++,,.,,*)+**,'%'&,0-+,,-*,+++()+*-+,+.-(,,)*)(''+*.,+))+,**,*,+,*,,+,+*+*'+,-+,++**++,,,+*+-,'+,*,)+'*)-')$#)$(+--+)(,,)-+-,,++++*+,-*-+*,,+*+)$,%$+,+++***+,(,,(+.+,+**+,+,++),),+*'$'--,.,).,,*+,-++++,)***,+**(+*%$(+).,++,+-+**+,(-),,**,-+-+)(++,,*(($#',)(,)+-+**+++,.,-,+,*))*+$$#$#.'*&),)*+,*,())**,+-+.*+))*)&)'%'**'$#")/+.*'.*++,,'&(%'"#+()/,,,*,*+,++++*++))+++,+,-+(,*$"#(+)+.+*--),-+).*,)+*+)*-***+**%'%#++*%(,,)+,,,,-+-+,,-,++*%#$"--*+,)*+,,-,+,-,,,+),,+()##&%$'+,)+*,++*-,*,)+--+)$&&$"%'#.+))-#,%,((+.,),*-,+'*+*&)*+&&'/(%%#%&&.*&,,'0,*-0.,+.)(#-*+0**(++(*,++-,+*++,+++,+,++,++*#((())*.-))+,)*&$+,'*.*(.,,),,-+*+,+***+)(#"#'$)&+,*-)+)-.,++++-+-++)##'&*)++),*)**,**'-(**+)()%*)')&&$#$2)+-.-++**)+++,+*.**+*,+-)#&'./+*/,.,*-+,)++-,*,+***,+&$"#$#(--&'-'*,&%""""""""$#&')"#'##$###$$$'*$&"'$##"&##&$#))0/,,-,-.+*++*+**+)++(*++**+&%%$1$&-(*+-,'++,,,+++)*++(+)),$%"#*'%'*+)0,+**++**,+,+*(,)-+*-)+*++,&##'/%)'0,,*-,++%+(+,-**/+,,*)$%%*(*--.-,*/+),,,***+,+*-,-*&&$#&#**&++*%,*,*+*)'&.,(,')(##$$$&+%"*)'(&$$,.,.,-+--+(,,+,+,*+,),++**+)+%&$#+%,-,,(+--.+,,,+,)+,+*,-'%&%&%%$-(*-'-)-+,++,),+*)*(&)""$'("$%%#($"&#"""&$#&%#"('&#$'#(*&'%($()'-%&""%#"$'&&$''%"""$'('#"#$'&,#+)&#%"%$%&#",'###%$$&&&)

These generated sequences can be mapped using minimap2 using the same command defined for the single-mod script. Once the filtered, sorted and indexed BAM file is obtained, the next step is to retrieve mapping information for every modified nucletode detected:

	python3 modification_detector.py \
		-b {BAM file} \
		-f {fasta/q } \
		-q {minimum base quality} ### minumum quality threshold to count a modified nucleotide or not during aggregation onto genome-space ###

Both per-read and aggregated predictions will contain an additional column with the detected alternative modified nucleotide.

## **Additional Information**

### **Terms and Licence**:
All the provided models work only for the Nanopore libraries produced by ONT SQK-RNA001 and SQK-RNA002 kits or RNA004 and are demonstrative prototypes. The single modification model (R9: only inosine) was trained on a bigger dataset from differnt organisms and IVT constructs and, even if it possesses high generalization capacity, it is still a prototype. We are committed to developing production models for both R9 and RNA004 pores. The 2 modification model (R9: I + m6A) and the bigger 9 modification model (R9 see models/NanoSpeech_RNA002_16classes_9mods_m50Mplus_DS4M_E54.h5.cfg file) are prototypes trained only on syntehtic IVT data. A pilot model for pore RNA004 trained only on curlcakes in-vitro transcribed synthetic molecules is provided but generalization capacities are not ensured.

Credits:
This library adapts and utilizes the model architecture and certain functionalities from an official example code on keras.io, [Transformer ASR Example] https://github.com/keras-team/keras-io/blob/master/examples/audio/transformer_asr.py which is released under the Apache 2.0 License.

Research Purpose Only.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
