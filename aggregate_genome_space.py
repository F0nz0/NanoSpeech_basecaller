import os, sys, pysam
from __init__ import __version__
import argparse
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from misc import phred_score_to_symbol
from modification_detector import retrieve_depth_stranded
from modification_detector import aggregate_genome_space
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"""NanoSpeech aggregate_genome_space.py v. {__version__}\n\nA function to extract aggregate onto genome-space dimension per-read predictions.\n.""")
    parser.add_argument("-b",
                        "--bam_filepath",
                        required=True,
                        type=str,
                        help="--bam_filepath: \t a <str> with the fullpath for the input BAM file.")
    parser.add_argument("-p",
                        "--per_read_preds",
                        required=True,
                        type=str,
                        help="--per_read_preds: \t a <str> with the fullpath for the per-read predictions.")
    parser.add_argument("-q",
                        "--min_qual",
                        required=False,
                        default=None,
                        type=int,
                        help="--min_qual: \t a <int> used as minimum quality to filter modified basecalled bases/nucleotides during genome space aggregation step. [None]")
    parser.add_argument("-c",
                        "--min_mod_count",
                        required=False,
                        default=None,
                        type=int,
                        help="--min_mod_count: \t a <int> used as minimum modification count after genome-space aggregation to maintain a candidate site and compute depth and modification frequency. [None]")
    parser.add_argument("-o",
                        "--output_prefix",
                        required=False,
                        default=None,
                        type=str,
                        help="--output_prefix: \t a <str> indicating where to write the output file (prefix). [None --> it will save using the BAM file-path as prefix]")

    args = parser.parse_args()
    bam_filepath = args.bam_filepath
    per_read_preds = args.per_read_preds
    min_qual = args.min_qual
    min_mod_count = args.min_mod_count
    output_prefix = args.output_prefix
    if not output_prefix:
        output_prefix = bam_filepath


    # print some starting info related to version, used program and to the input arguments
    print(f"[{datetime.now()}] NanoSpeech_basecaller version: {__version__}", flush=True)
    print(f"[{datetime.now()}] aggregate_genome_space.py Input arguments:", flush=True)
    for argument in args.__dict__.keys():
        print(f"\t- {argument} --> {args.__dict__[argument]}", flush=True)

    # launch main function
    df_genome_space = aggregate_genome_space(bam_filepath = bam_filepath,
                                             per_read_preds_filepath=per_read_preds,
                                             min_qual=min_qual,
                                             min_mods=min_mod_count,
                                             save_to_tsv=False)

    print(f"[{datetime.now()}] Aggregation onto genome-space completed...", flush=True)

    # deduce output filepath
    output_filepath = output_prefix + ".genome_space.tsv"

    # ensure that an already existing tsv won't be overwritten
    if os.path.exists(output_filepath):
        print(f"\n[{datetime.now()}] WARNING: The final output file already exists: it will be added a random number within the name to create the new independent tsv file.", flush=True)
        output_filepath = output_prefix + f".{np.random.randint(1000000)}.genome_space.tsv"

    # saving and closing...
    print(f"[{datetime.now()}] Saving the tsv file at: {output_filepath}", flush=True)
    df_genome_space.to_csv(output_filepath, index=False, sep="\t")
    print(f"\n[{datetime.now()}] Computation Finished.", flush=True)