import sys

def filter_fastq_with_chunks_joins_indexes(fastq_with_chunks_idxs_fp):
    out_filt_fastq_fp = fastq_with_chunks_idxs_fp+".idxsfilt.fastq"
    print("Trying to filter chunks join points indexes from '+' lines...", flush=True)
    print("Input FASTQ:", fastq_with_chunks_idxs_fp, flush=True)
    print("Output Filtered FASTQ (no chunks idxs):", out_filt_fastq_fp, flush=True)
    with open(out_filt_fastq_fp, "w") as out:
        with open(fastq_with_chunks_idxs_fp, "r") as f:
            plus_line = 2
            for c,l in enumerate(f):
                if c != plus_line:
                    out.write(l)
                elif c == plus_line:
                    out.write("+\n")
                    plus_line += 4

fastq_with_chunks_idxs_fp = sys.argv[1]

filter_fastq_with_chunks_joins_indexes(fastq_with_chunks_idxs_fp)