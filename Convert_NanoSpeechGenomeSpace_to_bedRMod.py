import os, sys
from datetime import datetime
from math import ceil
import pandas as pd

fileformat=sys.argv[1]
organism=sys.argv[2]
modification_type=sys.argv[3]
assembly=sys.argv[4]
annotation_source=sys.argv[5]
annotation_version=sys.argv[6]
sequencing_platform=sys.argv[7]
basecalling=sys.argv[8]
bioinformatics_workflow=sys.argv[9]
experiment=sys.argv[10]
external_source=sys.argv[11]
mod_symbols_color_table = sys.argv[12]
genome_space_tsv=sys.argv[13]
output=sys.argv[14]

print(f"################################################################################################", flush=True)
print(f"################################################################################################", flush=True)
print(f"################################################################################################", flush=True)
print(f"[{datetime.now()}] Conversion of NanoSpeech Genome-Space tsv output file to bedRMod format.", flush=True)
print(f"[{datetime.now()}] Input Modification symbols and color table: {mod_symbols_color_table}", flush=True)
print(f"[{datetime.now()}] Input Genome-Space TSV: {genome_space_tsv}", flush=True)
print(f"[{datetime.now()}] Output bed2Mod: {output}", flush=True)
print("", flush=True)

# load symbols table
mod_symbols_color_table_df = pd.read_csv(mod_symbols_color_table)
#print(mod_symbols_color_table_df, flush=True)

modification_names=[] # deduce modification_names string for bedRMod header
for m in mod_symbols_color_table_df[~mod_symbols_color_table_df["MODOMICS"].isna()].itertuples():
    modification_names.append(f"{m._4}:{m.MODOMICS}:{m.CanonicalBase}")
modification_names = ",".join(modification_names)

# create header
header_string = \
f'''#fileformat={fileformat}
#organism={organism}
#modification_type={modification_type}
#modification_names={modification_names}
#assembly={assembly}
#annotation_source={annotation_source}
#annotation_version={annotation_version}
#sequencing_platform={sequencing_platform}
#basecalling={basecalling}
#bioinformatics_workflow={bioinformatics_workflow}
#experiment={experiment}
#external_source={external_source}
#chrom\tchromStart\tchromEnd\tname\tscore\tstrand\tthickStart\tthickEnd\titemRgb\tcoverage\tfrequency
'''

print(f"[{datetime.now()}] Printing Header String", flush=True)
print(header_string, flush=True)

print(f"[{datetime.now()}] Starting conversion...", flush=True)
with open(genome_space_tsv,'r') as gs:
    with open(output,'w') as o:
        o.write(header_string)
        for _,l in enumerate(gs):
            if not l.startswith('region'):
                line = l.strip().split('\t')
                #print("#")
                #print(header, flush=True)
                #print(line, flush=True)
                mod_code_q = mod_symbols_color_table_df[mod_symbols_color_table_df["ModificationCharNanoSpeech"] == line[3] ]["Dorado_code (pseudo ChEBI)"] # to change to MODOMICS
                if mod_code_q.shape[0] == 1:
                    mod_code = str(mod_code_q.values[0])
                itemRgb = "255,0,0" # SINGLE COLOR
                freq = str(round(100*float(line[-1]),2)) # freq as percentage
                line_out = "\t".join([line[0], str(int(float(line[1]))), str(int(float(line[1]))+1),
                                      mod_code, str(ceil(float(line[6]))), line[2],
                                      str(int(float(line[1]))), str(int(float(line[1]))+1),
                                      itemRgb, line[-2], freq])+"\n"
                #print(line_out, flush=True)
                o.write(line_out)
            else:
                line = l.strip().split('\t')
                header=line

print(f"\n[{datetime.now()}] Conversion finished. EXITING.", flush=True)