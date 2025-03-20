# import basic modules
import os, sys, pysam, shutil
from glob import iglob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from joblib import load
from model import Transformer, CustomSchedule, VectorizeCharMulti, initialize_model
from misc import generator, convert_MODtoCAN, raw_to_pA, phred_score_to_symbol, create_tf_dataset, create_tf_dataset_basecaller, generate_chunks, generator_consumer
from __init__ import __version__
import argparse
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file
import math
import pandas as pd
import numpy as np
from multiprocessing import Queue, Process, Value
from math import ceil

def producer(fast5_folderpath, q, threads_n, clip_outliers, n_reads_to_process=None, print_read_name=None, fast5list_filepath=None, readslist_filepath=None, chunks_len=None):
    print(f"\n[{datetime.now()}] [Producer Message] Performing basecalling on fast5 files into input folder: {fast5_folderpath}", flush=True)
    print(f"\n[{datetime.now()}] [Producer Message] Outliers clipping limits: {clip_outliers}", flush=True)
    print(f"\n[{datetime.now()}] [Producer Message] Chunks length used by generator: {chunks_len}", flush=True)
    total_fast5_detected = 0
    fast5_processed = 0
    reads_processed_counter = 0
    # if fast5list file not provided scan the input folder/file
    if not fast5list_filepath:
        # if fast5_folderpath is not a file
        if not os.path.isfile(fast5_folderpath):
            # detect total number of fast5 files to be processed
            for fast5_fullpath in iglob(fast5_folderpath + "/**/*fast5", recursive=True):    
                if os.path.isfile(fast5_fullpath):
                    total_fast5_detected += 1
            print(f"\n[{datetime.now()}] [Producer Message] Total fast5 detected into input fast5 main folder: {total_fast5_detected}", flush=True)
            fast5gen = iglob(fast5_folderpath + "/**/*fast5", recursive=True) 
        else:
            # in this case a single fast5 file has been provided as input
            total_fast5_detected += 1
            print(f"\n[{datetime.now()}] [Producer Message] Provided a fast5 file as input. Total fast5 to process: {total_fast5_detected}", flush=True)
            fast5gen = [fast5_folderpath] # as single fast5 to basecall
    else:
        # limit basecalling to a given list of fast5 files
        print(f"\n[{datetime.now()}] [Producer Message] Provided a list of fast5 files. Limiting basecalling to this list: {fast5list_filepath}", flush=True)
        with open(fast5list_filepath, "r") as fast5list_file:
            fast5list = []
            for _l_ in fast5list_file:
                fast5list.append(_l_.rstrip())
        total_fast5_detected = len(fast5list)
        print(f"\n[{datetime.now()}] [Producer Message] Total fast5 to be processed based on provided fast5 list: {total_fast5_detected}", flush=True)
        fast5gen = fast5list

    # if a list of of reads ids has been provided
    if readslist_filepath:
        print(f"\n[{datetime.now()}] A list of reads to focus the basecalling on has been provided at: {readslist_filepath}", flush=True)
        # load reads list
        readslist = []
        with open(readslist_filepath, "r") as readslist_file:
            for _l_ in readslist_file:
                readslist.append(_l_.rstrip())
        n_reads_to_process = len(readslist)
        print(f"\n[{datetime.now()}] A total of {n_reads_to_process} reads have been detected into reads list file. Basecalling will be limited on these ids.", flush=True)

    ### start iteration across fast5s into input fast5_folder ###
    print(f"[{datetime.now()}] [Producer Message] Starting basecalling on reads into fast5 files...", flush=True) 
    with tqdm(total=total_fast5_detected, file=sys.stdout, desc="Fast5s processed:") as pbar:
        for fast5_fullpath in fast5gen:
            if os.path.isfile(fast5_fullpath):
                print(f"\n[{datetime.now()}] [Producer Message] Performing basecalling on fast5 file: {fast5_fullpath}", flush=True)
                fast5_processed += 1
                with get_fast5_file(fast5_fullpath) as f5:
                    for r in f5.get_reads():
                        read_name_id = r.read_id
                        if readslist_filepath:
                            if not read_name_id in readslist:
                                # no basecall for this read id! Go to next one...
                                continue
                        pA_data = raw_to_pA(r)
                        if print_read_name:
                            print(f"\n[{datetime.now()}] [Producer Message] Extracting electric signal for read: {read_name_id} from fast5: {fast5_fullpath}", flush=True)
                        # clip pA_data for outlier with mean value
                        currents_chunk_df = pd.Series(pA_data) # convert to pandas series
                        currents_chunk_df[(currents_chunk_df<clip_outliers[0])|(currents_chunk_df>clip_outliers[1])] = round(currents_chunk_df.mean(), 3) # clip outliers to the average values of the current chunk
                        pA_data = currents_chunk_df.values
                        #print("PRODUCER MESSAGE DEVELOPMENT - pA_data type:", type(pA_data), flush=True)
                        ###!!! add logic for DNA basecalling here ###!!!
                        X = generate_chunks(pA_data[::-1], chunks_len=chunks_len)
                        #print("PRODUCER MESSAGE DEVELOPMENT - X shape:", X.shape, flush=True)
                        q.put([read_name_id, fast5_fullpath, X])
                        reads_processed_counter += 1
                        # block producer if required
                        if n_reads_to_process:
                            if reads_processed_counter / n_reads_to_process % 0.1 == 0:
                                print(f"\n[{datetime.now()}] [Producer Message] Reads processed {reads_processed_counter}/{n_reads_to_process} ({round(100*(reads_processed_counter/n_reads_to_process),2)}).", flush=True)
                            if reads_processed_counter == n_reads_to_process:
                                print(f"\n[{datetime.now()}] [Producer Message] Reached requester amount of reads to be processed. Producer is stopping and will produce end-signals for consumers workers.", flush=True)
                                print(f"\n[{datetime.now()}] [Producer Message] Total number of fast5 files processed: {fast5_processed}", flush=True)
                                print(f"\n[{datetime.now()}] [Producer Message] Total number of reads processed: {reads_processed_counter}", flush=True)
                                break
                if n_reads_to_process:
                    if reads_processed_counter / n_reads_to_process % 0.1 == 0:
                        print(f"\n[{datetime.now()}] [Producer Message] Reads processed {reads_processed_counter}/{n_reads_to_process} ({round(100*(reads_processed_counter/n_reads_to_process),2)}).", flush=True)
                    if reads_processed_counter == n_reads_to_process:
                        print(f"\n[{datetime.now()}] [Producer Message] Reached requester amount of reads to be processed. Producer is stopping and will produce end-signals for consumers workers.", flush=True)
                        print(f"\n[{datetime.now()}] [Producer Message] Total number of fast5 files processed: {fast5_processed}", flush=True)
                        print(f"\n[{datetime.now()}] [Producer Message] Total number of reads processed: {reads_processed_counter}", flush=True)
                        break
                pbar.update(1)
    # append end signal for every consumer
    print(f"\n[{datetime.now()}] [Producer Message] Producing end pills for consumers/workers", flush=True)
    for t in range(threads_n):
        q.put(None)

def consumer_worker(q, id_consumer, model_weigths, out_folderpath, extention, print_gpu_memory=None, print_chunks_idxs=None, mods=None, vocab_string=None, max_len=None, pad_len=971, 
                    num_hid=250,
                    num_head=10,
                    num_feed_forward=450,
                    num_layers_enc=4,
                    num_layers_dec=2):
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    # limit gpu usage per model
    # taken from --> https://stackoverflow.com/questions/55788883/limiting-gpu-memory-usage-by-keras-tf-2019
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                #print("OK! GPU FOUND!")
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.autograph.set_verbosity(1)
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras import layers

    # loading mods, canonical corresponding version and vocabulary
    mods_dict = {}
    for _m_ in mods.split(";"):
        mod_base = _m_.split("->")[0]
        can_base = _m_.split("->")[1]
        mods_dict[mod_base] = can_base
    print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Modification Dictionary: {mods_dict}", flush=True)
    vocab_list = list(vocab_string)
    print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Vocabulary list: {vocab_list}", flush=True)

    # create a VectorizeChar instance
    vectorizer = VectorizeCharMulti(max_len=max_len, vocab=vocab_list)
    # create sofmax node to scale output logits to probability distribution among the possible tokens/symbols
    softmax = tf.keras.layers.Softmax(axis=-1)
    out_filepath_cons = os.path.join(out_folderpath, f"tmp_cons_{id_consumer}.{extention}")
    print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Output temporary file: {out_filepath_cons}", flush=True)
    # inizialize the model for the current consumer worker
    print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Initializing NanoSpeech transformer model...", flush=True)
    model = initialize_model(id_consumer, model_weigths, num_classes=len(vocab_list), target_maxlen=max_len, pad_len=pad_len,
                             num_hid=num_hid,
                             num_head=num_head,
                             num_feed_forward=num_feed_forward,
                             num_layers_enc=num_layers_enc,
                             num_layers_dec=num_layers_dec)
    #print(model.summary())
    output = open(out_filepath_cons, "w")
    print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Starting processing fast5 files...", flush=True)
    while True:
        # fetch from queue populated by producer as a list: [read_name_id, fast5_fullpath, pA_data]
        # create ds from inverted pA signal (since we are working on direct-RNA seq data)
        prod_out = q.get()
        if prod_out != None:
            read_name_id = prod_out[0]
            fast5_fullpath = prod_out[1]
            X = prod_out[2]
            #print(f"\n##################################\n[{datetime.now()}] [Consumer {id_consumer} Message]", read_name_id, X.shape, flush=True) ##########!!!!!!! DEVELOPMENT
            #print(f"[{datetime.now()}] [Consumer {id_consumer} Message]", X, flush=True) ##########!!!!!!! DEVELOPMENT
            #ds = create_tf_dataset_basecaller(X, bs=512)
            ds = tf.data.Dataset.from_generator(generator_consumer, args=[X, pad_len], output_types=(tf.float32), output_shapes = ((pad_len,126)), )
            ds = ds.batch(6)
            #print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message]\t", read_name_id, ds) ##########!!!!!!!
            if print_gpu_memory:
                print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] GPU memory in use: {round(tf.config.experimental.get_memory_info('GPU:0')['current']/1024/1024/1024, 4)} GB", file=sys.stderr, flush=True)
            # make predictions for current read using pA signals
            PREDS=[]
            ENDS_OF_CHUNKS = []
            if extention == "fastq":
                PROBS = []
            target_end_token_idx = 2
            idx_to_char = vectorizer.get_vocabulary()
            c=0
            last_end_of_chunk = -1 # a counter to take trace of the last end of the chunk (0-based)
            for i in ds:

                if extention == "fasta":
                    ### FASTA CODE...
                    # generate prediction and convert to text version
                    try:
                        pred = model.generate(i, 1)
                    except:
                        print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Error! Problem during inference on batch of read: {read_name_id} from fast5: {fast5_fullpath}. SKIPPING BATCH...", flush=True, file=sys.stderr)
                        continue
                    
                    for p in pred:
                        c+=1
                        prediction = ""
                        for idx in p:
                            prediction += idx_to_char[idx]
                            if idx == target_end_token_idx:
                                break
                        # assess if start and ends with start-stop symbols
                        if prediction.startswith("<") and prediction.endswith(">"):
                            # asses if predicted bases are inside a list of allowed nucleotides
                            if set( [i in vectorizer.vocab[3:] for i in set(prediction[1:-1])] ) == {True}:
                                PREDS.append(prediction[1:-1])
                                last_end_of_chunk += len(prediction[1:-1])
                                ENDS_OF_CHUNKS.append(last_end_of_chunk)
                            else:
                                #print(f"[{datetime.now()}] [Consumer {id_consumer} Message] Error (UNPREDICTED_BASEs_BETWEEN_START_AND_STOP_TOKENS) for chunk n° {c} on read: {read_name_id} fast5: {fast5_fullpath}.", file=sys.stderr, flush=True)
                                ENDS_OF_CHUNKS.append(-2)
                        else:
                            #print(f"[{datetime.now()}] [Consumer {id_consumer} Message] Error (NO_START_OR_STOP) for chunk n° {c} on read: {read_name_id} fast5: {fast5_fullpath}.", file=sys.stderr, flush=True)
                            ENDS_OF_CHUNKS.append(-1)

                elif extention == "fastq":
                    ### FASTQ CODE...
                    # generate prediction and convert to text version
                    try:
                        pred, prob = model.generate(i, 1, return_proba=True)
                    except Exception as e:
                        print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Error! Problem during inference on batch of read: {read_name_id} from fast5: {fast5_fullpath}. SKIPPING BATCH...", flush=True, file=sys.stderr)
                        print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Exception --> {e}", flush=True,file=sys.stderr)
                        continue
                    for p,p_ in zip(pred, prob): # p -> prediction, p_ -> probabilities
                        #print("p:", p, flush=True) #### !!!! DEVELOPMENT
                        c+=1
                        prediction = ""
                        probabilities_err = []
                        phred_scores = ""
                        for i__, idx in enumerate(p):
                            prediction += idx_to_char[idx]
                            #if idx == 8 or idx == 7:
                            #    print("############### ", idx, idx_to_char[idx], "on read:", read_name_id, flush=True)
                            # extract probability of error (as 1 - basecalled base proba scaled with softmax func.)
                            # to note: here it will start from second entry intro p because proba doesn't have start symbol
                            if i__ > 0:
                                prob_err = 1-softmax(p_[i__-1])[idx] # 1 - probability of called base/symbol/token
                                phred_score = int(-10 * math.log10( float(prob_err) ))
                                probabilities_err.append(round(phred_score))
                                phred_scores += phred_score_to_symbol(phred_score)
                            if idx == target_end_token_idx:
                                break
                        # asses if start and ends with start-stop symbols
                        if prediction.startswith("<") and prediction.endswith(">"):
                            # asses if predicted bases are inside a list of allowed nucleotides
                            if set( [i in vectorizer.vocab[3:] for i in set(prediction[1:-1])] ) == {True}:
                                PREDS.append(prediction[1:-1])
                                PROBS.append(phred_scores[:-1]) # lack of ">" start symbol
                                last_end_of_chunk += len(prediction[1:-1])
                                ENDS_OF_CHUNKS.append(last_end_of_chunk)
                            else:
                                #print(f"[{datetime.now()}] [Consumer {id_consumer} Message] Error (UNPREDICTED_BASEs_BETWEEN_START_AND_STOP_TOKENS) for chunk n° {c} on read: {read_name_id} fast5: {fast5_fullpath}.", file=sys.stderr, flush=True)
                                ENDS_OF_CHUNKS.append(-2)
                        else:
                            #print(f"[{datetime.now()}] [Consumer {id_consumer} Message] Error (NO_START_OR_STOP) for chunk n° {c} on read: {read_name_id} fast5: {fast5_fullpath}.", file=sys.stderr, flush=True)
                            ENDS_OF_CHUNKS.append(-1)
                            
            # join to create the final merged output read (in fasta or fastq format as request with the output filename extention)
            # assess if NO_START_OR_STOP
            pred_seq = "".join(PREDS).upper()
            if len(pred_seq) > 0: # no empty basecalled reads
                if extention == "fastq":
                    phred_scores_seq = "".join(PROBS)
                # convert modified nucleotide to canonical version into fasta/fastq sequence and retrieve predicted mods indexes
                pred_seq_conv, pred_seq_mods_idxs = convert_MODtoCAN(pred_seq, mods_dict=mods_dict)
                if extention == "fasta":
                    # generate the 2 fasta rows and write these into the output file
                    fastx_read = f">{read_name_id}"
                    for _key_ in mods_dict.keys():
                        if len(pred_seq_mods_idxs[_key_]) > 0:
                            fastx_read += f" {_key_}_{mods_dict[_key_]}={','.join([str(_m_idx) for _m_idx in pred_seq_mods_idxs[_key_]])}"
                    fastx_read+="\n"
                    fastx_read+=pred_seq_conv
                    fastx_read+="\n"
                    output.write(fastx_read)
                    output.flush()
                elif extention == "fastq":
                    # generate the 4 fastq rows and write these into the output file
                    fastx_read = f"@{read_name_id}"
                    for _key_ in mods_dict.keys():
                        if len(pred_seq_mods_idxs[_key_]) > 0:
                            fastx_read += f" {_key_}_{mods_dict[_key_]}={','.join([str(_m_idx) for _m_idx in pred_seq_mods_idxs[_key_]])}"
                    fastx_read+="\n"
                    fastx_read+=pred_seq_conv
                    fastx_read+="\n"
                    output.write(fastx_read)
                    if print_chunks_idxs:
                        output.write(f"+{','.join([str(__i__) for __i__ in ENDS_OF_CHUNKS])}\n")
                    else:
                        output.write(f"+\n")
                    output.write(f"{phred_scores_seq}\n")
                    output.flush()
            else:
                print(f"[{datetime.now()}] [Consumer {id_consumer} Message] Error (empty basecalled read) on read: {read_name_id} from fast5: {fast5_fullpath}", file=sys.stderr, flush=True)
        else:
            # Stopping the loop of the consumer if found a end-signal (None) in the Queue.
            print(f"\n[{datetime.now()}] [Consumer {id_consumer} Message] Found end of Queue.", flush=True)
            output.close()
            break


def basecaller(fast5_folderpath, out_filepath, model_weigths, clip_outliers = None, print_gpu_memory=False, 
               print_read_name = False, n_reads_to_process = None, n_models = 1, fast5list_filepath = None, 
               readslist_filepath = None, chunks_len = None, print_chunks_idxs = None, max_len = None):
    print(f"[{datetime.now()}] [Main Process Message] NanoSpeech modified basecaller (spectrogram from fast5)", flush=True)
    # detect extention of output file and if it has the right type
    extention = os.path.splitext(out_filepath)[1][1:].lower()
    out_folderpath = os.path.splitext(out_filepath)[0]
    if not extention in ["fasta", "fastq"]:
        sys.exit(f"[{datetime.now()}] [Main Process Message] Ouput file extention not allowed ({extention}). It should be either fasta or fastq. Exiting...")
    print(f"[{datetime.now()}] [Main Process Message] Extention detected: <.{extention}>", flush=True)
    print(f"[{datetime.now()}] [Main Process Message] Generating output folder where basecalled reads will be saved.", flush=True)
    if os.path.exists(out_folderpath):
        shutil.rmtree(out_folderpath)
    os.mkdir(out_folderpath)
    
    # loading configuration file for requested model
    configurations_fp = model_weigths+".cfg"
    print(f"[{datetime.now()}] [Main Process Message] Loading configuration file for selected model at: {configurations_fp}", flush=True)
    configurations = pd.read_csv(configurations_fp, index_col=0, header=None).iloc[:,0]
    # print(configurations)
    if not chunks_len:
        chunks_len = int(configurations["chunks_len"])
        print(f"[{datetime.now()}] [Main Process Message] chunks_len: {chunks_len}", flush=True)
    if not clip_outliers:
        clip_outliers = [int(i) for i in configurations["clip_outliers"].split("-")]
        print(f"[{datetime.now()}] [Main Process Message] clip_outliers: {clip_outliers}", flush=True)
    if not max_len:
        max_len = int(configurations["max_len"])
        print(f"[{datetime.now()}] [Main Process Message] max_len for output vectors: {max_len}", flush=True)
    mods = configurations["mods"]
    print(f"[{datetime.now()}] [Main Process Message] Modifications (mod_nucletide->canonical_nucleotide): {mods}", flush=True)
    vocab_string = configurations["vocab"]
    print(f"[{datetime.now()}] [Main Process Message] Vocabulary string: {vocab_string}", flush=True)
    pad_len = int(configurations["pad_len"])
    print(f"[{datetime.now()}] [Main Process Message] pad_len for input tensors: {pad_len}", flush=True)
    
    num_hid = int(configurations["num_hid"])
    num_head = int(configurations["num_head"])
    num_feed_forward = int(configurations["num_feed_forward"])
    num_layers_enc = int(configurations["num_layers_enc"])
    num_layers_dec = int(configurations["num_layers_dec"])


    start_global = datetime.now()
    q = Queue(maxsize=n_models*20)
    # create consumers processes
    consumers = []
    for t in range(n_models):
        # q, id_consumer, model_weigths, out_folderpath, extention, print_gpu_memory=None, print_chunks_idxs=None
        consumers.append( Process(target=consumer_worker, args=(q, t+1, model_weigths, out_folderpath, 
                                                                extention, print_gpu_memory, print_chunks_idxs, mods, vocab_string, max_len, pad_len,
                                                                num_hid,
                                                                num_head,
                                                                num_feed_forward,
                                                                num_layers_enc,
                                                                num_layers_dec))
                                                                )
    print(f"[{datetime.now()}] [Main Process Message] Generating requested consumers. N° of Consumers: {len(consumers)}", flush=True)
    # start consumers
    for c in consumers:
        c.start()
    # create a producer process
    # fast5_folderpath, q, threads_n, clip_outliers, n_reads_to_process=None, print_read_name=None, fast5list_filepath=None, readslist_filepath=None, chunks_len=2800
    p = Process(target=producer, args=(fast5_folderpath, q, n_models, clip_outliers, n_reads_to_process, 
                                       print_read_name, fast5list_filepath, readslist_filepath, chunks_len))
    p.start()
    
    # join consumers
    for c in consumers:
        c.join()
    # join producer
    p.join()

    # merge fastq files of consumers, and delete temporary files
    print(f"[{datetime.now()}] [Main Process Message] Merging fasta/q files generated by consumers at path: {out_folderpath}", flush=True)
    os.system(f"cat {os.path.join(out_folderpath, 'tmp_cons_')}* > {out_filepath}")
    os.system(f"rm -r {out_folderpath}")
    
    stop_global = datetime.now()
    print(f"[{datetime.now()}] [Main Process Message] Computation finished. Global Elapsed time: {stop_global - start_global}", flush=True)
    print(f"[{datetime.now()}] [Main Process Message] EXITING...Queue final size is:", q.qsize(), flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"NanoSpeech basecaller_multi_mod.py v. {__version__}")
    parser.add_argument("-d",
                        "--fast5_folderpath",
                        required=True,
                        type=str,
                        help="--fast5_folderpath: \t a <str> with the fullpath for the input fast5 folderpath.")
    parser.add_argument("-o",
                        "--out_filepath",
                        required=True,
                        type=str,
                        help="--out_filepath: \t a <str> with the fullpath for the output fasta/fastq file generated during the basecalling.")
    parser.add_argument("-m",
                        "--model_weigths",
                        required=True,
                        type=str,
                        help="--model_weigths: \t a <str> with the fullpaht for the h5 file containing the weights to inizialize pretrained transformer model.")
    parser.add_argument("-t",
                        "--threads_n",
                        required=False,
                        default=1,
                        type=int,
                        help="--threads_n: \t a <int> indicating the number of basecaller workers.")
    parser.add_argument("-c",
                        "--clip_outliers",
                        required=False,
                        default=None,
                        type=str,
                        help="--clip_outliers: \t a <str> indicating the min-max currents value to be clipped with mean. [None]")
    parser.add_argument("-u",
                        "--print_gpu_memory",
                        required=False,
                        default=False,
                        type=str,
                        help="--print_gpu_memory: \t <str> Set to True to print gpu usage for every worker starting a new read (experimental). [False]")
    parser.add_argument("-r",
                        "--print_read_name",
                        required=False,
                        default=False,
                        type=str,
                        help="--print_read_name: \t <bool> Set to True to let producer to print reads names added to queue. [False]")
    parser.add_argument("-nr",
                        "--n_reads_to_process",
                        required=False,
                        default=None,
                        help="--n_reads_to_process: <int> Numer of reads to limit basecalling. [None]")
    parser.add_argument("-fl",
                        "--fast5list_filepath",
                        required=False,
                        default=None,
                        help="--fast5list_filepath: \n <str> Fullpath for a file with list of paths to fast5 files to limit basecalling on. [None]")
    parser.add_argument("-rl",
                        "--readslist_filepath",
                        required=False,
                        default=None,
                        help="--readslist_filepath: \n <str> Fullpath for a file with list of reads ids to limit basecalling on. [None]")
    parser.add_argument("-cl",
                        "--chunks_len",
                        required=False,
                        default=None,
                        type=int,
                        help="--chunks_len: \n <int> Chunks lenght the generator will be output from raw signals. [None]")
    parser.add_argument("-idxs",
                        "--print_chunks_idxs",
                        required=False,
                        default=None,
                        type=str,
                        help="--print_chunks_idxs: \n <bool> Set to True to print 0-based index for the end of chunks in the + line in fastq output [None]")

    args = parser.parse_args()
    fast5_folderpath = args.fast5_folderpath
    out_filepath = args.out_filepath
    model_weigths = args.model_weigths
    n_models = args.threads_n
    clip_outliers = args.clip_outliers
    if clip_outliers:
        clip_outliers = [int(i) for i in clip_outliers.split("-")]
    print_gpu_memory = args.print_gpu_memory
    if type(print_gpu_memory) == str:
        if print_gpu_memory == "True":
            print_gpu_memory = True
        elif print_gpu_memory == "False":
            print_gpu_memory = False
        else:
            print(f"ERROR!! print_gpu_memory has to be either True or False")
    print_read_name = args.print_read_name
    if type(print_read_name) == str:
        if print_read_name == "True":
            print_read_name = True
        elif print_read_name == "False":
            print_read_name = False
        else:
            print(f"ERROR!! print_read_name has to be either True or False")
    n_reads_to_process = args.n_reads_to_process
    if n_reads_to_process:
        n_reads_to_process = int(n_reads_to_process)
    fast5list_filepath = args.fast5list_filepath
    if type(fast5list_filepath) == str:
        if fast5list_filepath == "True":
            fast5list_filepath = True
        elif fast5list_filepath == "False":
            fast5list_filepath = False
    readslist_filepath = args.readslist_filepath
    if type(readslist_filepath) == str:
        if readslist_filepath == "True":
            readslist_filepath = True
        elif readslist_filepath == "False":
            readslist_filepath = False
    chunks_len = args.chunks_len
    print_chunks_idxs = args.print_chunks_idxs
    if type(print_chunks_idxs) == str:
        if print_chunks_idxs == "True":
            print_chunks_idxs = True
        elif print_chunks_idxs == "False":
            print_chunks_idxs = False

    # print some starting info related to version, used program and to the input arguments
    print(f"[{datetime.now()}] NanoSpeech_basecaller version: {__version__}", flush=True)
    print(f"[{datetime.now()}] NanoSpeech.py Input arguments:", flush=True)
    for argument in args.__dict__.keys():
        print(f"\t- {argument} --> {args.__dict__[argument]}", flush=True)

    # launch main function
    basecaller(fast5_folderpath = fast5_folderpath, 
               out_filepath = out_filepath, 
               model_weigths = model_weigths, 
               clip_outliers = clip_outliers, 
               print_gpu_memory = print_gpu_memory, 
               print_read_name = print_read_name, 
               n_reads_to_process = n_reads_to_process, 
               n_models = n_models, 
               fast5list_filepath = fast5list_filepath, 
               readslist_filepath = readslist_filepath,
               chunks_len = chunks_len,
               print_chunks_idxs=print_chunks_idxs)