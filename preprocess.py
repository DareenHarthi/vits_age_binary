# import argparse
# import text
# from utils import load_filepaths_and_text

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--out_extension", default="cleaned")
#   parser.add_argument("--text_index", default=5, type=int)
#   parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
#   parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

#   args = parser.parse_args()
    

#   for filelist in args.filelists:
#     print("START:", filelist)
#     filepaths_and_text = load_filepaths_and_text(filelist)
#     for i in range(len(filepaths_and_text)):
#       original_text = filepaths_and_text[i][args.text_index]
#       cleaned_text = text._clean_text(original_text, args.text_cleaners)
#       filepaths_and_text[i][args.text_index] = cleaned_text

#     new_filelist = filelist + "." + args.out_extension
#     with open(new_filelist, "w", encoding="utf-8") as f:
#       f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

# import argparse
# import text
# from utils import load_filepaths_and_text
# from multiprocessing import Pool
# from tqdm import tqdm


# def clean_single_text(args):
#     """Clean text for a single entry"""
#     entry, text_index, text_cleaners = args
#     original_text = entry[text_index]
#     cleaned_text = text._clean_text(original_text, text_cleaners)
#     entry[text_index] = cleaned_text
#     return entry


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--out_extension", default="cleaned")
#     parser.add_argument("--text_index", default=5, type=int)
#     parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
#     parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

#     args = parser.parse_args()

#     for filelist in args.filelists:
#         print("START:", filelist)
#         filepaths_and_text = load_filepaths_and_text(filelist)
        
#         # Prepare arguments for multiprocessing
#         mp_args = [(entry, args.text_index, args.text_cleaners) for entry in filepaths_and_text]
        
#         # Use multiprocessing instead of for loop
#         with Pool() as pool:
#             filepaths_and_text = list(tqdm(
#                 pool.imap(clean_single_text, mp_args), 
#                 total=len(mp_args),
#                 desc=f"Processing {filelist}"
#             ))

#         new_filelist = filelist + "." + args.out_extension
#         with open(new_filelist, "w", encoding="utf-8") as f:
#             f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

import argparse
import text
from utils import load_filepaths_and_text
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os


def clean_text_entry(args_tuple):
    """
    Clean a single text entry.
    
    Args:
        args_tuple: (filepath_and_text, text_index, text_cleaners)
    
    Returns:
        Modified filepath_and_text entry
    """
    filepath_and_text, text_index, text_cleaners = args_tuple
    
    # Make a copy to avoid modifying the original
    result = filepath_and_text.copy()
    
    # Clean the text at the specified index
    original_text = result[text_index]
    cleaned_text = text._clean_text(original_text, text_cleaners)
    result[text_index] = cleaned_text
    
    return result


def process_filelist(filelist, text_index, text_cleaners, out_extension, num_processes=None):
    """
    Process a single filelist with multiprocessing and progress bar.
    
    Args:
        filelist: Path to the filelist file
        text_index: Index of text column in the filelist
        text_cleaners: List of text cleaners to apply
        out_extension: Output file extension
        num_processes: Number of processes to use (None for auto-detect)
    """
    print(f"START: {filelist}")
    
    # Load data
    filepaths_and_text = load_filepaths_and_text(filelist)
    
    if not filepaths_and_text:
        print(f"Warning: No data found in {filelist}")
        return
    
    # Prepare arguments for multiprocessing
    args_list = [
        (entry, text_index, text_cleaners) 
        for entry in filepaths_and_text
    ]
    
    # Use multiprocessing with progress bar
    if num_processes is None:
        num_processes = min(cpu_count(), len(filepaths_and_text))
    
    print(f"Processing {len(filepaths_and_text)} entries with {num_processes} processes...")
    
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        cleaned_data = list(tqdm(
            pool.imap(clean_text_entry, args_list),
            total=len(args_list),
            desc=f"Cleaning {os.path.basename(filelist)}",
            unit="entries"
        ))
    
    # Write cleaned data to new file
    new_filelist = filelist + "." + out_extension
    
    with open(new_filelist, "w", encoding="utf-8") as f:
        f.writelines(["|".join(x) + "\n" for x in cleaned_data])
    
    print(f"COMPLETED: {new_filelist}")


def process_filelist_chunked(filelist, text_index, text_cleaners, out_extension, num_processes=None, chunk_size=1000):
    """
    Process a single filelist with chunked multiprocessing for very large files.
    
    Args:
        filelist: Path to the filelist file
        text_index: Index of text column in the filelist
        text_cleaners: List of text cleaners to apply
        out_extension: Output file extension
        num_processes: Number of processes to use (None for auto-detect)
        chunk_size: Size of chunks for processing
    """
    print(f"START (chunked): {filelist}")
    
    # Load data
    filepaths_and_text = load_filepaths_and_text(filelist)
    
    if not filepaths_and_text:
        print(f"Warning: No data found in {filelist}")
        return
    
    if num_processes is None:
        num_processes = min(cpu_count(), len(filepaths_and_text))
    
    print(f"Processing {len(filepaths_and_text)} entries with {num_processes} processes (chunk size: {chunk_size})...")
    
    cleaned_data = []
    
    # Process in chunks
    for i in tqdm(range(0, len(filepaths_and_text), chunk_size), desc="Processing chunks"):
        chunk = filepaths_and_text[i:i + chunk_size]
        
        # Prepare arguments for this chunk
        args_list = [
            (entry, text_index, text_cleaners) 
            for entry in chunk
        ]
        
        # Process chunk with multiprocessing
        with Pool(processes=num_processes) as pool:
            chunk_results = pool.map(clean_text_entry, args_list)
        
        cleaned_data.extend(chunk_results)
    
    # Write cleaned data to new file
    new_filelist = filelist + "." + out_extension
    
    with open(new_filelist, "w", encoding="utf-8") as f:
        f.writelines(["|".join(x) + "\n" for x in cleaned_data])
    
    print(f"COMPLETED: {new_filelist}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean text in filelists with multiprocessing")
    parser.add_argument("--out_extension", default="cleaned", help="Output file extension")
    parser.add_argument("--text_index", default=5, type=int, help="Index of text column in filelist")
    parser.add_argument("--filelists", nargs="+", 
                       default=["filelists/ljs_audio_text_val_filelist.txt", 
                               "filelists/ljs_audio_text_test_filelist.txt"],
                       help="List of filelist files to process")
    parser.add_argument("--text_cleaners", nargs="+", 
                       default=["english_cleaners2"],
                       help="List of text cleaners to apply")
    parser.add_argument("--num_processes", type=int, default=None,
                       help="Number of processes to use (default: auto-detect)")
    parser.add_argument("--chunk_size", type=int, default=1000,
                       help="Chunk size for processing large files")
    parser.add_argument("--use_chunked", action="store_true",
                       help="Use chunked processing for very large files")
    
    args = parser.parse_args()
    
    # Process each filelist
    for filelist in args.filelists:
        try:
            if args.use_chunked:
                process_filelist_chunked(
                    filelist, 
                    args.text_index, 
                    args.text_cleaners, 
                    args.out_extension,
                    args.num_processes,
                    args.chunk_size
                )
            else:
                process_filelist(
                    filelist, 
                    args.text_index, 
                    args.text_cleaners, 
                    args.out_extension,
                    args.num_processes
                )
        except Exception as e:
            print(f"Error processing {filelist}: {e}")
            continue
    
    print("All filelists processed!")