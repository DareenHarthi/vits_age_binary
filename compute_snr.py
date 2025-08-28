import os
import glob
import subprocess
import tempfile
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import logging
import librosa  # For audio loading
from pathlib import Path
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the meta parameters
DATA_PATH = "/ocean/projects/cis220031p/alharthi/data/commonvoice/en/"
NUM_PROC = 100
OUTPUT_FILE = "snr_results.txt"

def wada_snr(wav):
    """
    Direct blind estimation of the SNR of a speech signal.
    
    Paper on WADA SNR:
    http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    
    This function was adapted from this matlab code:
    https://labrosa.ee.columbia.edu/projects/snreval/#9
    """
    # init
    eps = 1e-10
    
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = np.arange(-20, 101)
    g_vals = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 
                       0.40999006, 0.41027138, 0.41052627, 0.41101024, 0.41143264, 
                       0.41231718, 0.41337272, 0.41526426, 0.4178192 , 0.42077252, 
                       0.42452799, 0.42918886, 0.43510373, 0.44234195, 0.45161485, 
                       0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52353709, 
                       0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 
                       0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 
                       0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418, 
                       0.95655449, 0.9835349 , 1.01047155, 1.0362095 , 1.06136425, 
                       1.08579312, 1.1094819 , 1.13277995, 1.15472826, 1.17627308, 
                       1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 
                       1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935, 
                       1.3605727 , 1.37345513, 1.38577122, 1.39733504, 1.40856397, 
                       1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 
                       1.46669568, 1.47486938, 1.48269965, 1.49034339, 1.49748214, 
                       1.50435106, 1.51076426, 1.51698915, 1.5229097 , 1.528578 , 
                       1.53389835, 1.5391211 , 1.5439065 , 1.54858517, 1.55310776, 
                       1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767, 
                       1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 
                       1.59162477, 1.5941969 , 1.59693155, 1.599446 , 1.60185011, 
                       1.60408668, 1.60627134, 1.60826199, 1.61004547, 1.61192472, 
                       1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 
                       1.62135878, 1.62268119, 1.62390423, 1.62513143, 1.62632463, 
                       1.6274027 , 1.62842767, 1.62945532, 1.6303307 , 1.63128026, 
                       1.63204102])
    
    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    if len(wav) == 0:
        return np.nan
        
    wav = wav / (abs(wav).max() + eps)  # Add eps to avoid division by zero
    abs_wav = abs(wav)
    abs_wav[abs_wav < eps] = eps
    
    # calculate statistics
    # E[|z|]
    v1 = max(eps, abs_wav.mean())
    # E[log|z|]
    v2 = np.log(abs_wav).mean()
    # log(E[|z|]) - E[log(|z|)]
    v3 = np.log(v1) - v2
    
    # table interpolation
    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    
    # handle edge cases or interpolate
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
    else:
        wav_snr = db_vals[wav_snr_idx] + \
                  (v3-g_vals[wav_snr_idx]) / (g_vals[wav_snr_idx+1] - 
                   g_vals[wav_snr_idx]) * (db_vals[wav_snr_idx+1] - db_vals[wav_snr_idx])
    
    # Calculate SNR
    dEng = sum(wav**2)
    if dEng == 0:
        return np.nan
        
    dFactor = 10**(wav_snr / 10)
    dNoiseEng = dEng / (1 + dFactor)  # Noise energy
    dSigEng = dEng * dFactor / (1 + dFactor)  # Signal energy
    
    if dNoiseEng <= 0:
        return np.nan
        
    snr = 10 * np.log10(dSigEng / dNoiseEng)
    return snr

def compute_file_snr(file_path):
    """Load audio file and compute SNR using Python WADA implementation."""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return np.nan, file_path
        
        # Load audio file using librosa (handles many formats)
        try:
            wav, sr = librosa.load(file_path, sr=16000, mono=True)
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            return np.nan, file_path
        
        if len(wav) == 0:
            logger.warning(f"Empty audio file: {file_path}")
            return np.nan, file_path
        
        # Compute SNR
        snr_value = wada_snr(wav)
        
        if np.isnan(snr_value):
            logger.warning(f"NaN SNR computed for {file_path}")
        
        return snr_value, file_path
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return np.nan, file_path

def save_results_to_file(file_snrs, output_file):
    """Save results to text file in format: file_path|snr_value"""
    valid_results = []
    nan_results = []
    
    with open(output_file, 'w') as f:
        for snr_value, file_path in file_snrs:
            if np.isnan(snr_value):
                nan_results.append(file_path)
                f.write(f"{file_path}|NaN\n")
            else:
                valid_results.append((file_path, snr_value))
                f.write(f"{file_path}|{snr_value:.6f}\n")
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Valid SNR values: {len(valid_results)}")
    logger.info(f"NaN values: {len(nan_results)}")
    
    return valid_results, nan_results

def main():
    """Main function to process all wav files and compute SNR values."""
    
    # Get all wav files
    wav_files = Path(f"{DATA_PATH}/**/*.wav") #list(glob.glob(f"{DATA_PATH}/**/*.wav")) 
    
    logger.info(f"Found {len(wav_files)} wav files to process")
    
    if len(wav_files) == 0:
        logger.error(f"No wav files found in {DATA_PATH}")
        return
    
    # Process files
    logger.info("Starting SNR computation using Python WADA implementation...")
    
    if NUM_PROC == 1:
        # Single-threaded processing
        file_snrs = []
        for wav_file in tqdm(wav_files, desc="Processing files"):
            result = compute_file_snr(wav_file)
            file_snrs.append(result)
    else:
        # Multi-threaded processing
        with Pool(NUM_PROC) as pool:
            file_snrs = list(tqdm(
                pool.imap(compute_file_snr, wav_files), 
                total=len(wav_files),
                desc="Processing files"
            ))
    
    # Save results
    valid_results, nan_results = save_results_to_file(file_snrs, OUTPUT_FILE)
    
    # Print summary
    logger.info("="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total files processed: {len(wav_files)}")
    logger.info(f"Valid SNR values: {len(valid_results)}")
    logger.info(f"Failed/NaN values: {len(nan_results)}")
    logger.info(f"Success rate: {len(valid_results)/len(wav_files)*100:.1f}%")
    
    if valid_results:
        snr_values = [snr for _, snr in valid_results]
        logger.info(f"SNR statistics:")
        logger.info(f"  Mean: {np.mean(snr_values):.2f} dB")
        logger.info(f"  Std:  {np.std(snr_values):.2f} dB")
        logger.info(f"  Min:  {np.min(snr_values):.2f} dB")
        logger.info(f"  Max:  {np.max(snr_values):.2f} dB")
    
    if nan_results:
        logger.info("\nFiles with NaN SNR values:")
        for file_path in nan_results[:10]:  # Show first 10
            logger.info(f"  {file_path}")
        if len(nan_results) > 10:
            logger.info(f"  ... and {len(nan_results) - 10} more")
        
        # Save failed files list
        failed_file = "failed_snr_files.txt"
        with open(failed_file, 'w') as f:
            for file_path in nan_results:
                f.write(f"{file_path}\n")
        logger.info(f"Failed files list saved to {failed_file}")

if __name__ == "__main__":
    main()