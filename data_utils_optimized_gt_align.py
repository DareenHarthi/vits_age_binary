import time
import os
import random
import numpy as np
import torch
import torch.utils.data
import pickle
import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence

import random
import numpy as np
from collections import defaultdict, Counter

def create_balanced_dataset_by_duplication(audiopaths_data, lengths_data, 
                                         balance_by=['age', 'dialect'], 
                                         target_samples_per_group=None,
                                         age_bins=None,
                                         max_duplications_per_sample=1000,
                                         random_seed=1234):
    """
    Create a balanced dataset by duplicating underrepresented samples.
    
    Args:
        audiopaths_data: List of [audiopath, sid, age, sex, dialect, text]
        lengths_data: List of corresponding lengths
        balance_by: List of attributes to balance by ['age', 'dialect', 'sex']
        target_samples_per_group: Target number of samples per group (None = max group size)
        age_bins: List of tuples [(18,30), (31,50), (51,70), (71,90)] or None for auto-binning
        max_duplications_per_sample: Maximum times a single sample can be duplicated
        random_seed: Random seed for reproducibility
    
    Returns:
        balanced_audiopaths: Balanced list of samples
        balanced_lengths: Corresponding lengths
        balance_stats: Statistics about the balancing
    """
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Creating balanced dataset by duplicating samples...")
    print(f"Original dataset size: {len(audiopaths_data)}")
    print(f"Balancing by: {balance_by}")
    
    if age_bins is None:
        age_bins = [(18, 30), (31, 45), (46, 60), (61, 90)]
    
    def get_age_bin(age):
        age = int(age)
        for i, (min_age, max_age) in enumerate(age_bins):
            if min_age <= age <= max_age:
                return f"age_{min_age}_{max_age}"
        return f"age_other"  
    
    def get_group_key(sample):
        audiopath, sid, age, sex, dialect, text = sample
        key_parts = []
        
        if 'age' in balance_by:
            key_parts.append(get_age_bin(age))
        if 'dialect' in balance_by:
            key_parts.append(f"dialect_{dialect}")
        if 'sex' in balance_by:
            key_parts.append(f"sex_{sex}")
        if 'speaker' in balance_by:
            key_parts.append(f"speaker_{sid}")
            
        return "_".join(key_parts)
    
    groups = defaultdict(list)
    group_lengths = defaultdict(list)
    
    for i, sample in enumerate(audiopaths_data):
        group_key = get_group_key(sample)
        groups[group_key].append(sample)
        group_lengths[group_key].append(lengths_data[i])
    
    print("\nOriginal distribution:")
    for group_key in sorted(groups.keys()):
        print(f"  {group_key}: {len(groups[group_key])} samples")
    
    group_sizes = [len(group) for group in groups.values()]
    if target_samples_per_group is None:
        target_size = max(group_sizes)
        print(f"\nTarget samples per group: {target_size} (maximum group size)")
    else:
        target_size = target_samples_per_group
        print(f"\nTarget samples per group: {target_size} (specified)")
    
    balanced_audiopaths = []
    balanced_lengths = []
    duplication_stats = {}
    
    for group_key, group_samples in groups.items():
        group_length_data = group_lengths[group_key]
        current_size = len(group_samples)
        
        if current_size == 0:
            continue
            
        # Calculate how many samples we need
        needed_samples = target_size - current_size
        
        if needed_samples <= 0:
            balanced_audiopaths.extend(group_samples)
            balanced_lengths.extend(group_length_data)
            duplication_stats[group_key] = {
                'original_size': current_size,
                'final_size': current_size,
                'duplications_added': 0,
                'avg_duplications_per_sample': 0
            }
        else:
            # Need to duplicate samples
            balanced_audiopaths.extend(group_samples)
            balanced_lengths.extend(group_length_data)
            
            duplications_per_sample = min(needed_samples // current_size, max_duplications_per_sample - 1)
            remaining_duplications = needed_samples % current_size
            
            duplications_added = 0
            sample_duplication_count = defaultdict(int)
            
            for _ in range(duplications_per_sample):
                for i, sample in enumerate(group_samples):
                    balanced_audiopaths.append(sample)
                    balanced_lengths.append(group_length_data[i])
                    sample_duplication_count[i] += 1
                    duplications_added += 1
            
            # Add remaining random duplications
            if remaining_duplications > 0:
                available_indices = []
                for i in range(current_size):
                    current_dups = sample_duplication_count[i]
                    max_additional = max_duplications_per_sample - 1 - current_dups
                    available_indices.extend([i] * max(max_additional, 1))
                
                if available_indices:
                    selected_indices = random.sample(
                        available_indices, 
                        min(remaining_duplications, len(available_indices))
                    )
                    
                    for idx in selected_indices:
                        balanced_audiopaths.append(group_samples[idx])
                        balanced_lengths.append(group_length_data[idx])
                        sample_duplication_count[idx] += 1
                        duplications_added += 1
            
            total_duplications = sum(sample_duplication_count.values())
            avg_dups = total_duplications / current_size if current_size > 0 else 0
            
            duplication_stats[group_key] = {
                'original_size': current_size,
                'final_size': current_size + duplications_added,
                'duplications_added': duplications_added,
                'avg_duplications_per_sample': avg_dups,
                'max_duplications_for_single_sample': max(sample_duplication_count.values()) if sample_duplication_count else 0
            }
    
    combined = list(zip(balanced_audiopaths, balanced_lengths))
    random.shuffle(combined)
    balanced_audiopaths, balanced_lengths = zip(*combined)
    balanced_audiopaths = list(balanced_audiopaths)
    balanced_lengths = list(balanced_lengths)
    


    return balanced_audiopaths, balanced_lengths



class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        
        audio, sampling_rate = load_wav_to_torch(filename)
        
        print(filename)
        print(f"audio.shape: {audio.shape}")
        print("sampling_rate: {sampling_rate}")
        # if audio.shape[-1] == 2:
        #     audio = audio[:, 0]
        
        
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
            spec = spec.clone().detach()

        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        
    
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text_path, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text_path)
        self.audiopaths_sid_text_path = audiopaths_sid_text_path
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 3)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
 
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()
        # self.audiopaths_sid_text = self.audiopaths_sid_text[len(self.audiopaths_sid_text)//4:]
        print("After filtering audiopaths_sid_text: ", len(self.audiopaths_sid_text))

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        path = self.audiopaths_sid_text_path.replace(".txt", ".balanced.pkl")
        if os.path.exists(path):
            print(f"Loading filtered audiopaths_sid_text from {path}")
            with open(path, 'rb') as f:
                combined = pickle.load(f)
            self.audiopaths_sid_text, self.lengths = zip(*combined)
        
            return None

        for audiopath, sid, age, sex, dialect, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, age, sex, dialect, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        

        
        if len(self.audiopaths_sid_text) > 10000:

            print(f"Filtered audiopaths_sid_text: {len(self.audiopaths_sid_text)} samples")

            self.audiopaths_sid_text, self.lengths = create_balanced_dataset_by_duplication(
                self.audiopaths_sid_text, 
                self.lengths,
                balance_by=['age', 'dialect'])
            
            
            combined = list(zip(self.audiopaths_sid_text, self.lengths))

            # Shuffle the combined list
            random.shuffle(combined)


            # Unzip them back into two lists
            
            print(f"After balancing, audiopaths_sid_text: {len(self.audiopaths_sid_text)} samples")
            
            with open(path, 'wb') as f:
                pickle.dump(combined, f)
            
            
            self.audiopaths_sid_text, self.lengths = zip(*combined)

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        # audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        # text = self.get_text(text)
        # spec, wav = self.get_audio(audiopath)
        # sid = self.get_sid(sid)
        # return (text, spec, wav, sid)

        audiopath, sid, age, sex, dialect, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], audiopath_sid_text[3], audiopath_sid_text[4], audiopath_sid_text[5]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        age = self.get_age(age)
        sex = self.get_sex(sex)
        dialect = self.get_dialect(dialect)
        alignments = self.get_alignments(audiopath)
        return (text, spec, wav, sid, age, sex, dialect, alignments)

    def get_audio(self, filename):
        if "en/" in filename:
            filename = filename.replace(".wav", "_22050_trimmed.wav")
            
        audio, sampling_rate = load_wav_to_torch(filename)
        # if audio.shape[0] < 1000:
            
        #     original_filename = filename.replace("_22050.wav", ".wav")

        #     ffmpeg_cmd = f'ffmpeg -i "{original_filename}" -ar 22050 -ac 1 "{filename}" -v quiet'

        #     result = os.system(ffmpeg_cmd)
        #     audio, sampling_rate = load_wav_to_torch(filename)
        #     assert audio.shape[0] > 1000, f"Audio file {filename} is empty or not found."
        # print(filename)
        # print(f"audio.shape: {audio.shape}")
        # print(f"sampling_rate: {sampling_rate}")

        # if audio.shape[-1] == 2:
        #     audio = audio[:, 0]
        # audio = audio[:, None]
        # print(f"audio.shape: {audio.shape}")
        # print(f"sampling_rate: {sampling_rate}")

        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} {} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))
            
            
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        spec_filename = filename.replace(".wav", ".spec.pt")
            
        if ".spec.pt" in spec_filename:
            if os.path.exists(spec_filename):
                try:
            # print("Loading spec from file: ", spec_filename)
                    spec = torch.load(spec_filename)
                except:
                    spec = spectrogram_torch(audio_norm, self.filter_length,
                    self.sampling_rate, self.hop_length, self.win_length,
                    center=False)
                    spec = torch.squeeze(spec, 0)
            # 
                    torch.save(spec, spec_filename)
    
            else:
                spec = spectrogram_torch(audio_norm, self.filter_length,
                    self.sampling_rate, self.hop_length, self.win_length,
                    center=False)
                spec = torch.squeeze(spec, 0)
        # 
                torch.save(spec, spec_filename)
        
        # assert spec.shape[-1] > 75, f"Spectrogram for {filename}, {spec.shape}, {audio.shape}. \n{text}"
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        try:
            num = int(sid)
            sid = torch.LongTensor([num])
        except:
            try:
                sid = np.load(sid)
                sid = torch.tensor(sid, dtype=torch.float)
            except Exception as e:
                print(f"Error loading sid {sid}: {e}")
        return sid
    # def get_sid(self, sid):
    #     sid = torch.tensor(np.load(sid))
    #     return sid
    
    def get_alignments(self, filename):
        path = filename.replace("_22050.wav", ".alignments.npy")
        alignments = np.load(path, allow_pickle=True)
        return torch.FloatTensor(alignments[0][0])
    
    
    def get_age(self, age):
        age = int(age)
        # age = random.randint(age, age + 7)
        age = torch.LongTensor([age])
        return age

    def get_sex(self, sex):
        sex = int(sex)
        sex = torch.LongTensor([sex])
        return sex
    def get_dialect(self, dialect):
        dialect = int(dialect) 
        dialect = torch.LongTensor([dialect])
        return dialect
    
    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        # print("Inside collect function")
        # print(batch[0][1].shape)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if batch[0][3].dim() == 1:
            sid = torch.LongTensor(len(batch))
        else:
            sid = torch.FloatTensor(len(batch), 192)
        age = torch.LongTensor(len(batch))
        sex = torch.LongTensor(len(batch))
        dialect = torch.LongTensor(len(batch))
        
        alignments_padded = torch.FloatTensor(len(batch),1,max_spec_len, max_text_len)  
        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        alignments_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]
            age[i] = row[4]
            sex[i] = row[5]
            dialect[i] = row[6]
            alignments_padded[i, :, :row[7].size(0), :row[7].size(1)] = row[7]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, age, sex, dialect, alignments_padded, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, age, sex, dialect, alignments_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          print(f"Bucket {i}: {len_bucket} samples, adding {rem} samples")
          
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
