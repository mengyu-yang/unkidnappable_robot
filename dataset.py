import os

import librosa
import numpy as np
import soundfile as sf
import torch

from torch.utils.data import Dataset
import pandas as pd
from data_utils import get_split


class HumanDetectionDataset(Dataset):
    def __init__(self, args, split="train", depth=False):
        self.args = args
        self.split = split
        self.depth = depth

        csv_file_path = os.path.join(args.data_dir, "data.csv")
        data_df = get_split(csv_file_path, args.test_room, split)
        empty_df = data_df[data_df["path"].str.contains("empty")]
        person_df = data_df[~data_df["path"].str.contains("empty")]

        # Subsample the data
        person_df = person_df.iloc[:: self.args.subsample, :]
        empty_df = empty_df.iloc[:: self.args.subsample_empty, :]

        # Person/no-person classification is not necessarily balanced so we can weight the loss
        self.cls_posw = empty_df.shape[0] / person_df.shape[0]

        # Mix person and empty data examples
        data_df = pd.concat([person_df, empty_df])

        # Initialize list of empty augmentation paths
        if split == "train":
            empty_aug_df = pd.read_csv(
                os.path.join(args.empty_data_dir, "empty_aug.csv")
            )

            self.empty_paths = empty_aug_df["path"]
            self.empty_lengths = empty_aug_df["length"]

        # Remove instances of standing still data
        if self.args.nostill:
            data_df = data_df[data_df["path"].str.contains("still") == False]

        # Read columns and convert to list
        self.sample_paths = data_df["path"].values.tolist()
        self.presences = data_df["person"].astype(float).tolist()
        self.centerpoints = data_df["centerpoint_x"].values.tolist()
        self.depths = data_df["depth"].astype(float).tolist()
        self.emptyconds = data_df["emptycond"].values.tolist()
        self.classes = data_df["classes"].astype("float64").tolist()

        # Don't train on empty data
        if args.no_empty:
            self.sample_paths = [
                path
                for path in self.sample_paths
                if "empty_dynamic" not in path and "empty_static" not in path
            ]

        # For caching calculated empty room spectrograms used for background subtraction
        self.empty_conditioning = {}

    def generate_spectrogram(self, audio, sr):
        num_aud_channels = audio.shape[1]

        specs = []
        for i in range(num_aud_channels):
            if self.args.use_mel:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio[:, i],
                    sr=sr,
                    n_fft=self.args.n_fft,
                    hop_length=self.args.hop_length,
                    win_length=self.args.win_length,
                    n_mels=self.args.n_mels,
                    fmin=self.args.fmin,
                    fmax=self.args.fmax,
                )
                if self.args.log_spec:
                    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                mel_spec = np.expand_dims(mel_spec, axis=0)
                specs.append(mel_spec)

            else:
                spectro = librosa.core.stft(
                    audio[:, i],
                    n_fft=self.args.n_fft,
                    hop_length=self.args.hop_length,
                    win_length=self.args.win_length,
                    center=True,
                )

                if self.args.log_spec:
                    real = librosa.amplitude_to_db(np.real(spectro), ref=np.max)
                    real = np.expand_dims(real, axis=0)
                else:
                    real = np.expand_dims(np.real(spectro), axis=0)

                imag = np.expand_dims(np.imag(spectro), axis=0)
                spectro_two_channel = np.concatenate((real, imag), axis=0)

                specs.append(spectro_two_channel)

        return specs

    def get_empty_conditioning(self, emptycond_subdir, empty_aug_file):
        # Calculate background subtraction spectrogram for empty augmentation
        if empty_aug_file is not None:
            # Calculate if we haven't already for this file
            if empty_aug_file not in self.empty_conditioning.keys():
                self.empty_conditioning[
                    empty_aug_file
                ] = self.calculate_empty_conditioning(empty_aug_file, is_synthetic=True)
            empty_aug_specs = self.empty_conditioning[empty_aug_file]

        # Calculate background subtraction spectrogram for real empty room
        if emptycond_subdir not in self.empty_conditioning.keys():
            self.empty_conditioning[
                emptycond_subdir
            ] = self.calculate_empty_conditioning(emptycond_subdir, is_synthetic=False)

        empty_specs = self.empty_conditioning[emptycond_subdir]

        if empty_aug_file is not None:
            # Interpolate between the two empty conditionings
            for i in range(len(empty_specs)):
                empty_specs[i] = (
                    empty_specs[i] * (1 - self.args.empty_w)
                    + empty_aug_specs[i] * self.args.empty_w
                )

        return empty_specs

    def calculate_empty_conditioning(self, empty_basename, is_synthetic):
        tr0_specs = []
        tr1_specs = []
        tr2_specs = []
        tr3_specs = []
        if is_synthetic:
            # Load the first 10 seconds of the file
            for i in range(20):
                empty_aud, sr = sf.read(
                    os.path.join(self.args.empty_data_dir, empty_basename),
                    start=i * 44100,
                    stop=(i + 1) * 44100,
                    dtype="float32",
                    always_2d=True,
                )
                empty_aud = self.normalize(empty_aud)

                # Calculate the spectrogram
                spectrograms = self.generate_spectrogram(empty_aud, sr)

                # Append to the list
                tr0_specs.append(spectrograms[0])
                tr1_specs.append(spectrograms[1])
                tr2_specs.append(spectrograms[2])
                tr3_specs.append(spectrograms[3])

        else:
            if "emptycond" in empty_basename:
                wav_file = os.path.join(
                    self.args.empty_data_dir, f"{empty_basename}.wav"
                )
                empty_aud, sr = sf.read(wav_file, dtype="float32", always_2d=True)
                sr = int(sr)
                empty_aud_length = empty_aud.shape[0] / sr

                # Load the first 25 seconds of the file
                for i in range(int(empty_aud_length)):
                    aud_clip = empty_aud[i * sr : (i + 1) * sr]
                    aud_clip = self.normalize(aud_clip)

                    # Calculate the spectrogram
                    spectrograms = self.generate_spectrogram(aud_clip, sr)

                    # Append to the list
                    tr0_specs.append(spectrograms[0])
                    tr1_specs.append(spectrograms[1])
                    tr2_specs.append(spectrograms[2])
                    tr3_specs.append(spectrograms[3])
            else:
                # Get all empty files from basename
                empty_files = [
                    file for file in self.sample_paths if empty_basename in file
                ]
                empty_files = sorted(empty_files)

                empty_files = empty_files[: int(20 / (self.args.subsample_empty / 4))]

                for empty_file in empty_files:
                    # Get the corresponding wav file
                    wav_file = os.path.join(self.args.data_dir, empty_file, "audio.wav")
                    empty_aud, sr = sf.read(wav_file, dtype="float32", always_2d=True)
                    empty_aud = self.normalize(empty_aud)

                    # Calculate the spectrogram
                    spectrograms = self.generate_spectrogram(empty_aud, sr)

                    # Append to the list
                    tr0_specs.append(spectrograms[0])
                    tr1_specs.append(spectrograms[1])
                    tr2_specs.append(spectrograms[2])
                    tr3_specs.append(spectrograms[3])

        # Calculate the mean spectrogram
        tr0_specs = np.stack(tr0_specs, axis=0)
        tr1_specs = np.stack(tr1_specs, axis=0)
        tr2_specs = np.stack(tr2_specs, axis=0)
        tr3_specs = np.stack(tr3_specs, axis=0)

        mean_tr0_spec = np.mean(tr0_specs, axis=0)
        mean_tr1_spec = np.mean(tr1_specs, axis=0)
        mean_tr2_spec = np.mean(tr2_specs, axis=0)
        mean_tr3_spec = np.mean(tr3_specs, axis=0)

        if self.args.mean_col:
            # Calculate mean column
            mean_tr0_spec = np.mean(mean_tr0_spec, axis=-1)
            mean_tr1_spec = np.mean(mean_tr1_spec, axis=-1)
            mean_tr2_spec = np.mean(mean_tr2_spec, axis=-1)
            mean_tr3_spec = np.mean(mean_tr3_spec, axis=-1)

            # Expand last dimension
            mean_tr0_spec = np.expand_dims(mean_tr0_spec, axis=-1)
            mean_tr1_spec = np.expand_dims(mean_tr1_spec, axis=-1)
            mean_tr2_spec = np.expand_dims(mean_tr2_spec, axis=-1)
            mean_tr3_spec = np.expand_dims(mean_tr3_spec, axis=-1)

        # Convert to tensor
        mean_tr0_spec = torch.from_numpy(mean_tr0_spec).float()
        mean_tr1_spec = torch.from_numpy(mean_tr1_spec).float()
        mean_tr2_spec = torch.from_numpy(mean_tr2_spec).float()
        mean_tr3_spec = torch.from_numpy(mean_tr3_spec).float()

        return [mean_tr0_spec, mean_tr1_spec, mean_tr2_spec, mean_tr3_spec]

    def augment_audio(self, audio, random_factor=None):
        if random_factor == None:
            random_factor = np.random.random() + 0.5  # 0.5 - 1.5

        audio = audio * random_factor
        audio[audio > 1.0] = 1.0
        audio[audio < -1.0] = -1.0
        return audio

    def normalize(self, audio, desired_rms=0.02, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(audio**2)))
        return audio * (desired_rms / rms)

    def __len__(self):
        if self.split == "train":
            return int(len(self.sample_paths) * self.args.data_subset)
        else:
            return int(len(self.sample_paths))

    def __getitem__(self, idx):
        empty_augment = False
        subdir = self.sample_paths[idx]

        # Azimuth angle
        centerpoint_x = self.centerpoints[idx]

        # Process depth ground truth
        depth = self.depths[idx]
        raw_depth = depth
        if self.args.binary_depth:
            if depth != -1:
                depth = 1 if depth >= self.args.depth_threshold else 0
        elif self.args.multiclass_depth:
            if depth != -1:
                depth = np.digitize(depth, self.bins) - 1
                depth = torch.from_numpy(np.array([depth]))

        # Data prefix for corresponding empty room recording
        emptycond_subdir = self.emptyconds[idx]

        # Binary person (1) or empty (0) label
        person = self.presences[idx]

        # Data classes. Empty (0), standing still (1), quiet walking (2), normal walking (3), loud walking (4)
        classes = self.classes[idx]

        # Convert centerpoint_x into sin/cos embedding
        centerpoint_sin = np.sin(2 * np.pi * centerpoint_x / 1440)
        centerpoint_cos = np.cos(2 * np.pi * centerpoint_x / 1440)

        # With probability p, augment with synthetic empty audio
        if np.random.random() < self.args.empty_aug_prob and self.split == "train":
            empty_augment = True
            # Randomly select an index from the list of empty files
            empty_idx = np.random.randint(len(self.empty_paths))

            # Randomly select a start time
            empty_start_time = np.random.randint(
                20, self.empty_lengths[empty_idx] - self.args.clip_len - 1
            )

            # Get empty audio clip
            empty_aud_start = int(empty_start_time * 44100)
            empty_aud_end = empty_aud_start + int(self.args.clip_len * 44100)

            empty_aud, sr = sf.read(
                os.path.join(self.args.empty_data_dir, self.empty_paths[empty_idx]),
                start=empty_aud_start,
                stop=empty_aud_end,
                dtype="float32",
                always_2d=True,
            )
            empty_aud = self.normalize(empty_aud)

        # Load specs
        aud_clip, sr = sf.read(
            os.path.join(self.args.data_dir, subdir, "audio.wav"),
            dtype="float32",
            always_2d=True,
        )
        aud_clip = self.normalize(aud_clip)

        if empty_augment:
            # Augment with additional empty room noise
            if self.args.rand_empty_w:
                self.args.empty_w = np.random.random()

            aud_clip = (
                1 - self.args.empty_w
            ) * aud_clip + self.args.empty_w * empty_aud
            aud_clip = self.normalize(aud_clip)

        if self.args.aug_wave:
            # Augment waveform with random factor
            aud_clip = self.augment_audio(aud_clip)

        if self.args.add_noise:
            # Add noise to audio
            noise = np.random.randn(len(aud_clip)) * self.args.noise_factor
            noise = np.expand_dims(noise, axis=1)
            aud_clip = aud_clip + noise

        # Calculate spectrograms
        spectrograms = self.generate_spectrogram(aud_clip, sr)

        # Convert spectrograms to tensors
        spectrograms = [torch.from_numpy(spec).float() for spec in spectrograms]

        centerpoint_x = 1800 if person == 0 else centerpoint_x

        # Do background subtraction
        if self.args.background_sub:
            if empty_augment:
                empty_spectrograms = self.get_empty_conditioning(
                    emptycond_subdir, self.empty_paths[empty_idx]
                )
            else:
                empty_spectrograms = self.get_empty_conditioning(emptycond_subdir, None)

            if self.args.learn_backsub:
                if self.depth:
                    return (
                        spectrograms,
                        empty_spectrograms,
                        person,
                        classes,
                        centerpoint_sin,
                        centerpoint_cos,
                        centerpoint_x,
                        depth,
                        raw_depth,
                        subdir,
                    )
                else:
                    return (
                        spectrograms,
                        empty_spectrograms,
                        person,
                        classes,
                        centerpoint_sin,
                        centerpoint_cos,
                        centerpoint_x,
                        subdir,
                    )
            else:
                for spec, empty_spec in zip(spectrograms, empty_spectrograms):
                    spec = spec - self.args.backsub_w * empty_spec

        if self.depth:
            return (
                spectrograms,
                person,
                classes,
                centerpoint_sin,
                centerpoint_cos,
                centerpoint_x,
                depth,
                raw_depth,
                subdir,
            )
        else:
            return (
                spectrograms,
                person,
                centerpoint_sin,
                centerpoint_cos,
                centerpoint_x,
                subdir,
            )
