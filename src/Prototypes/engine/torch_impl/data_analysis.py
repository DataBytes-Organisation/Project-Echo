# %% 

import os
import glob
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import joblib

# %%
AUDIO_EXTENSIONS = ['*.wav', '*.mp3', '*.flac', '*.ogg']

def get_audio_metadata(root_dir):
	"""
	Walks through the directory structure and collects metadata for audio files.
	Assumes subdirectories represent class labels (e.g., animal names).
	"""
	data = []

	print(f"Scanning directory: {root_dir}...")
	all_files = []
	for ext in AUDIO_EXTENSIONS:
		# Recursive search for audio files
		all_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))

	print(f"Found {len(all_files)} audio files. Extracting metadata...")

	for file_path in tqdm(all_files):
		try:
			folder_name = os.path.basename(os.path.dirname(file_path))

			info = sf.info(file_path)
			duration = info.duration
			samplerate = info.samplerate

			data.append({
				'path': file_path,
				'filename': os.path.basename(file_path),
				'label': folder_name,
				'duration': duration,
				'samplerate': samplerate
			})
		except Exception as e:
			print(f"Error processing {file_path}: {e}")
			continue

	return pd.DataFrame(data)

# %%
df = get_audio_metadata("b3")

# %%
df.head()

# %%
joblib.dump(df, "b3.joblib", compress=3)

# %%
def analyze_and_plot(df):
	"""
	Generates improved statistical summaries and plots with readable layouts.
	"""
	if df.empty:
		print("No data found.")
		return

	print("\n=== Dataset Summary ===")
	print(f"Total Files: {len(df)}")
	print(f"Total Classes: {df['label'].nunique()}")
	print(f"Total Duration: {df['duration'].sum() / 3600:.2f} hours")
	
	print("\n--- Sample Rate Stats ---")
	sr_counts = df['samplerate'].value_counts()
	print(sr_counts)
	dominant_sr = sr_counts.idxmax()
	print(f"\n-> Dominant Sample Rate: {dominant_sr} Hz")

	print("\n--- Duration Stats (seconds) ---")
	print(df['duration'].describe())

	sns.set_theme(style="whitegrid")
	
	# Calculate dynamic height based on number of classes
	# (Allocates 0.3 inches per class to ensure labels don't overlap)
	n_classes = df['label'].nunique()
	fig_height = max(12, n_classes * 0.3) 
	
	fig = plt.figure(figsize=(20, fig_height), constrained_layout=True)
	gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5]) 

	# PLOT 1: Count per Class (Horizontal)
	ax1 = fig.add_subplot(gs[0, 0])
	order = df['label'].value_counts().index
	sns.countplot(data=df, y='label', ax=ax1, palette='viridis', order=order)
	
	ax1.set_title(f'Files per Animal Class (Total: {n_classes})', fontsize=16)
	ax1.set_xlabel('Count')
	ax1.set_ylabel('') # Remove redundant label
	ax1.bar_label(ax1.containers[0], padding=3) # Add numbers at end of bars

	# PLOT 2: Duration Distribution (Log Scale)
	ax2 = fig.add_subplot(gs[0, 1])
	sns.histplot(data=df, x='duration', kde=True, ax=ax2, color='skyblue', log_scale=True)
	
	ax2.set_title('Duration Distribution (Log Scale)', fontsize=16)
	ax2.set_xlabel('Duration (Seconds) - Log Scale')
	
	# Add vertical lines for quartiles to see where "most" data is
	quantiles = df['duration'].quantile([0.25, 0.5, 0.75])
	for q in quantiles:
		ax2.axvline(q, color='red', linestyle='--', alpha=0.6)
	ax2.text(quantiles[0.5], ax2.get_ylim()[1]*0.9, f'Median: {quantiles[0.5]:.2f}s', color='red')

	# PLOT 3: Duration Variance (Horizontal Boxplot)
	ax3 = fig.add_subplot(gs[1, :]) # Spans full width
	
	sns.boxplot(data=df, x='duration', y='label', ax=ax3, palette='Set2', order=order)
	
	ax3.set_title('Duration Distribution by Animal', fontsize=16)
	ax3.set_xscale('log') 
	ax3.set_xlabel('Duration (Seconds) - Log Scale')
	ax3.set_ylabel('')

	# PLOT 4: Sample Rate (Donut Chart)
	ax4 = fig.add_subplot(gs[2, 0])
	
	sr_data = df['samplerate'].value_counts()
	
	wedges, texts, autotexts = ax4.pie(
		sr_data, 
		labels=sr_data.index, 
		autopct='%1.1f%%', 
		startangle=140, 
		colors=sns.color_palette('pastel'),
		pctdistance=0.85,
		wedgeprops=dict(width=0.3),
	)
	ax4.set_title('Sample Rate Distribution', fontsize=16)

	plt.show()
	return dominant_sr

# %%
dominant_sr = analyze_and_plot(df)

# %%
def suggest_audio_params(dominant_sr, target_resolution_ms=64):
	"""
	Calculates optimal spectrogram parameters based on the dataset's dominant sample rate.
	"""
	print("\n=== Recommended Audio Parameters ===")
	
	# n_fft (Window Size)
	# Formula: size = (sr * ms) / 1000
	# We round to the nearest power of 2 for FFT efficiency
	ideal_window = int(dominant_sr * (target_resolution_ms / 1000))
	n_fft = 2**int(np.round(np.log2(ideal_window)))
	
	# hop_length (Stride)
	# Standard overlap is 75%, so hop is 25% of n_fft
	hop_length = n_fft // 4
	
	# fmax (Nyquist Frequency)
	# The highest frequency we can mathematically represent is SR / 2
	fmax = int(dominant_sr / 2)
	
	# n_mels (Frequency Bins)
	n_mels = 260
	
	print(f"Based on Dominant Sample Rate: {dominant_sr} Hz")
	print("-" * 40)
	print(f"sample_rate: {dominant_sr}")
	print(f"n_fft      : {n_fft}  (Window: ~{n_fft/dominant_sr*1000:.1f} ms)")
	print(f"hop_length : {hop_length}   (Overlap: 75%)")
	print(f"n_mels     : {n_mels}   (Standard for CNNs)")
	print(f"fmin       : 20  (Standard)")
	print(f"fmax       : {fmax} (Nyquist Limit)")
	print("-" * 40)
	
	return {
		"sample_rate": dominant_sr,
		"n_fft": n_fft,
		"hop_length": hop_length,
		"n_mels": n_mels,
		"fmin": 20,
		"fmax": fmax
	}

# %%
suggest_audio_params(dominant_sr)
# %%
