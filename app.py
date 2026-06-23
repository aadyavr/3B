import streamlit as st
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import tempfile
import pandas as pd
import os
import gdown

# Download data files from Google Drive if not present
if not os.path.exists("song_database.json"):
    st.info("Loading song database...")
    gdown.download(id="17ud8YmF-Yv4x27A4849D2XlnpKHNXaYo", output="song_database.json", quiet=False)

if not os.path.exists("fingerprint_lookup.json"):
    st.info("Loading fingerprint lookup...")
    gdown.download(id="1j7yB7V9X2bxzt2ep0RRdvcmNz0rgyqYl", output="fingerprint_lookup.json", quiet=False)

# Load database
with open("song_database.json", "r") as f:
    song_database = json.load(f)
print("Loaded song_database, entries:", len(song_database))

with open("fingerprint_lookup.json", "r") as f:
    fingerprint_lookup = json.load(f)
print("Loaded fingerprint_lookup, entries:", len(fingerprint_lookup))

def create_fingerprints(path):
    y, sr = librosa.load(path, sr=None)
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)))
    local_max = maximum_filter(S_db, size=20) == S_db
    threshold = np.percentile(S_db, 98)
    peaks = np.where(local_max & (S_db > threshold))
    freq_idx, time_idx = peaks[0], peaks[1]
    fps = []
    for i in range(len(time_idx)):
        for j in range(i+1, min(i+5, len(time_idx))):
            dt = int(time_idx[j] - time_idx[i])
            if dt > 0:
                fps.append((int(freq_idx[i]), int(freq_idx[j]), dt, int(time_idx[i])))
    return fps, y, sr, S_db

def match_song(query_fps):
    offset_counts = defaultdict(lambda: defaultdict(int))
    for (f1, f2, dt, t_q) in query_fps:
        key = f"{f1}_{f2}_{dt}"
        if key in fingerprint_lookup:
            for song, t_db in fingerprint_lookup[key]:
                offset_counts[song][t_db - t_q] += 1
    scores = {song: max(offsets.values()) for song, offsets in offset_counts.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

st.title("EE200: Audio Fingerprinting")
mode = st.radio("Select Mode", ["Single Clip", "Batch Mode"])

if mode == "Single Clip":
    uploaded = st.file_uploader("Upload audio clip", type=["mp3","wav","flac"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        fps, y, sr, S_db = create_fingerprints(tmp_path)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(S_db, origin='lower', aspect='auto')
        ax.set_title("Spectrogram")
        st.pyplot(fig)
        results = match_song(fps)
        if results:
            best_song = results[0][0]
            st.success(f"Identified: {best_song}")
            offsets = []
            for (f1, f2, dt, t_q) in fps:
                key = f"{f1}_{f2}_{dt}"
                if key in fingerprint_lookup:
                    for song, t_db in fingerprint_lookup[key]:
                        if song == best_song:
                            offsets.append(t_db - t_q)
            fig2, ax2 = plt.subplots()
            ax2.hist(offsets, bins=50)
            ax2.set_title("Offset Histogram")
            st.pyplot(fig2)

elif mode == "Batch Mode":
    uploaded_files = st.file_uploader("Upload multiple clips",
                                       type=["mp3","wav","flac"],
                                       accept_multiple_files=True)
    if uploaded_files and st.button("Run Batch"):
        results_list = []
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            fps, _, _, _ = create_fingerprints(tmp_path)
            results = match_song(fps)
            prediction = results[0][0].replace(".mp3","").replace(".wav","") if results else "unknown"
            results_list.append({"filename": f.name, "prediction": prediction})
        df = pd.DataFrame(results_list)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("Download results.csv", csv, "results.csv")
