
import streamlit as st
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import tempfile
import pandas as pd

with open("song_database.pkl", "rb") as f:
    song_database = pickle.load(f)

fingerprint_lookup = defaultdict(list)
for song, fps in song_database.items():
    for (f1, f2, dt, t_anchor) in fps:
        fingerprint_lookup[(f1, f2, dt)].append((song, t_anchor))

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
        if (f1, f2, dt) in fingerprint_lookup:
            for (song, t_db) in fingerprint_lookup[(f1, f2, dt)]:
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
                if (f1, f2, dt) in fingerprint_lookup:
                    for (song, t_db) in fingerprint_lookup[(f1, f2, dt)]:
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
