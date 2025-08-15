import streamlit as st
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, savgol_filter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from math import pi
from PIL import Image
from tensorflow.keras.models import load_model
st.title("ECG → Filtrage → FrFT → Image 224x224 → Classification")

st.subheader("📖 Description des arythmies")
st.write("Cette application classe les battements ECG selon leur type d’arythmie :")

st.write("""
- **N** : NORMAL  
- **S** : SUPRAVENTICULAR  
- **V** : VENTRICULAR  
- **F** : FUSION  
- **Q** : UNKNOWN
""")

# --- Classes réelles ---
CLASS_NAMES = ['F3', 'N0', 'Q4', 'S1', 'V2']  # Remplace par tes vraies classes

# --- Filtre passe-bande ---
def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# --- Filtre Savitzky-Golay ---
def smooth_signal(signal, window_length=11, polyorder=3):
    return savgol_filter(signal, window_length, polyorder)

# --- FrFT optimisée ---
def frft(f, a):
    N = len(f)
    shft = np.arange(N)
    shft = np.where(shft > N/2, shft - N, shft)
    
    alpha = a * pi / 2
    if a == 0:
        return f
    if a == 1:
        return np.fft.fft(f)
    if a == 2:
        return np.flipud(f)
    if a == -1:
        return np.fft.ifft(f)

    tana2 = np.tan(alpha/2)
    sina = np.sin(alpha)

    chirp1 = np.exp(-1j * pi * (shft**2) * tana2 / N)
    f = f * chirp1

    F = np.fft.fft(f * np.exp(-1j * pi * (shft**2) / (N * sina)))
    F = F * np.exp(-1j * pi * (shft**2) * tana2 / N)

    return F

# --- Charger le modèle ---
MODEL_PATH = "best_model_single.h5"
model = load_model(MODEL_PATH)

# --- Interface Streamlit ---
st.title("ECG → Filtrage → FrFT → Image 224x224 → Classification")

uploaded_file = st.file_uploader("Chargez un fichier ECG (.mat ou .csv)", type=["mat", "csv"])
fs = st.number_input("Fréquence d'échantillonnage (Hz)", value=360)
fraction_order = st.slider("Ordre de la FrFT (a)", 0.0, 2.0, 1.0, 0.1)

if uploaded_file is not None:
    # Charger le signal
    if uploaded_file.name.endswith(".mat"):
        mat_data = sio.loadmat(uploaded_file)
        for key in mat_data.keys():
            if not key.startswith("__"):
                signal = np.ravel(mat_data[key])
                break
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        signal = df.iloc[:, 0].values

    st.subheader("Signal brut")
    st.line_chart(signal)

    # Filtrage
    filtered = bandpass_filter(signal, 0.5, 50, fs)
    filtered = smooth_signal(filtered)

    st.subheader("Signal filtré")
    st.line_chart(filtered)

    # FrFT
    frft_result = frft(filtered, fraction_order)
    magnitude = np.abs(frft_result)

    # Génération image 224x224
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.plot(magnitude)
    plt.tight_layout(pad=0)

    # Conversion figure -> numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # Garder seulement RGB
    img_array = img_array[:, :, :3]

    # Conversion en PIL + redimensionnement 224x224
    img_pil = Image.fromarray(img_array).resize((224, 224))
    st.image(img_pil, caption="Image 224x224 générée", use_container_width=True)

    # Préparation pour le modèle
    img_input = np.array(img_pil) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Prédiction
    predictions = model.predict(img_input)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_index]

    st.subheader("Résultat de la classification")
    st.write("Classe prédite :", predicted_class)

    # Affichage des probabilités par classe
    st.subheader("📊 Probabilités par classe")
    all_probs = predictions[0]
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name} : {all_probs[i]*100:.2f}%")
