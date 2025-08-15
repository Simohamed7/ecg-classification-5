import streamlit as st
import numpy as np
import pandas as pd
import scipy.io as sio
import pywt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from tensorflow.keras.models import load_model
from math import pi

# --- Classes réelles ---
CLASS_NAMES = ['F3', 'N0', 'Q4', 'S1', 'V2']  # Remplace par tes vraies classes

# --- Charger le modèle ---
MODEL_PATH = "best_model_single.h5"
model = load_model(MODEL_PATH)

# --- Streamlit ---
st.title("ECG → Wavelet db4 → PCA → Savitzky-Golay → FrFT → Image 224x224 → Classification")

st.subheader("📖 Description des arythmies")
st.write("Cette application classe les battements ECG selon leur type d’arythmie :")
st.write("""
- **N** : NORMAL  
- **S** : SUPRAVENTICULAR  
- **V** : VENTRICULAR  
- **F** : FUSION  
- **Q** : UNKNOWN
""")

uploaded_file = st.file_uploader("Chargez un fichier ECG (.mat ou .csv)", type=["mat", "csv"])
fs = st.number_input("Fréquence d'échantillonnage (Hz)", value=360)
fraction_order = st.slider("Ordre de la FrFT (a)", 0.01, 1.0, 0.5, 0.01)

# --- FrFT fonction ---
def frft(f, a):
    N = len(f)
    shft = np.arange(N)
    shft = np.where(shft > N/2, shft - N, shft)
    alpha = a * pi / 2
    if a == 0: return f
    if a == 1: return np.fft.fft(f)
    if a == 2: return np.flipud(f)
    if a == -1: return np.fft.ifft(f)
    tana2 = np.tan(alpha/2)
    sina = np.sin(alpha)
    chirp1 = np.exp(-1j * pi * (shft**2) * tana2 / N)
    f = f * chirp1
    F = np.fft.fft(f * np.exp(-1j * pi * (shft**2) / (N * sina)))
    F = F * np.exp(-1j * pi * (shft**2) * tana2 / N)
    return F

if uploaded_file is not None:
    # --- Charger signal ---
    if uploaded_file.name.endswith(".mat"):
        mat_data = sio.loadmat(uploaded_file)
        for key in mat_data.keys():
            if not key.startswith("__"):
                signal = np.ravel(mat_data[key])
                break
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        signal = df.iloc[:,0].values

    # --- 1️⃣ Signal brut ---
    st.subheader("Signal brut")
    st.line_chart(signal)

    # --- 2️⃣ Filtrage Wavelet db4 ---
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    filtered_signal = pywt.waverec(coeffs, 'db4')

    # --- 3️⃣ Savitzky-Golay ---
    smoothed_signal = savgol_filter(filtered_signal, window_length=11, polyorder=3)

    st.subheader("Signal filtré (Wavelet + Savitzky-Golay)")
    st.line_chart(smoothed_signal)

    # --- 4️⃣ PCA (optionnel, pour réduire dimension avant FrFT) ---
    pca_signal = PCA(n_components=1).fit_transform(smoothed_signal.reshape(-1,1)).ravel()
    # st.subheader("Signal après PCA")
    # st.line_chart(pca_signal)  # tu peux activer si tu veux voir PCA

    # --- 5️⃣ FrFT ---
    frft_signal = frft(pca_signal, fraction_order)
    magnitude = np.abs(frft_signal)

    # --- 6️⃣ Normalisation et centrage ---
    scaler = StandardScaler()
    magnitude_normalized = scaler.fit_transform(magnitude.reshape(-1,1)).ravel()

    # --- 7️⃣ Génération image 224x224 ---
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.plot(magnitude_normalized)
    plt.tight_layout(pad=0)

    canvas = FigureCanvas(fig)
    canvas.draw()
    img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1]+(4,))
    plt.close(fig)

    img_array = img_array[:,:,:3]  # RGB
    img_pil = Image.fromarray(img_array).resize((224,224))
    st.image(img_pil, caption="Image 224x224 générée", use_container_width=True)

    # --- 8️⃣ Préparer pour modèle ---
    img_input = np.array(img_pil)/255.0
    img_input = np.expand_dims(img_input, axis=0)

    # --- 9️⃣ Classification ---
    predictions = model.predict(img_input)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_index]

    st.subheader("Résultat de la classification")
    st.write("Classe prédite :", predicted_class)

    # --- 10️⃣ Affichage probabilités ---
    st.subheader("📊 Probabilités par classe")
    all_probs = predictions[0]
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name} : {all_probs[i]*100:.2f}%")
