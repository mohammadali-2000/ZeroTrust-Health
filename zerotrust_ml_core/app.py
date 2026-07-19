import streamlit as st
import pandas as pd
import numpy as np
import tenseal as ts
import os
import time

st.set_page_config(
    page_title="ZeroTrust Patient Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e66ff;
        color: white;
        border-radius: 5px;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1a4dff;
        border-color: #1a4dff;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load and train the model locally
@st.cache_data
def prepare_model():
    script_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(script_directory, "data", "cancer_data.csv")
    
    # Load dataset
    health_data = pd.read_csv(csv_file_path)
    health_data["Diagnosis"] = health_data["Diagnosis"].apply(lambda x: 1 if x == "M" else 0)
    health_data = health_data.drop("ID", axis=1)

    # Split Dataset
    train_size = int(0.8 * len(health_data))
    train_data = health_data[:train_size]
    test_data = health_data[train_size:]

    # Subsets (as per the zero-trust fragmented architecture)
    subset_25_size = int(0.25 * len(train_data))
    subset_75_size = int(0.75 * len(train_data))
    subset_sm_data = train_data[:subset_25_size]
    subset_lg_data = train_data[subset_25_size:]

    # Arrays
    X_train_subset_sm = subset_sm_data.drop("Diagnosis", axis=1).values
    X_train_subset_lg = subset_lg_data.drop("Diagnosis", axis=1).values
    y_train_subset_sm = subset_sm_data["Diagnosis"].values
    y_train_subset_lg = subset_lg_data["Diagnosis"].values

    # Augmented matrices
    X_train_subset_sm_aug = np.hstack((np.ones((X_train_subset_sm.shape[0], 1)), X_train_subset_sm))
    X_train_subset_lg_aug = np.hstack((np.ones((X_train_subset_lg.shape[0], 1)), X_train_subset_lg))

    # Thetas (Normal Equation)
    theta_subset_sm = np.linalg.inv(X_train_subset_sm_aug.T @ X_train_subset_sm_aug) @ X_train_subset_sm_aug.T @ y_train_subset_sm
    theta_subset_lg = np.linalg.inv(X_train_subset_lg_aug.T @ X_train_subset_lg_aug) @ X_train_subset_lg_aug.T @ y_train_subset_lg

    # Combined Theta
    weights_sm = subset_25_size / train_size
    weights_lg = subset_75_size / train_size
    combined_theta = (theta_subset_sm * weights_sm) + (theta_subset_lg * weights_lg)

    return test_data, combined_theta

st.title("🛡️ ZeroTrust Patient Monitor")
st.markdown("### Edge-Encrypted AI Anomaly Detection via Fully Homomorphic Encryption (FHE)")
st.markdown("---")

test_data, combined_theta = prepare_model()

# Sidebar: Patient Selection
st.sidebar.header("Wearable IoT Edge Device")
st.sidebar.markdown("Simulate a patient's IoT wearable sending a stream of biometric diagnostic data.")

# Generate a random patient stream seed so the user can click a button to "fetch next patient stream"
if "patient_seed" not in st.session_state:
    st.session_state.patient_seed = 42

if st.sidebar.button("📡 Fetch Next Patient Stream"):
    st.session_state.patient_seed += 1

random_row = test_data.sample(n=1, random_state=st.session_state.patient_seed)
X_test = random_row.drop("Diagnosis", axis=1).values
X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
X_test_values = X_test_augmented[0]

true_diagnosis = "Malignant Anomaly" if random_row["Diagnosis"].values[0] == 1 else "Benign (Healthy)"
plaintext_prediction = X_test_values.dot(combined_theta)
predicted_class = "Malignant Anomaly" if round(plaintext_prediction) == 1 else "Benign (Healthy)"

st.sidebar.markdown(f"**Actual Ground Truth:** `{true_diagnosis}`")
st.sidebar.markdown("*(The AI should predict this without ever seeing the raw data)*")

st.markdown("#### 1. Captured Biometric Stream (Raw Data)")
st.markdown("This is the raw data captured by the IoT device. Under normal circumstances, this highly sensitive data would be sent in plaintext to a cloud server, violating privacy.")
st.dataframe(random_row.drop("Diagnosis", axis=1), use_container_width=True)

st.markdown("---")
st.markdown("#### 2. ZeroTrust FHE Pipeline")

if st.button("🔒 Execute Encrypted AI Diagnostics"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Edge Device (Client)")
        with st.status("Initializing TenSEAL Cryptographic Context...", expanded=True) as status1:
            time.sleep(0.8)
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            context.global_scale = 2**40
            context.generate_galois_keys()
            st.write("✅ CKKS Context Generated")
            
            time.sleep(0.8)
            st.write("Encrypting biometric stream into ciphertext...")
            encrypted_patient_data = ts.ckks_vector(context, X_test_values.tolist())
            time.sleep(1)
            st.write(f"✅ Data Encrypted: `{str(encrypted_patient_data)[:40]}...`")
            status1.update(label="Edge Device encryption complete!", state="complete", expanded=False)
            
    with col2:
        st.markdown("##### AI Cloud Server")
        with st.status("Waiting for Encrypted Payload...", expanded=True) as status2:
            time.sleep(2.5) # Wait for edge device to encrypt
            st.write("📥 Received Ciphertext payload from edge device.")
            
            time.sleep(1.2)
            st.write("🧠 Computing AI model (Dot Product) on Ciphertext...")
            # FHE dot product
            encrypted_prediction = encrypted_patient_data.dot(combined_theta.tolist())
            time.sleep(1)
            st.write(f"✅ Prediction computed mathematically in encrypted space!")
            status2.update(label="Server Blind Computation complete!", state="complete", expanded=False)

    st.markdown("---")
    st.markdown("#### 3. Decrypted Results")
    st.markdown("The server returns the encrypted prediction to the edge device, which decrypts it locally using its private key.")
    
    with st.spinner("Decrypting payload..."):
        time.sleep(1)
        decrypted_prediction = encrypted_prediction.decrypt()[0]
        fhe_class = "Malignant Anomaly" if round(decrypted_prediction) == 1 else "Benign (Healthy)"
        
    mcol1, mcol2, mcol3 = st.columns(3)
    
    with mcol1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="FHE AI Diagnosis", value=fhe_class)
        st.markdown("</div>", unsafe_allow_html=True)
    with mcol2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="FHE Raw Output", value=f"{decrypted_prediction:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with mcol3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Plaintext Baseline", value=f"{plaintext_prediction:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    if abs(decrypted_prediction - plaintext_prediction) < 0.001:
        st.success("✅ cryptographic proof successful: FHE encrypted prediction mathematically perfectly matches the plaintext baseline prediction.")
    else:
        st.error("Error: Precision degradation detected in FHE context.")
