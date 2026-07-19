# ZeroTrust Health

![ZeroTrust Health Logo](public/ZeroTrustHealthLogo.png)

**ZeroTrust Health** is a decentralized cryptographic engine designed to enable hospital networks and medical institutions to train predictive AI models on highly sensitive patient data without ever exposing the raw data itself.

Powered by Secure Multi-Party Computation (MPC) via the Nillion network, this platform demonstrates how to run a breast cancer classification model across multiple healthcare providers in a completely "blind" state.

## 🚀 The Vision

In modern healthcare, data silos are a matter of life and death. Advances in AI are bottlenecked because institutions cannot legally or securely share diagnostic imaging data with one another due to strict privacy regulations (HIPAA, GDPR).

**ZeroTrust Health** solves this by moving from "Data Sharing" to "Data Collaboration." We ensure the world's most critical data type—medical diagnostics—can be leveraged globally while remaining strictly encrypted at rest, in transit, and during computation.

## 🧠 Architecture & Workflow

![Architecture Diagram](public/Diagram.png)

Our architecture is split into a localized baseline test and a decentralized cryptographic test:

### 1. Local Baseline Training
- Analyzes over 550 diagnostic instances containing 30 unique cellular parameters and 1 target diagnosis.
- Automatically handles an 80/20 train-test split.
- Generates regression coefficients (thetas) and local test predictions to serve as the ground truth.

### 2. Decentralized MPC Execution
- Simulates a fragmented healthcare system by dividing the dataset (25% to Provider A, 75% to Provider B).
- Utilizes a weighted average protocol to mathematically combine localized model weights within the encrypted state.
- Executes scaling factors dynamically to translate floating-point ML logic into secure, integer-based cryptographic circuits within the Nada DSL.

## 🔐 Core Infrastructure Stack

- **Nillion MPC Protocol**: The backbone for blind computation, decentralizing trust across multiple nodes.
- **Machine Learning Integration**: Leverages standardized diagnostic imaging cancer data from the UC Irvine repository.
- **Nada DSL**: Custom cryptographic circuits that execute the classification.

---

## 💻 Installation & Setup

### Requirements
- Python 3.11+
- Terminal / CLI

**1. Clone the repository**
```bash
git clone https://github.com/mohammadali-2000/ZeroTrust-Health.git
cd ZeroTrust-Health
```

**2. Configure Environment**
Copy the template and input your cryptographic keys (if applicable):
```bash
cp .env.example .env
```

**3. Initialize Virtual Environment**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**4. Execute the ZeroTrust Engine**
```bash
cd zerotrust_ml_core
python3 main_compute.py --disable_plot
```

*(Note: Nillion MPC network execution requires the local `nilup` SDK.)*
