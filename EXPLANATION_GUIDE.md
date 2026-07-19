# ZeroTrust Health: Explanation Guide (FHE Edition)

This guide breaks down the core concepts behind **ZeroTrust Health**, detailing how it transitions from a traditional Machine Learning script into a **Fully Homomorphic Encryption (FHE)** engine.

## 1. What is Fully Homomorphic Encryption?
Normally, to perform calculations on data, the data must be decrypted first. This represents a massive security flaw in healthcare; if a hospital sends data to an AI provider for diagnostics, the AI provider gains full access to that raw medical data during computation.

**Fully Homomorphic Encryption (FHE)** solves this by allowing mathematical operations (like addition and multiplication) to be performed *directly on the ciphertext*. The AI model evaluates the data while it is entirely encrypted, and produces an encrypted prediction. Only the original key holder can decrypt the final result.

## 2. Microsoft TenSEAL & The CKKS Scheme
ZeroTrust Health utilizes **TenSEAL**, an open-source FHE library built by Microsoft. 

Under the hood, TenSEAL implements the **CKKS (Cheon-Kim-Kim-Song)** encryption scheme. Unlike other FHE schemes that only work on integers or booleans, CKKS is specifically designed for approximate arithmetic on floating-point numbers. This makes it the absolute perfect choice for Machine Learning, where data points and model weights (thetas) are highly precise floating-point vectors.

## 3. The Execution Flow

When you run `python3 main_compute.py`, the following sequence occurs:

1. **Local Data Processing:** The script loads the breast cancer dataset and splits it into subsets to simulate fragmented healthcare data.
2. **Local Model Training:** A linear regression model is trained locally using the Normal Equation, generating the "model weights" (thetas).
3. **Data Encryption:** A random test patient's row of diagnostic data is selected. Instead of processing it normally, the script uses the TenSEAL `context` to encrypt this data into a `CKKSVector` ciphertext.
4. **Blind Computation:** The script computes the dot product of the *encrypted patient data* and the *plaintext model weights*. This simulates the AI evaluating the patient blindly.
5. **Decryption:** The resulting encrypted prediction is decrypted, and we verify that it matches the exact prediction the model would have made if the data was never encrypted.

## 4. Why this matters for Web3
Projects bridging FHE and Web3 (like Fhenix) represent the frontier of decentralized privacy. By wrapping ML pipelines in FHE, we can theoretically deploy these encrypted AI models to blockchains, smart contracts, or decentralized nodes, ensuring absolute data privacy in public environments.
