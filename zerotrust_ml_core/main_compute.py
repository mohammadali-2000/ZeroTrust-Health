import numpy as np
import pandas as pd
from utils.func import (
    plot_distributions,
    compute_prediction,
)
import os
import sys
from dotenv import load_dotenv
import argparse
import tenseal as ts

load_dotenv()


def main():
    print("\n\n******* ZeroTrust Health FHE Engine *******\n\n")
    
    # Create a parser
    parser = argparse.ArgumentParser(description="Check if plot is disabled")
    parser.add_argument("--disable_plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    # Load dataset
    script_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(script_directory, "data", "cancer_data.csv")
    health_data = pd.read_csv(csv_file_path)

    # Replace values in the 'Diagnosis' column
    health_data["Diagnosis"] = health_data["Diagnosis"].apply(
        lambda x: 1 if x == "M" else 0
    )

    # Remove ID column
    health_data = health_data.drop("ID", axis=1)

    # Split Dataset, 80 percent training, 20 percent testing
    train_size = int(0.8 * len(health_data))
    train_data = health_data[:train_size]
    test_data = health_data[train_size:]

    # Create subsets
    subset_25_size = int(0.25 * len(train_data))
    subset_75_size = int(0.75 * len(train_data))
    subset_sm_data = train_data[:subset_25_size]
    subset_lg_data = train_data[subset_25_size:]

    if not args.disable_plot:
        # Plot datasets
        plot_distributions(train_data, 7, 2, 25, "Distribution Plots of Training Data")
        plot_distributions(
            subset_sm_data, 7, 2, 25, "Distribution Plots of Subset - Sm"
        )
        plot_distributions(
            subset_lg_data, 7, 2, 25, "Distribution Plots of Subset - Lg"
        )

    # Set up data for training
    X_train = train_data.drop("Diagnosis", axis=1).values
    X_train_subset_sm = subset_sm_data.drop("Diagnosis", axis=1).values
    X_train_subset_lg = subset_lg_data.drop("Diagnosis", axis=1).values
    y_train = train_data["Diagnosis"].values
    y_train_subset_sm = subset_sm_data["Diagnosis"].values
    y_train_subset_lg = subset_lg_data["Diagnosis"].values

    # Select a random test_data and drop 'Diagnosis' column
    random_row = test_data.sample(n=1, random_state=42)
    print("\nTest instance from test_data before removing target field (Diagnosis).")
    print("Data will be used in blind computation test.")
    print(random_row)
    X_test = random_row.drop("Diagnosis", axis=1).values

    # Add a column of ones to X_train for the intercept term
    X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_train_subset_sm_augmented = np.hstack(
        (np.ones((X_train_subset_sm.shape[0], 1)), X_train_subset_sm)
    )
    X_train_subset_lg_augmented = np.hstack(
        (np.ones((X_train_subset_lg.shape[0], 1)), X_train_subset_lg)
    )

    # Calculate coefficients using the normal equation
    theta = (
        np.linalg.inv(X_train_augmented.T @ X_train_augmented)
        @ X_train_augmented.T
        @ y_train
    )
    theta_subset_sm = (
        np.linalg.inv(X_train_subset_sm_augmented.T @ X_train_subset_sm_augmented)
        @ X_train_subset_sm_augmented.T
        @ y_train_subset_sm
    )
    theta_subset_lg = (
        np.linalg.inv(X_train_subset_lg_augmented.T @ X_train_subset_lg_augmented)
        @ X_train_subset_lg_augmented.T
        @ y_train_subset_lg
    )

    # Compute prediction value using combined thetas from subsets
    weights_subset_sm = subset_25_size / train_size
    weights_subset_lg = subset_75_size / train_size

    combined_theta = (theta_subset_sm * weights_subset_sm) + (
        theta_subset_lg * weights_subset_lg
    )
    print("\nCombined theta via weighted average of train datasets:")
    print(combined_theta)
    
    # Predictions for the test data
    X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    X_test_values = X_test_augmented[0]

    Y_prediction = X_test_values.dot(theta)
    print("\nPlaintext Y_Prediction value and classification on the full test data:")
    print(f"{Y_prediction}, classifying as {'M' if round(Y_prediction) == 1 else 'B'}")

    #############################################
    ############# TenSEAL FHE Section ###########
    #############################################

    print("\n*******************************")
    print("**** FHE Encrypted Compute ****")
    print("*******************************\n")

    # 1. Setup TenSEAL Context for CKKS Scheme
    print("Initializing TenSEAL Cryptographic Context (CKKS Scheme)...")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    
    # 2. Encrypt the Patient Data
    print("\nEncrypting the Patient's Diagnostic Data...")
    encrypted_patient_data = ts.ckks_vector(context, X_test_values.tolist())
    print(f"Encrypted Data Object: {encrypted_patient_data}")
    
    # 3. Perform Blind Computation (Dot Product of Encrypted Data with Combined Model Weights)
    print("\nExecuting Blind Computation on the Encrypted Patient Data...")
    # FHE allows us to multiply encrypted data by plaintext weights (or encrypted weights)
    # Here we simulate the network running the model on the encrypted patient data.
    encrypted_prediction = encrypted_patient_data.dot(combined_theta.tolist())
    print(f"Encrypted Prediction Result: {encrypted_prediction}")
    
    # 4. Decrypt the Result
    print("\nDecrypting the Computed Prediction...")
    decrypted_prediction = encrypted_prediction.decrypt()[0]
    
    print(f"\n✅ FHE Decrypted Prediction: {decrypted_prediction}")
    print(f"✅ Final Classification: {'M' if round(decrypted_prediction) == 1 else 'B'}")
    
    print("\nComparison Check:")
    print(f"FHE Decrypted : {decrypted_prediction}")
    print(f"Plaintext     : {X_test_values.dot(combined_theta)}")


if __name__ == "__main__":
    main()
