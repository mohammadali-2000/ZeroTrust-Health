import asyncio
import numpy as np
import pandas as pd
from utils.func import (
    calc_scaling_factor,
    compute_scaled_data,
    plot_distributions,
    compute_prediction,
)
import os
import py_nillion_client as nillion
import sys
from dotenv import load_dotenv
import pprint
import argparse
from config import (
    CONFIG_PROGRAM_NAME,
    CONFIG_TEST_PARTY_1,
    CONFIG_HP_PARTIES,
    CONFIG_NUM_PARAMS
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mpc_network_utils.nillion_client_helper import create_nillion_client
from mpc_network_utils.nillion_keypath_helper import getUserKeyFromFile, getNodeKeyFromFile

load_dotenv()


async def main():
    print("\n\n******* Healthcare Imaging Compute Program *******\n\n")
    
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

    # Print the rows of data
    print("\nFirst 50 rows of data' column:")
    print(health_data.head(50))

    # Remove ID column
    health_data = health_data.drop("ID", axis=1)

    # Split Dataset, 80 percent training, 20 percent testing
    train_size = int(0.8 * len(health_data))
    train_data = health_data[:train_size]
    test_data = health_data[train_size:]

    # Create a subset that is 25% of train_data
    subset_25_size = int(0.25 * len(train_data))
    subset_75_size = int(0.75 * len(train_data))
    subset_sm_data = train_data[:subset_25_size]

    # Create a subset that is 75% of train_data
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
    random_row = test_data.sample(n=1)
    # random_row = test_data.sample(n=1, random_state=20) # fixed random_state=20, selects 541st row with Diagnosis=0
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

    # Coefficients
    print("\n\nComputed theta values:")
    print("\nTheta (full train data):")
    print(theta)
    print("\nTheta (theta_subset_sm):")
    print(theta_subset_sm)
    print("\nTheta (theta_subset_lg):")
    print(theta_subset_lg)

    # Determine best scaling factor and compute scaled theta values
    precision = 10
    factor_theta = calc_scaling_factor(theta, precision)
    factor_theta_subset_sm = calc_scaling_factor(theta_subset_sm, precision)
    factor_theta_subset_lg = calc_scaling_factor(theta_subset_lg, precision)
    scaling_factor = max(factor_theta, factor_theta_subset_sm, factor_theta_subset_lg)

    scaled_theta = compute_scaled_data(theta, scaling_factor)
    scaled_theta_subset_sm = compute_scaled_data(theta_subset_sm, scaling_factor)
    scaled_theta_subset_lg = compute_scaled_data(theta_subset_lg, scaling_factor)

    ###### TEST ONLY - Print scaling factor and scaled theta values
    # print(f"\nScaling Factor needed for a precision value ({precision}):")
    # print(scaling_factor)
    # print("\nScaled Coefficients (theta):")
    # print(scaled_theta)
    # print("\nScaled Coefficients (scaled_theta_subset_sm):")
    # print(scaled_theta_subset_sm)
    # print("\nScaled Coefficients (scaled_theta_subset_lg):")
    # print(scaled_theta_subset_lg)
    ##### END TEST ONLY #####

    ##### TEST ONLY - Compute predictions for the test data using np
    # prediction = X_test_augmented @ theta
    # prediction_subset_sm = X_test_augmented @ theta_subset_sm
    # prediction_subset_lg = X_test_augmented @ theta_subset_lg
    # print("\nPrediction for the full test data using np:")
    # print(prediction)
    # print("\nPrediction for the test data using the small subset of train data:")
    # print(prediction_subset_sm)
    # print("\nPrediction for the test data using the large subset of train data:")
    # print(prediction_subset_lg)
    ##### END TEST ONLY #####

    # Compute prediction value using combined thetas from subsets
    # Calculate weights for each subset
    weights_subset_sm = subset_25_size / train_size
    weights_subset_lg = subset_75_size / train_size

    # Use weighted average to combine thetas
    combined_theta = (theta_subset_sm * weights_subset_sm) + (
        theta_subset_lg * weights_subset_lg
    )
    print("\nCombined theta via weighted average of train datasets:")
    print(combined_theta)
    
    # Predictions for the test data. Add a column of ones to X_test for the intercept term
    X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    print("\nX_test_augmented with intercept term added (Ones column)")
    print(X_test_augmented)

    ##### TEST ONLY - Compute predictions for the test data using np
    # prediction_subsets_combined = X_test_augmented @ combined_theta
    # print("\nY_Prediction value for the test data using the combined theta:")
    # print(prediction_subsets_combined)
    ##### END TEST ONLY #####

    # Setup final computation without use of np
    # Get the values from the augmented test data row
    X_test_values = X_test_augmented[0]

    # Scale the test data
    scaled_test_data = compute_scaled_data(X_test_values, scaling_factor)
    # print("\nScaled Test Data:")
    # print(scaled_test_data)

    # Perform element-wise multiplication and summation on full test data
    Y_prediction = compute_prediction(X_test_values, theta)

    print("\nY_Prediction value and classification on the full test data:")
    print(f"{Y_prediction}, classifying as {'M' if round(Y_prediction) == 1 else 'B'}")

    # Perform element-wise multiplication and summation on scaled data
    # Descale test and theta values
    descaled_test_data = [(value / scaling_factor) for value in scaled_test_data]
    descaled_theta = [(value / scaling_factor) for value in scaled_theta]

    Y_prediction_scaled = compute_prediction(descaled_test_data, descaled_theta)

    print(
        "\n**Scaled Values check** Y_Prediction value and classification on the full test data (w/scaled values), result expected to be identical:"
    )
    print(
        f"{Y_prediction_scaled}, classifying as {'M' if round(Y_prediction_scaled) == 1 else 'B'}"
    )

    # Perform element-wise multiplication and summation on subset_sm data
    Y_prediction_subset_sm = compute_prediction(X_test_values, theta_subset_sm)

    print("\nY_Prediction value and classification on the subset_sm test data:")
    print(
        f"{Y_prediction_subset_sm}, classifying as {'M' if round(Y_prediction_subset_sm) == 1 else 'B'}"
    )

    # Perform element-wise multiplication and summation on subset_lg data
    Y_prediction_subset_lg = compute_prediction(X_test_values, theta_subset_lg)

    print("\nY_Prediction value and classification on the subset_lg test data:")
    print(
        f"{Y_prediction_subset_lg}, classifying as {'M' if round(Y_prediction_subset_lg) == 1 else 'B'}"
    )
    # Perform element-wise multiplication and summation on combined subset data
    Y_prediction_combined_theta = compute_prediction(X_test_values, combined_theta)

    print("\nY_Prediction value and classification on the combined theta:")
    print(
        f"{Y_prediction_combined_theta}, classifying as {'M' if round(Y_prediction_combined_theta) == 1 else 'B'}"
    )
    
    #############################################
    import sys
    print("\n[NOTE] Stopping execution here! The Nillion SDK download server (nilup.nilogy.xyz) is currently offline, so we cannot run the cryptographic Nillion network part of this demo right now.")
    sys.exit(0)
    ############# Nillion section ###############
    #############################################

    print("\n*******************************")
    print("**** Nillion Blind Compute ****")
    print("*******************************\n")

    # Setup nillion
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    program_mir_path = f"../circuits-compiled/{CONFIG_PROGRAM_NAME}.nada.bin"

    num_params = CONFIG_NUM_PARAMS
    # num_params = train_data.columns.size

    ###### Setup test patient client ######
    print("\nSetting up test patient client...\n")
    client_test_patient = create_nillion_client(
        getUserKeyFromFile(CONFIG_TEST_PARTY_1["userkey_file"]),
        getNodeKeyFromFile(CONFIG_TEST_PARTY_1["nodekey_file"]),
    )
    
    
    party_id_test_patient = client_test_patient.party_id()
    user_id_test_patient = client_test_patient.user_id()
    print("Party ID Test Patient:", party_id_test_patient)
    print("User ID Test Patient:", user_id_test_patient)
    
    # Client test patient stores program
    print("\nClient Test Patient Storing program...\n")
    action_id = await client_test_patient.store_program(
        cluster_id, CONFIG_PROGRAM_NAME, program_mir_path
    )
    program_id = f"{user_id_test_patient}/{CONFIG_PROGRAM_NAME}"
    print("\nStored program. action_id:", action_id)
    print("\nStored program_id:", program_id)
    
    # Create secrets for test patient
    print("\nSetting up secrets for test patient...\n")
    party_test_patient_dict = {}
    
    for i in range(num_params):
        print(f"x_test{i}: {scaled_test_data[i]}")
        party_test_patient_dict[f"x_test{i}"] = nillion.SecretInteger(
            scaled_test_data[i]
        )
        
    print("\nParty Test Patient:")
    pprint.PrettyPrinter(indent=4).pprint(party_test_patient_dict)
    
    # Test Patient store secrets
    test_patient_secrets = nillion.Secrets(party_test_patient_dict)
    
    # Create test patient input bindings for program
    print("\nSetting up test patient input bindings...\n")
    program_bindings = nillion.ProgramBindings(program_id)
    program_bindings.add_input_party(
        CONFIG_TEST_PARTY_1["party_name"], party_id_test_patient
    )
    
    # Store secrets on the network
    print("\nStoring Test Patients secrets on the network...\n")
    store_id_test_patient = await client_test_patient.store_secrets(
        cluster_id, program_bindings, test_patient_secrets, None
    )
    print(f"\nSecrets for Test Patient: {test_patient_secrets} at program_id: {program_id}")
    print(f"\nStore_id: {store_id_test_patient}")
    
    ###### Setup Health Provider Party clients and store secrets ######
    
    store_ids = []
    party_ids = []
    
    for party_info in CONFIG_HP_PARTIES:
        print(f"\nSetting up {party_info['party_name']} client...\n")
        # Setup client
        client_n = create_nillion_client(
            getUserKeyFromFile(party_info["userkey_file"]),
            getNodeKeyFromFile(party_info["nodekey_file"]),
        )
        party_id_n = client_n.party_id()
        user_id_n = client_n.user_id()
        party_name = party_info["party_name"]
        print(f"Party ID for {party_info['party_name']}:", party_id_n)
        print(f"User ID for {party_info['party_name']}:", user_id_n)
    
        # Create secrets for Health Provider parties
        party_n_dict = {}
        dataset = []
        
        # Add dataset secret based on config
        if party_info["dataset"] == "scaled_theta_subset_sm":
            dataset = scaled_theta_subset_sm
            weight = 25
            party_n_dict["dataset1_w"] = nillion.SecretInteger(weight)
            
        elif party_info["dataset"] == "scaled_theta_subset_lg":
            dataset = scaled_theta_subset_lg
            weight = 75
            party_n_dict["dataset2_w"] = nillion.SecretInteger(weight)
        else:
            print("Error: Invalid dataset")
            return
        
        # Add secrets to party
        print(f"\nAdding {num_params} data points to {party_name}:")
        
        for i in range(num_params):
            party_n_dict[f"{party_info['secret_name']}{i}"] = nillion.SecretInteger(dataset[i])
            print(f"{party_info['secret_name']}{i}: {dataset[i]}")
            
        party_secret = nillion.Secrets(party_n_dict)
            
        # Create input bindings for the program
        print(f"\nSetting up input bindings for {party_name}...\n")
        secret_bindings = nillion.ProgramBindings(program_id)
        secret_bindings.add_input_party(party_name, party_id_n)
        
        # Create access_control object
        print(f"\nSetting up access_control for {party_name}...\n") 
        access_control = nillion.Permissions.default_for_user(user_id_n)
        
        # Give compute access_control to the first party
        compute_access_control = {
            user_id_test_patient: {program_id},
        }
        permissions.add_compute_permissions(compute_permissions)
        
        # Store the secret with permissions
        print(f"\nStoring secrets for {party_name} at program_id: {program_id}")
        store_id = await client_n.store_secrets(
            cluster_id, secret_bindings, party_secret, permissions
        )
        
        store_ids.append(store_id)
        party_ids.append(party_id_n)
        
        print(f"\nStore_id: {store_id}")
        print(f"Party ID: {party_id_n}")
        
    party_ids_to_store_ids = ' '.join([f'{party_id}:{store_id}' for party_id, store_id in zip(party_ids, store_ids)])
    
    print(f"\nParty IDs to Store IDs: {party_ids_to_store_ids}")
        
            
    ###### Setup Compute ######
       
    # Bind the parties in the computation to the client to set input and output parties
    print("\nSetting up compute bindings..\n")
    
    client_compute = create_nillion_client(
        getUserKeyFromFile(CONFIG_TEST_PARTY_1["userkey_file"]),
        getNodeKeyFromFile(CONFIG_TEST_PARTY_1["nodekey_alternate_file"]),
    )
    party_id_compute = client_compute.party_id()
    user_id_compute = client_compute.user_id()
    program_id_compute=f"{user_id_compute}/{CONFIG_PROGRAM_NAME}"
    
    print(f"\nComputing on program ID: {program_id_compute} with party ID: {party_id_compute}")
    print(f"\nUser ID: {user_id_compute}")
    
    print("\nAdding input parties to the computation...")
    print(f"{CONFIG_TEST_PARTY_1['party_name']}: {party_id_compute}")
    print(f"{CONFIG_HP_PARTIES[0]['party_name']}: {party_ids[0]}")
    print(f"{CONFIG_HP_PARTIES[1]['party_name']}: {party_ids[1]}")
    compute_bindings = nillion.ProgramBindings(program_id_compute)

    compute_bindings.add_input_party(CONFIG_TEST_PARTY_1["party_name"], party_id_compute)
    compute_bindings.add_output_party(CONFIG_TEST_PARTY_1["party_name"], party_id_compute)
    
    compute_bindings.add_input_party(CONFIG_HP_PARTIES[0]["party_name"], party_ids[0])
    compute_bindings.add_input_party(CONFIG_HP_PARTIES[1]["party_name"], party_ids[1])
      
    # # Setup public variables and compute time secrets
    # public_variables = {}
    computation_time_secrets = {}
    
    # Compute on the secret with all store ids
    print("\nComputing on the secret with all store ids...\n")
    print(f"Party Test Patient secret store_id: {store_id_test_patient}")
    print(f"Store IDs from HP Parties: {store_ids}")
    
    print("\nCombined Store IDs list:")
    print([store_id_test_patient] + store_ids)
    # print([store_id_test_patient] + store_ids)
    
    compute_id = await client_compute.compute(
        cluster_id,
        compute_bindings,
        [store_id_test_patient] + store_ids,
        nillion.Secrets(computation_time_secrets),
        nillion.PublicVariables({}),
    )
    print(f"\nThe computation was sent to the network - compute_id: {compute_id}")
    
    # Compute Results
    while True:
        compute_event_result = await client_compute.next_compute_event()
        if isinstance(compute_event_result, nillion.ComputeFinishedEvent):
            print(f"✅  Compute complete for compute_id {compute_event_result.uuid}")
            print(f"✅  The returned value1: {compute_event_result.result.value}")
            print(f"✅  Scaling Factor: {scaling_factor}")
            # Scaling factor is squared in the computation, thus we need to divide by returned value twice
            patient_prediction = (
                (compute_event_result.result.value["patient_test_prediction"] / scaling_factor / scaling_factor)
            )
            print(f"✅  The prediction is {patient_prediction}, classifying as {'M' if round(patient_prediction) == 1 else 'B'}")
            print("\n\n*** This is only a test performing blind computations and not a real prediction. ***\n\n")
            return compute_event_result.result.value


if __name__ == "__main__":
    asyncio.run(main())
