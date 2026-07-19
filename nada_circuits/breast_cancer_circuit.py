from nada_dsl import *


def nada_main():
    num_params = 31

    # Define the parties
    patient = Party(name="Party1")
    health_provider_1 = Party(name="Party2")
    health_provider_2 = Party(name="Party3")

    # Define the secret inputs
    hp_1_data_w = SecretInteger(Input(name="dataset1_w", party=health_provider_1))
    hp_2_data_w = SecretInteger(Input(name="dataset2_w", party=health_provider_2))
    patient_image_data = []  # Compute time secrets
    hp_1_theta = []  # Stored secrets
    hp_2_theta = []  # Stored secrets

    for i in range(num_params):
        patient_image_data.append(
            SecretInteger(Input(name=f"x_test{i}", party=patient))
        )
        hp_1_theta.append(
            SecretInteger(Input(name=f"hp1_p{i}", party=health_provider_1))
        )
        hp_2_theta.append(
            SecretInteger(Input(name=f"hp2_p{i}", party=health_provider_2))
        )

    # # Use weighted average to compute the combined theta
    combined_theta = []
    for hp_1_param, hp_2_param in zip(hp_1_theta, hp_2_theta):
        combined_theta.append(
            ((hp_1_param * hp_1_data_w) / Integer(100))
            + ((hp_2_param * hp_2_data_w) / Integer(100))
        )

    # Perform element-wise multiplication and summation for scaled data
    prediction = Integer(0)
    for x_val, theta_val in zip(patient_image_data, combined_theta):
        prediction += x_val * theta_val

    return [
        Output(prediction, "patient_test_prediction", patient),
    ]
