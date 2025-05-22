# compute_calibration_error
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from calibration_metrics import convert_to_binary


def normalize_confidence_score(confidence):
    """
    Normalizes a confidence score by ensuring it falls within the valid range [0, 1].
    """
    if confidence is None:
        return 0.5

    try:
        float_confidence = float(confidence)
        if float_confidence < 0:
            return 0
        elif float_confidence > 1:
            return 1
        return float_confidence
    except (ValueError, TypeError):
        return 0.5


def extract_model_metrics(data):
    """
    Extract confidence scores and correctness from different models in the data.

    Returns:
        dict: Dictionary of model_name -> {metric_name -> values}
    """
    model_metrics = {}

    for question_id, question_data in data.items():
        # Skip if no correctness information
        if "all_model_responses_correctness" not in question_data:
            continue

        correctness_scores = question_data["all_model_responses_correctness"]

        # Extract metrics for each model
        for key in question_data:
            if key.endswith("_confidence_scores"):
                model_name = key.replace("_confidence_scores", "")

                # Initialize model dict if not exists
                if model_name not in model_metrics:
                    model_metrics[model_name] = {
                        "CCTP": [],
                        "VCAA": [],
                        "correctness": [],
                    }

                # Add confidence scores
                conf_scores = question_data[key]
                if "CCTP" in conf_scores:
                    model_metrics[model_name]["CCTP"].extend(conf_scores["CCTP"])
                    model_metrics[model_name]["correctness"].extend(correctness_scores)

                if "VCAA" in conf_scores:
                    model_metrics[model_name]["VCAA"].extend(conf_scores["VCAA"])

    return model_metrics


def brier_score(confidence_scores, correctness_scores):
    """
    Compute the Brier Score for a set of predictions.
    """
    if len(confidence_scores) != len(correctness_scores):
        raise ValueError("Confidence and correctness scores must have the same length")

    confidence_scores = np.array(confidence_scores)
    correctness_scores = np.array(correctness_scores)

    # Clip confidence scores to [0, 1]
    confidence_scores = np.clip(confidence_scores, 0, 1)

    # Compute Brier Score
    return np.mean((confidence_scores - correctness_scores) ** 2)


def ubCE(predictions, targets):
    """
    Calculate the Uncertainty-Balanced Calibration Error (ubCE).
    """
    predictions = np.array(predictions, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)

    # Clip predictions to [0, 1]
    predictions = np.clip(predictions, 0, 1)

    # Calculate components based on correctness
    corr_mask = targets == 1
    incorr_mask = targets == 0

    correct_ans_CE = 1 - np.mean(predictions[corr_mask]) if np.any(corr_mask) else 0
    incorrect_ans_CE = np.mean(predictions[incorr_mask]) if np.any(incorr_mask) else 0

    # Calculate overall ubCE
    returned_ubCE = np.mean(predictions + targets - 2 * predictions * targets)

    return returned_ubCE, correct_ans_CE, incorrect_ans_CE


def calculate_all_calibration_errors(model_metrics):
    """
    Calculate calibration errors for all models and metrics.

    Returns:
        list: List of dictionaries containing calibration metrics
    """
    results = []

    for model_name, metrics in model_metrics.items():
        # Process CCTP
        if len(metrics["CCTP"]) > 0 and len(metrics["correctness"]) > 0:
            conf_scores = np.array(metrics["CCTP"])

            # Convert to binary
            conf_scores = convert_to_binary(conf_scores)

            corr_scores = np.array(metrics["correctness"])

            # Calculate metrics
            ubce_val, correct_ce, incorrect_ce = ubCE(conf_scores, corr_scores)
            bs_val = brier_score(conf_scores, corr_scores)

            results.append(
                {
                    "Model": model_name,
                    "Metric": "CCTP",
                    "ubCE": round(float(ubce_val), 3),
                    "Correct CE": round(float(correct_ce), 3),
                    "Incorrect CE": round(float(incorrect_ce), 3),
                    "BS": round(float(bs_val), 3),
                    "Count": len(conf_scores),
                }
            )

        # Process VCAA
        if len(metrics["VCAA"]) > 0 and len(metrics["correctness"]) > 0:
            conf_scores = np.array(metrics["VCAA"])
            corr_scores = np.array(metrics["correctness"])

            # Calculate metrics
            ubce_val, correct_ce, incorrect_ce = ubCE(conf_scores, corr_scores)
            bs_val = brier_score(conf_scores, corr_scores)

            results.append(
                {
                    "Model": model_name,
                    "Metric": "VCAA",
                    "ubCE": round(float(ubce_val), 3),
                    "Correct CE": round(float(correct_ce), 3),
                    "Incorrect CE": round(float(incorrect_ce), 3),
                    "BS": round(float(bs_val), 3),
                    "Count": len(conf_scores),
                }
            )

    return results


def main(filepath):
    """
    Main function to compute calibration errors for a dataset.

    Args:
        filepath (str): Path to the JSON file containing the data
    """
    # Load data
    print(f"Loading data from {filepath}...")
    with open(filepath, "r") as f:
        data = json.load(f)

    # Extract metrics
    print("Extracting model metrics...")
    model_metrics = extract_model_metrics(data)

    # Calculate calibration errors
    print("Calculating calibration errors...")
    results = calculate_all_calibration_errors(model_metrics)

    # Create DataFrame and save results
    results_df = pd.DataFrame(results)

    # Sort by Model and Metric
    results_df = results_df.sort_values(["Model", "Metric"])

    # Define output path - same filename with csv extension
    output_path = Path(filepath).with_suffix(".csv")

    # Save results
    print(f"Saving results to {output_path}...")
    results_df.to_csv(output_path, index=False)

    # Display results
    print("\nCalibration Error Results:")
    print(results_df.to_string())

    return results_df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_calibration.py <data_file_path>")
        sys.exit(1)

    filepath = sys.argv[1]
    main(filepath)
