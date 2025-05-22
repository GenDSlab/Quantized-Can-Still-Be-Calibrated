import json
from typing import Union, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from sklearn.metrics import auc
import torch
import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass
from numpy.typing import NDArray
import re
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


# Helper Functions
def validate_lists_length(
    cleaned_answers, cleaned_verbal_confidence_scores, cleaned_correctness_scores
):
    lists_length = {
        "cleaned_answers": len(cleaned_answers),
        "cleaned_verbal_confidence_scores": len(cleaned_verbal_confidence_scores),
        "cleaned_correctness_scores": len(cleaned_correctness_scores),
    }

    if len(set(lists_length.values())) != 1:
        unequal_lengths = [f"{name}: {length}" for name, length in lists_length.items()]
        raise ValueError(f"Lists have different lengths: {', '.join(unequal_lengths)}")


def remove_breacket(model_response):
    return re.sub(r"[\[\]]", "", model_response)


def normalize_confidence_score(confidence: str | float | None) -> float:
    """
    Normalizes a confidence score by extracting numeric values from strings (with or without brackets)
    and ensuring the result falls within the valid range [0, 1].

    Args:
        confidence: Input confidence score. Can be:
            - A string containing a number (e.g., "0.8" or "[0.8]")
            - A float value
            - None

    Returns:
        float: Normalized confidence score between 0 and 1. Returns 0.5 if:
            - Input cannot be converted to float
            - Input is None
            - Input format is invalid

    Examples:
        >>> normalize_confidence_score("[0.8]")
        0.8
        >>> normalize_confidence_score("0.8")
        0.8
        >>> normalize_confidence_score(1.5)
        1.0
        >>> normalize_confidence_score("invalid")
        0.5
    """
    if confidence is None:
        return 0.5

    try:
        if isinstance(confidence, str):
            # Try to extract number from brackets
            bracket_pattern = r"\[([\d.]+)\]"
            match = re.search(bracket_pattern, confidence)
            if match:
                float_confidence = float(match.group(1))
            else:
                # If no brackets found, try direct conversion
                confidence = re.sub(r"[\[\]]", "", confidence)
                float_confidence = float(confidence)
        else:
            float_confidence = float(confidence)

        # Validate confidence score range
        if float_confidence < 0:
            return 0
        elif float_confidence > 1:
            return 1
        return float_confidence

    except (ValueError, AttributeError):
        # If conversion fails, use default value
        return 0.5


def compute_correctness(
    confidence_scores1: List[float],
    confidence_scores2: List[float],
    correctness_scores: List[float],
    handle_multiple_max: bool = True,
) -> float:
    """
    Compute average correctness using numpy for better performance.
    """
    if not handle_multiple_max:
        max_index1 = np.argmax(confidence_scores1)
        max_index2 = np.argmax(confidence_scores2)
        return (correctness_scores[max_index1] + correctness_scores[max_index2]) / 2

    # Convert to numpy arrays for vectorized operations
    conf1 = np.array(confidence_scores1)
    conf2 = np.array(confidence_scores2)
    corr = np.array(correctness_scores)

    if len(conf1) != len(conf1):
        raise ("confidence scores have different length")

    if len(conf1) == 0:
        print("no response and confidence")
        return 0

    # Find all indices where value equals maximum
    max_mask1 = conf1 == np.max(conf1)
    max_mask2 = conf2 == np.max(conf2)

    # Get corresponding correctness scores
    selected_scores = np.concatenate([corr[max_mask1], corr[max_mask2]])
    # np.mean(selected_scores)
    return np.max(selected_scores)


def ubCE(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # expected prediction condition on correctnes
    corr_mask = targets == 1
    correct_ans_CE = 1 - np.mean(predictions[corr_mask])
    incorr_mask = targets == 0
    incorrect_ans_CE = np.mean(predictions[incorr_mask])

    reconstructed_ubCE = (
        np.mean(corr_mask) * correct_ans_CE + np.mean(incorr_mask) * incorrect_ans_CE
    )
    returned_ubCE = np.mean(predictions + targets - 2 * predictions * targets)

    return (
        returned_ubCE,
        correct_ans_CE,
        incorrect_ans_CE,
    )


# IntCE
@dataclass
class Interval:
    start: float
    end: float

    def __post_init__(self):
        if not (0 <= self.start <= 1 and 0 <= self.end <= 1):
            raise ValueError("Interval bounds must be in [0,1]")
        if self.start >= self.end:
            raise ValueError("Interval start must be less than end")


class IntervalPartition:
    def __init__(self, intervals: List[Interval]):
        """Initialize an interval partition of [0,1]."""
        self.intervals = sorted(intervals, key=lambda x: x.start)
        if not self.is_valid():
            raise ValueError(
                "Invalid partition: intervals must cover [0,1] without gaps or overlaps"
            )

    def is_valid(self) -> bool:
        """Validate that intervals form a proper partition of [0,1]."""
        if not self.intervals:
            return False

        # Check first and last intervals
        if not np.isclose(self.intervals[0].start, 0) or not np.isclose(
            self.intervals[-1].end, 1
        ):
            return False

        # Check for gaps and overlaps
        for i in range(len(self.intervals) - 1):
            if not np.isclose(self.intervals[i].end, self.intervals[i + 1].start):
                return False
        return True

    def width(self) -> float:
        """Calculate maximum bin width w(𝒯)."""
        return max(interval.end - interval.start for interval in self.intervals)


def compute_binned_ece(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    partition: IntervalPartition,
    eps: float = 1e-15,
) -> float:
    """
    Compute the binned Expected Calibration Error for a given partition.

    Args:
        predictions: Model predictions in [0,1]
        targets: Ground truth labels (0 or 1)
        partition: An interval partition of [0,1]
        eps: Small constant for numerical stability

    Returns:
        Normalized binned ECE value for the given partition
    """
    total_samples = len(predictions)
    if total_samples == 0:
        return 0.0

    binned_ece = 0.0

    for interval in partition.intervals:
        # Include right endpoint for last interval
        if np.isclose(interval.end, 1.0):
            mask = (predictions >= interval.start) & (predictions <= interval.end)
        else:
            mask = (predictions >= interval.start) & (predictions < interval.end)

        samples_in_bin = np.sum(mask)
        if samples_in_bin > 0:
            pred_in_bin = predictions[mask]
            targets_in_bin = targets[mask]

            # Compute normalized |E[(f-y)1(f ∈ Ij)]|
            calibration_error = np.abs(np.mean(pred_in_bin - targets_in_bin)) * (
                samples_in_bin / total_samples
            )
            binned_ece += calibration_error

    return binned_ece


def create_partition(bin_edges: NDArray[np.float64]) -> IntervalPartition:
    """Create a partition from bin edges."""
    intervals = [
        Interval(start, end) for start, end in zip(bin_edges[:-1], bin_edges[1:])
    ]
    return IntervalPartition(intervals)


def interval_calibration_error(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    min_bins: int = 2,
    max_bins: int = 50,
    eps: float = 1e-15,
) -> float:
    """
    Compute the Interval Calibration Error (intCE).

    Args:
        predictions: Model predictions in [0,1]
        targets: Ground truth labels (0 or 1)
        min_bins: Minimum number of bins to try
        max_bins: Maximum number of bins to try
        eps: Small constant for numerical stability

    Returns:
        The interval calibration error value
    """
    # Input validation
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    eps = 1e-10

    illegal_mask = (predictions < 0 - eps) | (predictions > 1 + eps)

    if not (predictions.shape == targets.shape and predictions.ndim == 1):
        print(f"{predictions.shape = }")
        print(f"{targets.shape = }")
        raise ValueError("predictions and targets must be 1D arrays of same shape")
    if len(predictions) == 0:
        raise ValueError("Empty input arrays")
    if np.any(illegal_mask):
        illegal_vals = predictions[illegal_mask]
        print(f"{illegal_vals = }")
        print(f"{predictions = }")
        print(f"{targets = }")
        raise ValueError("predictions must be in [0,1]")
    if not np.all(np.logical_or(np.isclose(targets, 0), np.isclose(targets, 1))):
        raise ValueError("targets must be binary (0 or 1)")

    # Clip predictions to [0,1] to handle numerical issues
    predictions = np.clip(predictions, 0, 1)

    min_error = float("inf")

    for num_bins in range(min_bins, max_bins + 1):
        bin_edges = np.linspace(0, 1, num_bins + 1)
        partition = create_partition(bin_edges)
        binned_ece = compute_binned_ece(predictions, targets, partition, eps)
        total_error = binned_ece + partition.width()
        min_error = min(min_error, total_error)

    return min_error


def brier_score(
    confidence_scores: List[float], correctness_scores: List[float]
) -> float:
    """
    Compute the Brier Score for a set of predictions.

    The Brier Score measures the mean squared difference between predicted
    probabilities (confidence scores) and the actual outcomes (correctness scores).
    A lower Brier Score indicates better calibration, with 0 being perfect.

    Args:
        confidence_scores (List[float]): A list of confidence scores (between 0 and 1).
        correctness_scores (List[float]): A list of correctness scores (0 or 1).

    Returns:
        float: The computed Brier Score.

    Raises:
        ValueError: If the input lists have different lengths or contain invalid values.
    """
    if len(confidence_scores) != len(correctness_scores):
        raise ValueError(
            "Confidence scores and correctness scores must have the same length."
        )

    illegal_vals = [score for score in confidence_scores if not (0 <= score <= 1)]

    # Raise error if there are illegal values and print them
    if illegal_vals:
        raise ValueError(
            f"Confidence scores must be between 0 and 1. Invalid values found: {illegal_vals}"
        )

    if not all(0 <= score <= 1 for score in correctness_scores):
        print(correctness_scores)
        raise ValueError("Correctness scores must be either 0 or 1.")

    confidence_scores = np.array(confidence_scores)
    correctness_scores = np.array(correctness_scores)

    # Compute Brier Score
    brier_score = np.mean((confidence_scores - correctness_scores) ** 2)

    return brier_score


# Extract correctness and confidence scores
from typing import Dict, List, Union, Any
from collections import defaultdict


def extract_model_metrics(
    data: Dict[str, Any],
) -> tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Extracts metrics from different model variants (full precision, BNB quantized, GPTQ quantized)
    from the input data dictionary.

    Args:
        data (Dict[str, Any]): Input dictionary containing model metrics and scores.
            Expected to have keys following the pattern:
            - *confidence_scores for full precision model
            - *bnb*confidence_scores for BNB quantized model
            - *GPTQ*confidence_scores for GPTQ quantized model
            Each model should have associated final_ans_correctness values.

    Returns:
        tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
            Three dictionaries containing metrics for full precision, BNB, and GPTQ models respectively.
            Each dictionary contains:
            - Confidence scores (CCTP, VCAA, GIAN, etc.)
            - final_ans_correctness
            - Question Correctness
    """
    # Initialize defaultdict for each model type to automatically create empty lists
    full_precision_scores = defaultdict(list)
    bnb_scores = defaultdict(list)
    gptq_scores = defaultdict(list)

    # Iterate through each question entry
    for question_id, question_data in data.items():

        # Process each model's metrics
        for key, value in question_data.items():
            # Handle confidence scores
            if "confidence_scores" in key:
                if "GPTQ" in key:
                    target_dict = gptq_scores
                elif "bnb" in key:
                    target_dict = bnb_scores
                else:
                    target_dict = full_precision_scores

                # Add all confidence scores
                for metric, score in value.items():
                    if isinstance(score, (list, tuple)):
                        # Extend list for metrics that return multiple values
                        target_dict[metric].extend(score)
                    else:
                        # Append single values
                        target_dict[metric].append(score)

            # Handle final answer correctness
            elif "all_model_responses_correctness" in key:
                gptq_scores["final_ans_correctness"].extend(value)
                bnb_scores["final_ans_correctness"].extend(value)
                full_precision_scores["final_ans_correctness"].extend(value)

    # Convert defaultdict to regular dict
    return (dict(full_precision_scores), dict(bnb_scores), dict(gptq_scores))


def print_model_metrics(model_name: str, metrics: Dict[str, List[float]]) -> None:
    """
    Helper function to print metrics for a given model in a formatted way.

    Args:
        model_name (str): Name of the model
        metrics (Dict[str, List[float]]): Dictionary containing model metrics
    """
    print(f"\n{model_name} Metrics:")
    for metric, values in metrics.items():
        # print(f"{metric}: {values}")
        print(metric, len(values))


def convert_to_binary(array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Converts a NumPy array to a binary array based on a threshold.

    Parameters:
        array (np.ndarray): The input array to be converted.
        threshold (float): The threshold value for binarization (default is 0.5).

    Returns:
        np.ndarray: The binary array where values >= threshold are 1 and < threshold are 0.
    """
    # Ensure the input is a NumPy array
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    # Convert to binary using vectorized operations
    binary_array = (array >= threshold).astype(int)

    return binary_array


def calculate_calibration_errors(
    model_metrics: Dict[str, Dict[str, List[float]]],
    interval_calibration_error: Callable[
        [NDArray[np.float64], NDArray[np.float64]], float
    ],
    brier_score: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    ubCE: Callable[
        [NDArray[np.float64], NDArray[np.float64]], Tuple[float, float, float]
    ],
) -> pd.DataFrame:
    """
    Calculate various calibration error metrics for different model confidence scores.

    Args:
        model_metrics (Dict[str, Dict[str, List[float]]]): Dictionary containing metrics for each model type.
            Expected structure: {model_name: {metric_name: metric_values}}
        interval_calibration_error (Callable): Function to calculate interval calibration error
        brier_score (Callable): Function to calculate Brier score
        ubCE (Callable): Function to calculate ubCE and its components

    Returns:
        pd.DataFrame: DataFrame containing calibration errors for each metric and model combination
        Columns: [Metric, Model Type, ICE, BS, ubCE, Correct CE, Incorrect CE]
    """
    # Initialize lists to store results
    results = []

    # Define metric groups by their correctness score type
    final_ans_metrics = ["CCTP", "VCAA"]
    question_correctness_metrics = ["GIET", "VCQO", "VCQMAC", "VRO", "GIAN"]
    #
    for model_name, metrics in model_metrics.items():
        # Process metrics using final_ans_correctness
        for metric in final_ans_metrics:
            if metric in metrics:
                confidence_scores = np.array(metrics[metric], dtype=np.float64)
                confidence_scores = np.nan_to_num(confidence_scores, nan=0.5)
                confidence_scores = np.clip(confidence_scores, 0, 1)
                correctness_scores = np.array(
                    metrics["final_ans_correctness"], dtype=np.float64
                )
                correctness_scores = convert_to_binary(correctness_scores)

                # Calculate all calibration metrics
                ice = interval_calibration_error(confidence_scores, correctness_scores)
                bs = brier_score(confidence_scores, correctness_scores)
                ubce, correct_ce, incorrect_ce = ubCE(
                    confidence_scores, correctness_scores
                )

                results.append(
                    {
                        "Metric": metric,
                        "Model Type": model_name,
                        "ICE": round(float(ice), 3),
                        "BS": round(float(bs), 3),
                        "ubCE": round(float(ubce), 3),
                        "Correct CE": round(float(correct_ce), 3),
                        "Incorrect CE": round(float(incorrect_ce), 3),
                    }
                )

        # Process metrics using Question Correctness
        for metric in question_correctness_metrics:
            if metric in metrics:
                confidence_scores = np.array(metrics[metric], dtype=np.float64)
                confidence_scores = np.nan_to_num(confidence_scores, nan=0.5)
                confidence_scores = np.clip(confidence_scores, 0, 1)
                correctness_scores = np.array(
                    metrics["Question Correctness"], dtype=np.float64
                )
                correctness_scores = convert_to_binary(correctness_scores)

                # Calculate all calibration metrics
                ice = interval_calibration_error(confidence_scores, correctness_scores)
                bs = brier_score(confidence_scores, correctness_scores)
                ubce, correct_ce, incorrect_ce = ubCE(
                    confidence_scores, correctness_scores
                )

                results.append(
                    {
                        "Metric": metric,
                        "Model Type": model_name,
                        "ICE": round(float(ice), 3),
                        "BS": round(float(bs), 3),
                        "ubCE": round(float(ubce), 3),
                        "Correct CE": round(float(correct_ce), 3),
                        "Incorrect CE": round(float(incorrect_ce), 3),
                    }
                )

    # Create DataFrame and sort by Metric and Model Type
    df = pd.DataFrame(results)
    df = df.sort_values(["Metric", "Model Type"])

    return df


def save_calibration_results(
    full_precision_metrics: Dict[str, List[float]],
    bnb_metrics: Dict[str, List[float]],
    gptq_metrics: Dict[str, List[float]],
    output_path: str,
    calibration_functions: Tuple[Callable, Callable, Callable],
) -> None:
    """
    Calculate and save calibration errors for all model types.

    Args:
        full_precision_metrics (Dict[str, List[float]]): Metrics for full precision model
        bnb_metrics (Dict[str, List[float]]): Metrics for BNB quantized model
        gptq_metrics (Dict[str, List[float]]): Metrics for GPTQ quantized model
        output_path (str): Path to save the output CSV file
        calibration_functions (Tuple[Callable, Callable, Callable]):
            Tuple of (interval_calibration_error, brier_score, ubCE) functions
    """
    # Prepare input for calibration calculation
    model_metrics = {
        "Full Precision": full_precision_metrics,
        "BNB": bnb_metrics,
        "GPTQ": gptq_metrics,
    }

    # Calculate calibration errors
    ice_func, bs_func, ubce_func = calibration_functions
    # print(f"{model_metrics=}")
    results_df = calculate_calibration_errors(
        model_metrics, ice_func, bs_func, ubce_func
    )

    # Save results to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Display results
    print("\nCalibration Error Results:")
    print(results_df.to_string())


from typing import Dict, List
import pandas as pd
import numpy as np


def calculate_calibration_differences(
    calibration_df: pd.DataFrame, output_path: str
) -> pd.DataFrame:
    """
    Calculate the difference between quantized models' calibration errors and
    the full precision model's calibration errors.

    Args:
        calibration_df (pd.DataFrame): DataFrame containing calibration errors for all models.
            Expected columns: [Metric, Model Type, ICE, BS, ubCE, Correct CE, Incorrect CE]
        output_path (str): Path to save the difference table CSV file

    Returns:
        pd.DataFrame: DataFrame containing the differences in calibration errors.
            Positive values indicate higher error in quantized model.
            Negative values indicate lower error in quantized model.
    """
    # List of metrics to calculate differences for
    error_columns = ["ICE", "BS", "ubCE", "Correct CE", "Incorrect CE"]

    # Initialize list to store results
    difference_results = []

    # Get unique metrics
    metrics = calibration_df["Metric"].unique()

    for metric in metrics:
        # Filter data for current metric
        metric_data = calibration_df[calibration_df["Metric"] == metric]

        # Get full precision model values
        full_precision = metric_data[
            metric_data["Model Type"] == "Full Precision"
        ].iloc[0]

        # Calculate differences for BNB and GPTQ
        for model in ["BNB", "GPTQ"]:
            quantized = metric_data[metric_data["Model Type"] == model].iloc[0]

            # Calculate differences for all error metrics
            differences = {
                col: round(float(quantized[col] - full_precision[col]), 3)
                for col in error_columns
            }

            # Add to results
            difference_results.append(
                {
                    "Metric": metric,
                    "Model Type": f"{model} vs Full Precision",
                    **differences,
                }
            )

    # Create DataFrame from results
    diff_df = pd.DataFrame(difference_results)

    # Sort by Metric and Model Type
    diff_df = diff_df.sort_values(["Metric", "Model Type"])

    # Save to CSV
    diff_df.to_csv(output_path, index=False)
    print(f"\nDifference results saved to {output_path}")

    # Add a summary of the differences
    print("\nSummary of Differences (Quantized - Full Precision):")
    print("Positive values indicate higher error in quantized model")
    print("Negative values indicate lower error in quantized model")
    print(diff_df.to_string())

    return diff_df


def analyze_calibration_differences(
    diff_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze the calibration differences to provide summary statistics.

    Args:
        diff_df (pd.DataFrame): DataFrame containing the calibration error differences

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing summary statistics for each model type
            Structure: {model_type: {metric: average_difference}}
    """
    summary_stats = {}
    error_columns = ["ICE", "BS", "ubCE", "Correct CE", "Incorrect CE"]

    for model in diff_df["Model Type"].unique():
        model_data = diff_df[diff_df["Model Type"] == model]

        # Calculate average absolute differences for each metric
        avg_diffs = {
            col: round(float(model_data[col].mean()), 3)
            # .abs()
            for col in error_columns
        }

        summary_stats[model] = avg_diffs

    return summary_stats


def compute_metric_calibration_errors(
    confidence_scores: np.ndarray, correctness_scores: np.ndarray
) -> Tuple[float, float, Tuple[float, float, float]]:
    """
    Compute different calibration errors for a single metric.

    Args:
        confidence_scores (np.ndarray): Array of confidence scores
        correctness_scores (np.ndarray): Array of correctness scores

    Returns:
        Tuple[float, float, Tuple[float, float, float]]:
            (interval_calibration_error, brier_score, (ubCE, correct_CE, incorrect_CE))
    """
    # Compute Interval Calibration Error
    ice = interval_calibration_error(confidence_scores, correctness_scores)

    # Compute Brier Score
    bs = brier_score(confidence_scores, correctness_scores)

    # Compute ubCE and its components
    ubce, correct_ce, incorrect_ce = ubCE(confidence_scores, correctness_scores)

    return ice, bs, (ubce, correct_ce, incorrect_ce)


def compare_model_calibration(
    full_precision_metrics: Dict[str, List[float]],
    quantized_metrics: Dict[str, List[float]],
    metric_name: str,
    model_name: str,
    alpha: float = 0.05,
) -> Dict[str, str]:
    """
    Compare calibration errors between quantized and full precision models for a single metric.

    Args:
        full_precision_metrics (Dict[str, List[float]]): Full precision model metrics
        quantized_metrics (Dict[str, List[float]]): Quantized model metrics
        metric_name (str): Name of the metric to compare
        model_name (str): Name of the quantized model
        alpha (float): Significance level

    Returns:
        Dict[str, str]: Results including differences and significance markers
    """
    # Get correctness scores based on metric type
    if metric_name in ["CCTP", "VCAA"]:
        correctness_key = "final_ans_correctness"
    else:
        correctness_key = "Question Correctness"

    # Get scores
    fp_confidence = np.array(full_precision_metrics[metric_name], dtype=np.float64)
    q_confidence = np.array(quantized_metrics[metric_name], dtype=np.float64)
    fp_confidence = np.nan_to_num(fp_confidence, nan=0.5)
    q_confidence = np.nan_to_num(q_confidence, nan=0.5)
    fp_confidence = np.clip(fp_confidence, 0, 1)
    q_confidence = np.clip(q_confidence, 0, 1)

    fp_correctness = np.array(full_precision_metrics[correctness_key], dtype=np.float64)
    fp_correctness = convert_to_binary(fp_correctness)
    q_correctness = np.array(quantized_metrics[correctness_key], dtype=np.float64)
    q_correctness = convert_to_binary(q_correctness)

    # Compute calibration errors for both models
    fp_ice, fp_bs, fp_ubce = compute_metric_calibration_errors(
        fp_confidence, fp_correctness
    )
    q_ice, q_bs, q_ubce = compute_metric_calibration_errors(q_confidence, q_correctness)

    # Compute differences
    differences = {
        "ICE": q_ice - fp_ice,
        "BS": q_bs - fp_bs,
        "ubCE": q_ubce[0] - fp_ubce[0],
        "Correct CE": q_ubce[1] - fp_ubce[1],
        "Incorrect CE": q_ubce[2] - fp_ubce[2],
    }

    # Perform Wilcoxon tests
    results = {}
    for col, diff in differences.items():
        # For ICE and BS, compare the individual error terms
        if col == "BS":
            q_errors = (q_confidence - q_correctness) ** 2
            fp_errors = (fp_confidence - fp_correctness) ** 2
        elif col == "ubCE":
            q_errors = q_confidence + q_correctness - 2 * q_confidence * q_correctness
            fp_errors = (
                fp_confidence + fp_correctness - 2 * fp_confidence * fp_correctness
            )
        else:
            results[col] = f"{diff:.3f}"
            continue

        # Perform Wilcoxon test
        try:
            _, pvalue = stats.wilcoxon(q_errors, fp_errors)
            # Format result with significance marker
            results[col] = f"{diff:.3f}{'*' if pvalue < alpha else ''}"
        except Exception as e:
            # Handle cases where test cannot be performed
            results[col] = f"{diff:.3f}"

    results["Metric"] = metric_name
    results["Model Type"] = f"{model_name} vs Full Precision"

    return results


def analyze_all_metrics_calibration(
    full_precision_metrics: Dict[str, List[float]],
    bnb_metrics: Dict[str, List[float]],
    gptq_metrics: Dict[str, List[float]],
    output_path: str,
) -> pd.DataFrame:
    """
    Analyze calibration differences for all metrics and models.

    Args:
        full_precision_metrics (Dict[str, List[float]]): Full precision model metrics
        bnb_metrics (Dict[str, List[float]]): BNB quantized model metrics
        gptq_metrics (Dict[str, List[float]]): GPTQ quantized model metrics
        output_path (str): Path to save results

    Returns:
        pd.DataFrame: Combined results for all metrics and models
    """
    results = []
    metrics = ["CCTP", "VCAA", "GIAN", "GIET", "VCQO", "VCQMAC", "VRO"]

    for metric in metrics:
        # Compare BNB vs Full Precision
        bnb_results = compare_model_calibration(
            full_precision_metrics, bnb_metrics, metric, "BNB"
        )
        results.append(bnb_results)

        # Compare GPTQ vs Full Precision
        gptq_results = compare_model_calibration(
            full_precision_metrics, gptq_metrics, metric, "GPTQ"
        )
        results.append(gptq_results)

    # Create DataFrame
    columns = [
        "Metric",
        "Model Type",
        "ICE",
        "BS",
        "ubCE",
        "Correct CE",
        "Incorrect CE",
    ]
    results_df = pd.DataFrame(results, columns=columns)

    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print("\nCalibration Error Differences (* indicates p < 0.05):")
    print(results_df.to_string())

    return results_df
