# code for datasets
# Python standard library
import json
import re
import string
from collections import Counter
from typing import Dict, List, Any, TypedDict
from pathlib import Path
from datasets import Dataset, load_dataset


def initialize_data_dict(
    process_fn: callable,
    dataset_name: str,
    config_name: str | None = None,
    split: str = "validation",
    **dataset_kwargs: Any,
) -> Dict[str, Dict]:
    """
    Initialize a dictionary from any dataset using specified parameters and processing function.

    Args:
        dataset_name (str): Name of the dataset to load (e.g., "trivia_qa")
        config_name (str | None, optional): Configuration name for the dataset.
            Defaults to None.
        split (str, optional): Dataset split to use. Defaults to "validation".
        process_fn (callable, optional): Function to process each sample.
            Defaults to process_triviaqa_sample.
            Must take a dictionary as input and return Dict[str, AnswerDict].
        **dataset_kwargs: Additional keyword arguments to pass to load_dataset

    Returns:
        Dict[str, AnswerDict]: Dictionary where:
            - key: question_id (str)
            - value: AnswerDict containing:
                - question: The question text
                - gt_ans: List of ground truth answers

    Examples:
        >>> # Initialize TriviaQA dataset
        >>> qa_dict = initialize_qa_dict(
        ...     "trivia_qa",
        ...     "rc.nocontext",
        ...     split="validation"
        ... )

        >>> # Initialize with custom processing function
        >>> def custom_processor(sample):
        ...     return {sample['id']: {
        ...         'question': sample['question'],
        ...         'gt_ans': [sample['answer']]
        ...     }}
        >>> custom_dict = initialize_qa_dict(
        ...     process_fn=custom_processor
        ...     "squad",
        ...     split="train",
        ... )

    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If there's an error loading or processing the dataset
    """
    try:
        # Load the dataset
        dds = load_dataset(dataset_name, config_name, split=split, **dataset_kwargs)
        dataset = dds.shuffle(seed=567)
        dict_len = len(dataset)
        print(f"{ dict_len = }")

        if not isinstance(dataset, Dataset):
            raise RuntimeError("Failed to load dataset properly")

        # Process all samples
        qa_dict: Dict[str, Dict] = {}
        process_count = 0
        for sample in dataset:
            try:
                processed = process_fn(sample)
                if not isinstance(processed, dict):
                    raise TypeError(
                        f"Process function must return dict, got {type(processed)}"
                    )
                qa_dict.update(processed)
                process_count += 1
            except Exception as e:
                raise RuntimeError(
                    f"Error processing sample: {str(e)}\nSample: {sample}"
                )

        return qa_dict

    except Exception as e:
        raise RuntimeError(f"Error initializing QA dictionary: {str(e)}")


def save_data_dict(qa_dict: Dict[str, Dict], filepath: str) -> None:
    """
    Save the processed QA dictionary to a JSON file.

    Args:
        qa_dict (Dict[str, AnswerDict]): The processed QA dictionary to save
        filepath (str): Path where the JSON file should be saved

    Raises:
        ValueError: If the filepath is invalid
        TypeError: If qa_dict is not a dictionary
        IOError: If there's an error writing to the file

    Examples:
        >>> qa_dict = initialize_qa_dict()
        >>> save_qa_dict(qa_dict, "processed_qa_validation.json")
    """
    if not isinstance(qa_dict, dict):
        raise TypeError("qa_dict must be a dictionary")

    # Convert to Path object for better path handling
    path = Path(filepath)

    # Ensure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(qa_dict, f, ensure_ascii=False, indent=2)
    except IOError as e:
        raise IOError(f"Error saving QA dictionary: {str(e)}")


def load_data_dict(filepath: str) -> Dict[str, Dict]:
    """
    Load a processed QA dictionary from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing the saved QA dictionary

    Returns:
        Dict[str, AnswerDict]: The loaded QA dictionary

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the file content is not valid JSON
        TypeError: If the loaded data doesn't match the expected format

    Examples:
        >>> # Save and then load the QA dictionary
        >>> qa_dict = initialize_qa_dict()
        >>> save_qa_dict(qa_dict, "processed_qa.json")
        >>> loaded_qa_dict = load_qa_dict("processed_qa.json")
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded_dict = json.load(f)

        # Validate the loaded dictionary structure
        for qid, qa_item in loaded_dict.items():
            if not isinstance(qid, str):
                raise TypeError(f"Question ID must be string, got {type(qid)}")
            if not isinstance(qa_item, dict):
                raise TypeError(f"QA item must be dictionary, got {type(qa_item)}")
            if "question" not in qa_item or "gt_ans" not in qa_item:
                raise ValueError(
                    f"Missing required keys in QA item for question ID {qid}"
                )
            if not isinstance(qa_item["gt_ans"], list):
                raise TypeError(
                    f"Ground truth answers must be a list for question ID {qid}"
                )

        return loaded_dict

    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading QA dictionary: {str(e)}")


# TriviaQA funcs
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def tqa_correctness_eval(
    ground_truths, predicted_answers, qid_list=None, mute=True
) -> list[float]:
    correctness_val = []
    for prediction in predicted_answers:
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths
        )
        correctness_val.append(max(em_for_this_question, f1_for_this_question) * 1.0)

    return correctness_val


def process_arc_sample(sample: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Process a sample from the ARC Challenge dataset and return a structured dictionary
    containing the question ID, question text, correct answer, and incorrect answers.
    Args:
        sample (Dict[str, Any]): A sample from the ARC dataset. Expected to have
            the following structure:
            {
                "id": str,
                "question": str,
                "answerKey": str,
                "choices": {
                    "label": List[str],
                    "text": List[str]
                }
            }
    Returns:
        Dict[str, AnswerDict]: A dictionary where:
            - key: question_id (str)
            - value: AnswerDict containing:
                - question: The question text
                - gt_ans: List containing only the correct answer text
                - incorrect_ans: List containing all incorrect answer texts
    Examples:
        >>> sample = {
        ...     "id": "Mercury_SC_405487",
        ...     "question": "One year, the oak trees in a park began producing more acorns than usual...",
        ...     "answerKey": "B",
        ...     "choices": {
        ...         "label": ["A", "B", "C", "D"],
        ...         "text": ["Shady areas increased.", "Food sources increased.",
        ...                  "Oxygen levels increased.", "Available water increased."]
        ...     }
        ... }
        >>> result = process_arc_sample(sample)
        >>> print(result)
        {
            'Mercury_SC_405487': {
                'question': 'One year, the oak trees in a park began producing more acorns than usual...',
                'gt_ans': ['Food sources increased.'],
                'incorrect_ans': ['Shady areas increased.', 'Oxygen levels increased.', 'Available water increased.']
            }
        }
    Raises:
        KeyError: If required fields are missing from the input sample
        ValueError: If answerKey doesn't match any choice label
        IndexError: If choices arrays don't align
    """
    if not isinstance(sample, dict):
        raise TypeError("Input sample must be a dictionary")
    try:
        # Extract required fields
        question_id = sample["id"]
        question_text = sample["question"]
        answer_key = sample["answerKey"]
        choice_labels = sample["choices"]["label"]
        choice_texts = sample["choices"]["text"]
        # Validate choice arrays
        if len(choice_labels) != len(choice_texts):
            raise ValueError("Choice labels and texts arrays must have same length")
        # Find the index of the correct answer
        try:
            answer_idx = choice_labels.index(answer_key)
        except ValueError:
            raise ValueError(f"Answer key '{answer_key}' not found in choice labels")
        # Get the correct answer text
        correct_answer = choice_texts[answer_idx]
        # Get incorrect answer texts (all choices except the correct one)
        incorrect_answers = [
            choice_texts[i] for i in range(len(choice_texts)) if i != answer_idx
        ]

        full_question = question_text + "\n Choices: " + str(choice_texts) + "."
        return {
            question_id: {
                "question": full_question,
                "gt_ans": [correct_answer],
                "incorrect_ans": incorrect_answers,
            }
        }
    except KeyError as e:
        raise KeyError(f"Missing required field in sample: {str(e)}")
    except IndexError as e:
        raise IndexError(f"Error accessing choices array: {str(e)}")


def obtain_correctness(data_file_path, gt_ans, model_ans):
    return tqa_correctness_eval(gt_ans, model_ans)
