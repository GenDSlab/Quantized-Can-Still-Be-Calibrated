# Python Standard Library
import argparse
import gc
import logging
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from logging import Logger
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    Literal,  # Added missing `Literal` which appeared later
)

# Third-party Libraries
import numpy as np
from scipy import stats
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# from sentence_transformers import SentenceTransformer
from datasets import Dataset
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import list_repo_files
from torch.nn.attention import SDPBackend, sdpa_kernel

from peft import (
    PeftModel,
    PeftType,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_config,
    get_peft_model,
)

# Custom Modules
from cus_datasets_funcs import *
from cus_utils import *
from calibration_metrics import *


DEFAULT_SAVE_DIR = Path("calibrated_models")

logger = logging.getLogger(__name__)

FORCE_RERUN_OBTAIN_RESPONSE = False
FORCE_RERUN_OBTAIN_CS = False

# Basic setup for console logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

stage1_QA_format_instr = (
    "Please only provide a short and concise answer to the following question: Q:"
)

stage2_final_ans_format_instr = "For the provided question and response, extract a brief and precise final answer to the question. Enclose this answer in double brackets, like this: [[YOUR ANSWER]]. Include only the final answer within the brackets, without any additional text such as explanations or evaluations. \n"


VCS_itr_instr = """Review the provided answer against the question and indicate the correctness by outputting a binary value: 1 for a concise and correct answer or 0 for an incorrect answer. Please evaluate the response's accuracy based solely on whether it concisely and appropriately addresses the given question. Your response must consist solely of your confidence probability value enclosed in double brackets (e.g., [[0.75]]), without any additional text, explanations, spaces, or formatting."""


VUS_itr_instr = """Review the provided answer against the question and indicate the correctness by outputting a binary value: 1 for a concise and correct answer or 0 for an incorrect answer. Please evaluate the response's accuracy based solely on whether it concisely and appropriately addresses the given question. Your response must consist solely of your confidence probability value enclosed in double brackets (e.g., [[0.75]]), without any additional text, explanations, spaces, or formatting."""

correctness_prob_instr = """Review the provided answer against the question and indicate the correctness by outputting a binary value: 1 for a concise and correct answer or 0 for an incorrect answer. Please evaluate the response's accuracy based solely on whether it concisely and appropriately addresses the given question. Your response must consist solely of your confidence probability value without any additional text, explanations, spaces, or formatting."""

QA_single_ans_format_instr = (
    "Answer the following question with a short, concise response. \n Question: "
)


# Model
model = None
tokenizer = None
calibrated = False

logger = logging.getLogger(__name__)


from pathlib import Path
from typing import Optional, List, Tuple
import re


def get_model_save_path(
    base_model_name: str,
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Get the path of the calibrated model with the lowest eval loss.

    Args:
        base_model_name: Name or path of the base model (e.g. 'hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4')
        save_dir: Optional custom save directory
    Returns:
        Path object pointing to the model directory with lowest eval loss.
        If no evaluated models are found, returns the base calibrated name.
    """
    if save_dir is None:
        raise ValueError("save_dir must be provided")

    # Create the save directory if it doesn't exist
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Remove any organization prefix (e.g., 'hugging-quants/')
    model_base_name = base_model_name.split("/")[-1]
    # Clean the model name - replace special characters with underscores
    model_base_name = model_base_name.replace("-", "_").replace(".", "_")

    # Find the model with lowest eval loss
    pattern = f"calibrated_{model_base_name}_eval_loss_"
    existing_models = _find_existing_models(save_dir, pattern)

    if not existing_models:
        # If no evaluated models found, return base calibrated name
        print("No evaluated models found, returning base calibrated name")
        return save_dir / f"calibrated_{model_base_name}"

    # Return the path with lowest eval loss
    best_path, _ = min(existing_models, key=lambda x: x[1])
    print(f"Found best calibrated model: {best_path}")
    return best_path


def _find_existing_models(save_dir: Path, pattern: str) -> List[Tuple[Path, float]]:
    """
    Find all existing calibrated models and their eval losses in the save directory.

    Args:
        save_dir: Directory to search in
        pattern: Pattern to match for calibrated model names
    Returns:
        List of tuples containing (model_path, eval_loss)
    """
    if not save_dir.exists():
        return []

    models_with_loss = []
    for path in save_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(pattern):
            continue

        # Extract eval loss from the directory name
        match = re.search(r"eval_loss_([0-9.]+)$", path.name)
        if match:
            try:
                eval_loss = float(match.group(1))
                models_with_loss.append((path, eval_loss))
            except ValueError:
                continue

    return models_with_loss


def get_model_basename(model_path):
    """Find the first model file basename"""
    try:
        files = list_repo_files(repo_id=model_path)
        # Look for model files (both safetensors and bin)
        model_files = [f for f in files if f.endswith((".safetensors", ".bin"))]
        if model_files:
            # Get basename by removing extension
            return os.path.splitext(model_files[0])[0]
    except Exception as e:
        print(f"Warning: Could not list repository files: {e}")
    return None


@contextmanager
def initialize_llm(
    model_name: str,
    quantization: str = None,
    check_flash_attention: bool = True,
    device: str = "cuda",
    save_dir: Optional[Path] = None,
    verbose: bool = True,  # Changed to True for better debugging
) -> Generator[AutoModelForCausalLM, None, None]:
    """
    Initialize a HuggingFace language model with automatic resource management.
    First tries to load a calibrated model, then falls back to the original model if not found.
    """
    global calibrated  # Access the global variable
    calibrated = False  # Default to False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        logger.warning(
            "Running on CPU. This will be significantly slower than GPU inference."
        )

    local_model_path = get_model_save_path(model_name, save_dir)

    model_kwargs = {
        "use_cache": True,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        # "torch_dtype": torch.float16,
        "use_safetensors": True,
        "attn_implementation": "flash_attention_2",
        "low_cpu_mem_usage": True,
    }

    # First load the base model
    if "parasail-ai/Mistral-7B-Instruct-v0.3-GPTQ-4bit" in model_name:
        identified_basename = get_model_basename(model_name)
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            model_basename=identified_basename,
            device="cuda:0",
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            **model_kwargs,
        )

    # Check if calibrated model exists before trying to load it
    adapter_config_path = local_model_path / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            # Load prompt tuning adapter
            model = PeftModel.from_pretrained(model, local_model_path)
            if verbose:
                print(f"Successfully loaded adapter weights from {local_model_path}")

            # Set global calibrated flag to True
            calibrated = True

            # Load saved layer norm weights if they exist
            norm_weights_path = local_model_path / "norm_weights.pt"
            if norm_weights_path.exists():
                if verbose:
                    print(f"Found saved norm weights at {norm_weights_path}")
                norm_state = torch.load(norm_weights_path, map_location=device)

                # Load the norm weights into the base model
                base_model = model.get_base_model()
                missing_keys = []
                for name, saved_tensor in norm_state.items():
                    if name in dict(base_model.named_parameters()):
                        dict(base_model.named_parameters())[name].data.copy_(
                            saved_tensor
                        )
                    else:
                        missing_keys.append(name)

                if missing_keys:
                    print(
                        f"Warning: Could not find these norm parameters: {missing_keys}"
                    )
                else:
                    print("Successfully loaded all norm weights")
            else:
                print("No saved norm weights found, using initial values")
        except Exception as e:
            if verbose:
                print(f"Error loading weights from {local_model_path}: {str(e)}")
                print("Falling back to using the base model without calibration")

            # Ensure calibrated flag is False if we fall back
            calibrated = False
    else:
        if verbose:
            print(
                f"No calibrated model found at {local_model_path}. Using base model from {model_name}"
            )

    if check_flash_attention:
        if hasattr(model.config, "attn_implementation"):
            logger.info(f"Attention implementation: {model.config.attn_implementation}")
        else:
            logger.warning("Model does not expose attention implementation in config")

        # Additional check for actual flash attention usage
        try:
            from flash_attn.flash_attn_interface import flash_attn_func

            logger.info("Flash Attention is available")
        except ImportError:
            logger.warning("Flash Attention is not installed")

    try:
        model.eval()
        with torch.inference_mode():
            yield model
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def greedy_inference_response(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_input: str
) -> str:
    """
    Generate a deterministic response using HuggingFace model.

    Args:
        model (AutoModelForCausalLM): HuggingFace model instance
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        model_input (str): Input text for generation

    Returns:
        str: Generated model response text
    """
    formatted_question = tokenizer.apply_chat_template(
        [{"role": "user", "content": model_input}], tokenize=False
    )

    inputs = tokenizer(formatted_question, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
        "cuda"
    ):
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Greedy decoding
            use_cache=True,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return response.strip()


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    """Calculate token-wise log probabilities"""
    if not input_texts:
        raise ValueError("No input texts provided.")

    input_ids = tokenizer.apply_chat_template(
        input_texts, tokenize=True, return_tensors="pt"
    ).to("cuda")

    model.eval()
    with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # Shift logprobs and input_ids for next-token prediction
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]

    # Get probabilities for actual next tokens
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    # Convert to token and probability pairs
    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, token_log_prob in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append(
                    (tokenizer.decode(token), float(token_log_prob.item()))
                )
        batch.append(text_sequence)
    return batch


def find_first_difference(list1, list2):
    for i, (item1, item2) in enumerate(zip(list1, list2)):
        # Compare first elements of tuples
        if item1[0] != item2[0]:
            return i

    # If we've exhausted zip and lists have different lengths
    if len(list1) != len(list2):
        return min(len(list1), len(list2))

    # If no differences found
    return None


def response_token_logprobs(model, tokenizer, input: str, response: str):
    """Debug version with side-by-side token comparison"""
    input_texts = [
        {"role": "user", "content": input},
        {"role": "assistant", "content": response},
    ]

    # Get full sequence tokens and probs
    resp_token_probs = to_tokens_and_logprobs(model, tokenizer, input_texts)[0]

    # Get template tokens
    empty_texts = [
        {"role": "user", "content": input},
        {"role": "assistant", "content": ""},
    ]
    empty_template_token_probs = to_tokens_and_logprobs(model, tokenizer, empty_texts)[
        0
    ]

    response_start_idx = find_first_difference(
        resp_token_probs, empty_template_token_probs
    )

    result = resp_token_probs[response_start_idx:]

    return result


def response_logprob(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input: str,
    response: str,
) -> float:
    """
    Calculate the total log probability of a response given an input.

    Computes the sum of log probabilities across all tokens in the response.

    Args:
        llm (LLM): Initialized VLLM model instance
        tokenizer (AutoTokenizer): HuggingFace tokenizer matching the model
        input (str): Original input prompt/question
        response (str): Model-generated response text

    Returns:
        float: Sum of log probabilities across all response tokens

    Raises:
        ValueError: If response tokens cannot be found in formatted input

    Example:
        ```python
        total_logprob = response_logprob(
            llm,
            tokenizer,
            "What is ML?",
            "Machine Learning is..."
        )
        ```
    """

    answer_single_logprobs = response_token_logprobs(llm, tokenizer, input, response)

    if not answer_single_logprobs:
        len_res = len(response)
        strip_ans = response.strip()
        len_strip_ans = len(strip_ans)
        print(f"{len_res=}")
        print(f"{response=}")
        print(f"{len_strip_ans=}")
        print(f"{strip_ans=}")
        print(f"{answer_single_logprobs=}")
        raise ValueError("Could not compute log probabilities for response")

    # Just sum the logprobs directly from the tuples
    res_logprob = sum(logprob for _, logprob in answer_single_logprobs)
    return res_logprob


# Iterative Prompting
def stage1_QA_input_format(question, prev_ans, instruction):
    if len(prev_ans) == 0:
        model_input = instruction + question
    else:
        model_input = (
            "Consider the following question: Q: "
            + question
            + " Another answer to question Q is: "
            + ". Another answer to question Q is: ".join(prev_ans)
            + ". "
            + instruction
            + question
        )
    return model_input


def stage2_final_answer_input_format(instruction, stage1_ques, stage1_ans):
    model_input = (
        "Consider the following question and answer:"
        + "Q: "
        + stage1_ques
        + ". Answer: "
        + stage1_ans
        + ". "
        + instruction
    )
    return model_input


def flatten_string_list(mixed_list):
    """
    Flattens a list containing both strings and lists of strings into a single list of strings.

    Args:
        mixed_list (list): A list containing strings and/or lists of strings

    Returns:
        list: A flattened list containing only strings

    Examples:
        >>> flatten_string_list(['a', ['b', 'c'], 'd', ['e']])
        ['a', 'b', 'c', 'd', 'e']
    """
    flattened = []
    for item in mixed_list:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def extract_assistant_content(model_input, model_response):
    formatted_model_input = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": model_input},
            {"role": "assistant", "content": ""},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    special_tokens = tokenizer.all_special_tokens + ["<|end_header_id|>"]
    special_tokens = flatten_string_list(special_tokens)

    for token in special_tokens:
        if token:

            model_response = model_response.replace(token, "")
            formatted_model_input = formatted_model_input.replace(token, "")
    formatted_model_input = formatted_model_input.strip()
    model_response = model_response.strip()

    start_idx = 0
    for i, (c1, c2) in enumerate(zip(formatted_model_input, model_response)):
        if c1 != c2:
            start_idx = i
            break

    res_content = model_response[start_idx : len(model_response)]

    return res_content.strip()


def extract_bracket_content(res_content):
    pattern = r"\[{1,2}(.*?)\]{1,2}"

    matches = re.findall(pattern, res_content.replace(r"assistant\n\n", ""))

    return matches[-1].strip() if matches else res_content.strip()


def iter_ans_2_stage(ques, num_ans, stage1_instr, stage2_instr):
    final_answers = []
    draft_ans = []
    for _ in range(num_ans):
        # stage 2: finalized short answer
        training_single_itr_instr = f"Please only provide a single correct and concise answer to the following question. Enclose the final answer in your response in double brackets, like this: [[YOUR ANSWER]]. Question: {ques}"

        outputs = greedy_inference_response(model, tokenizer, training_single_itr_instr)
        model_ans_stage2 = extract_assistant_content(training_single_itr_instr, outputs)

        model_final_ans = extract_bracket_content(model_ans_stage2).strip()
        model_final_ans = remove_breacket(model_final_ans)

        model_ans_draft = model_final_ans

        final_answers.append(model_final_ans)
        draft_ans.append(model_ans_draft)

    return final_answers, draft_ans


# Confidence Scores
# Verb Conf Score
def VCS_input_format(question, answers, instr):
    model_input = (
        instr + "Question: " + question + " Answers: [" + ", ".join(answers) + "]. "
    )
    return model_input


def swap_with_first_new(lst, index):
    new_list = lst.copy()
    if 0 <= index < len(lst):
        # Create a new list
        new_list[0], new_list[index] = new_list[index], new_list[0]
    return new_list  # Return a copy even if no swap occurred


def verb_conf_scores(question, answers, instr):
    vcfs = []
    full_res = []
    for curr_index, curr_ans in enumerate(answers):
        temp_ans = swap_with_first_new(answers, curr_index)

        VCS_input = VCS_input_format(question, temp_ans, instr)

        VCS_output = greedy_inference_response(model, tokenizer, VCS_input)

        model_ans = extract_assistant_content(VCS_input, VCS_output)

        curr_vcs = extract_bracket_content(model_ans)

        vcfs.append(curr_vcs)
        full_res.append(model_ans)

    return vcfs, full_res


# Verb Uncertainty Score
def VUS_input_format(question, instr):
    model_input = instr + "Question: " + question

    return model_input


def verb_uncertainty_scores(question, instr):
    VUS_input = VUS_input_format(question, instr)

    VUS_output = greedy_inference_response(model, tokenizer, VUS_input)
    model_ans = extract_assistant_content(VUS_input, VUS_output)
    model_ans = extract_bracket_content(VUS_output)

    return model_ans


# Response probability calculation
def find_response_indices(
    formatted_input_tokens: List[int], response_tokens: List[int]
) -> Optional[Tuple[int, int]]:
    """
    Locate the response tokens within the formatted input token sequence.

    Uses exact matching to find the start and end indices of response tokens
    within the full formatted input sequence.

    Args:
        formatted_input_tokens (List[int]): Full sequence of input tokens
        response_tokens (List[int]): Sequence of response tokens to find

    Returns:
        Optional[Tuple[int, int]]: Tuple of (start_index, end_index) if found,
            None if response tokens cannot be located

    Example:
        ```python
        indices = find_response_indices(
            tokenizer.encode("Q: What? A: This."),
            tokenizer.encode("This.")
        )
        ```
    """
    n = len(formatted_input_tokens)
    m = len(response_tokens)

    for i in range(n - m + 1):
        if formatted_input_tokens[i : i + m] == response_tokens:
            return i, i + m

    return None


def debug_tokens(tokenizer, formatted_tokens, response_tokens) -> None:
    print("Formatted text tokens:")
    for i, t in enumerate(formatted_tokens):
        print(f"{i}: {t} -> {tokenizer.decode([t])}")
    print("\nResponse tokens:")
    for i, t in enumerate(response_tokens):
        print(f"{i}: {t} -> {tokenizer.decode([t])}")


def ans_prob_format(model_input, model_resp):
    messages = [
        {
            "role": "user",
            "content": model_input,
        },
        {
            "role": "assistant",
            "content": model_resp,
        },
    ]
    return messages


## Marginal prob
def log_marg_prob_answers(
    question: str,
    instr: str,
    answers: List[str],
) -> float:
    if not answers:
        raise ValueError("Answers list cannot be empty")

    ans_log_marg_prob = []
    for ans in answers:
        model_input = stage1_QA_input_format(question, prev_ans=[], instruction=instr)
        ans_log_marg_prob.append(response_logprob(model, tokenizer, model_input, ans))
    return ans_log_marg_prob


## Joint prob
def log_joint_prob_answers(
    question: str,
    instr: str,
    answers: List[str],
) -> float:
    """
    Calculate the log joint probability of a sequence of answers given a question and instruction.

    This function computes P(a₁, a₂, ..., aₙ | q, i) by calculating:
    log P(a₁|q,i) + log P(a₂|a₁,q,i) + ... + log P(aₙ|a₁,...,aₙ₋₁,q,i)

    Args:
        question: The input question text
        instruction: The instruction or prompt for generating answers
        answers: List of answer strings to compute joint probability for
        model: Pre-trained language model for computing response probabilities
        tokenizer: Associated tokenizer for the language model

    Returns:
        float: Log joint probability of the answer sequence

    Raises:
        ValueError: If answers list is empty
        TypeError: If inputs are not of expected types
    """
    if not answers:
        raise ValueError("Answers list cannot be empty")

    log_joint_prob: float = 0.0
    for curr_index, curr_ans in enumerate(answers):
        pa = answers[0:curr_index]
        model_input = stage1_QA_input_format(question, prev_ans=pa, instruction=instr)
        log_cond_prob_ans = response_logprob(model, tokenizer, model_input, curr_ans)
        log_joint_prob += log_cond_prob_ans
    return log_joint_prob


def iterative_subsets(arr: List[str]) -> List[Tuple[str, ...]]:
    """
    Generate iterative subsets of tuples from the input array of strings.

    This function creates subsets starting from the first two elements
    and progressively includes more elements until the full array is covered.

    Args:
        arr (List[str]): The input array of strings.

    Returns:
        List[Tuple[str, ...]]: A list of tuples, where each tuple is a subset
        of the input array, starting with at least two elements.

    Examples:
        >>> iterative_subsets(['A', 'B', 'C', 'D'])
        [('A', 'B'), ('A', 'B', 'C'), ('A', 'B', 'C', 'D')]
        >>> iterative_subsets(['X', 'Y'])
        [('X', 'Y')]
        >>> iterative_subsets(['Z'])
        []

    Note:
        - If the input array has fewer than 2 elements, an empty list is returned.
        - The function preserves the order of elements in the input array.
    """
    return [tuple(arr[:i]) for i in range(2, len(arr) + 1)] if len(arr) >= 2 else []


# Compute the probability of answers
def compute_ans_probs(
    full_answers: List[str],
    question: str,
    instr: str,
    logger: Optional[Logger] = None,
) -> Tuple[float, float]:
    """
    Compute joint and marginal log probabilities for a set of answers given a question and instruction.

    This function calculates two probability metrics:
    1. Joint log probability of all answers together
    2. Sum of marginal log probabilities for individual answers

    Args:
        full_answers: List of answer strings to evaluate
        question: The question text being answered
        instr: Instruction or context for answer evaluation
        logger: Optional logger instance for error handling

    Returns:
        Tuple containing:
            - log_joint_prob: Log probability of all answers occurring together
            - sum_log_marg_prob: Sum of individual answer log probabilities

    Raises:
        No exceptions are raised directly; errors are logged if logger is provided
    """
    log_joint_prob: float = 0.0
    sum_log_marg_prob: float = 0.0

    try:
        log_joint_prob = log_joint_prob_answers(question, instr, full_answers)
        sum_log_marg_prob = sum(log_marg_prob_answers(question, instr, full_answers))
    except Exception as e:
        if logger:
            logger.error(f"Error processing answers {full_answers}: {str(e)}")

    return log_joint_prob, sum_log_marg_prob


# Generation Independence metrics
def pointwise_mutual_information(
    log_joint_prob: float,
    log_marginal_prob: float,
) -> float:
    """
    Estimate pointwise mutual information from a single sample using the point-wise mutual information.

    For a single observation (x,y), this calculates the point-wise mutual information:
    i(x,y) = log(P(x,y) / (P(x)P(y)))

    When working with log probabilities, this becomes:
    i(x,y) = log_joint_prob - log_marginal_prob

    Note that this is a point estimate. The true mutual information would be the expected
    value of point-wise mutual information over all possible (x,y) pairs.

    Args:
        log_joint_prob: Log joint probability log P(x,y) for the observed sample
        log_marginal_prob: Log of the product of marginal probabilities log(P(x)P(y))

    Returns:
        float: The point-wise mutual information for this sample

    Example:
        >>> log_joint = -2.3
        >>> log_marginal = -3.2
        >>> pmi = estimate_mutual_information(log_joint, log_marginal)
    """
    # Calculate point-wise mutual information
    return log_joint_prob - log_marginal_prob


def calc_GIET(question, instr, unique_final_ans, final_answers):
    if len(unique_final_ans) == 0:
        return None
    log_joint_prob, sum_log_marg_prob = compute_ans_probs(
        final_answers, question, instr
    )
    mi = pointwise_mutual_information(log_joint_prob, sum_log_marg_prob)
    gen_indep_exp_trans = np.exp(-1.0 * np.abs(mi))

    return gen_indep_exp_trans


def calculate_npmi(log_joint_prob: float, log_marginal_prob: float) -> float:
    """
    Calculate Normalized Pointwise Mutual Information (NPMI).

    NPMI(x,y) = PMI(x,y) / -log(P(x,y))
                = (log(P(x,y)) - log(P(x)P(y))) / -log(P(x,y))

    Args:
        log_joint_prob: Log of joint probability log P(x,y)
        log_marginal_prob: Log of product of marginal probabilities log(P(x)P(y))

    Returns:
        float: NPMI value in range [-1, 1]

    Raises:
        ValueError: If joint probability is 0 (log_joint_prob is -inf)
    """
    if np.isinf(log_joint_prob):
        raise ValueError("Joint probability cannot be zero")

    pmi = log_joint_prob - log_marginal_prob
    npmi = pmi / -max(log_joint_prob, log_marginal_prob)

    return npmi


def calc_GIAN(question, instr, unique_final_ans, final_answers):
    if len(unique_final_ans) == 0:
        return None
    log_joint_prob, sum_log_marg_prob = compute_ans_probs(
        final_answers, question, instr
    )
    npmi = calculate_npmi(log_joint_prob, sum_log_marg_prob)
    gen_indep_abs_npmi = 1 - np.abs(npmi)

    return gen_indep_abs_npmi


# Correctness Classification Probability
def correctness_classification_confidence_score(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_question: str,
    response: str,
) -> float:
    correctness_classification_confidence = None
    model_input = (
        f"{correctness_prob_instr} Question: {input_question} Answer: {response}"
    )
    prob_correct = np.exp(response_logprob(llm, tokenizer, model_input, "1"))
    prob_incorrect = np.exp(response_logprob(llm, tokenizer, model_input, "0"))
    if prob_correct is None or prob_incorrect is None:
        print(f"{prob_correct=}")
        print(f"{prob_incorrect=}")
        raise ValueError("Probability of correctness cannot be obtained")
    correctness_classification_confidence = prob_correct / (
        prob_correct + prob_incorrect
    )
    return correctness_classification_confidence



# Collect model answers
def Obtain_model_answers(
    model_name: str,
    filepath: str,
    num_answers: int,
    stage1_QA_instr: str,
    stage2_QA_instr: str,
) -> Dict[str, Dict]:
    """
    Process model responses for questions and update the data dictionary with new responses.

    Args:
        model_name (str): Name of the model generating responses
        filepath (str): Path to the JSON file containing the question-answer dictionary

    Returns:
        Dict[str, Dict]: Updated dictionary containing original data and new model responses

    Raises:
        FileNotFoundError: If the specified filepath doesn't exist
        json.JSONDecodeError: If the file content is not valid JSON
        KeyError: If required keys are missing in the data structure
    """
    global calibrated  # Access the global variable

    # Determine prefix based on whether we're using a calibrated model
    model_prefix = "calibrated_model_" if calibrated else "model_"

    # Load the data dictionary
    data_dict = load_data_dict(filepath)

    # Create a deep copy to avoid modifying the original data while processing
    updated_dict = deepcopy(data_dict)

    save_every_k = 10
    count = 0
    # Process each sample in the dictionary
    for sample_id, sample_data in tqdm(updated_dict.items()):
        model_fina_ans_key = f"{model_prefix}{model_name}_final_ans"
        model_draft_key = f"{model_prefix}{model_name}_draft_ans"

        if model_fina_ans_key in sample_data:
            model_ans_correctness = obtain_correctness(
                filepath, sample_data["gt_ans"], sample_data["all_model_responses"]
            )
            updated_dict[sample_id][
                "all_model_responses_correctness"
            ] = model_ans_correctness

            if not FORCE_RERUN_OBTAIN_RESPONSE:
                continue

        # Generate model responses
        question = sample_data["question"]
        model_ans, draft_ans = iter_ans_2_stage(
            question, num_answers, stage1_QA_instr, stage2_QA_instr
        )
        # print(f"{model_ans = }")

        # Update or create "all_model_responses"
        if "all_model_responses" not in sample_data:
            sample_data["all_model_responses"] = list(set(model_ans))
        else:
            # Extend existing responses and deduplicate
            sample_data["all_model_responses"] = list(
                set(sample_data["all_model_responses"] + model_ans)
            )

        # Add model-specific responses
        updated_dict[sample_id][model_fina_ans_key] = model_ans
        updated_dict[sample_id][model_draft_key] = draft_ans

        model_ans_correctness = obtain_correctness(
            filepath, sample_data["gt_ans"], sample_data["all_model_responses"]
        )
        updated_dict[sample_id][
            "all_model_responses_correctness"
        ] = model_ans_correctness

        count += 1
        if count % save_every_k == 0:
            save_data_dict(updated_dict, filepath)

    save_data_dict(updated_dict, filepath)
    return updated_dict


def Obtain_confidence_scores(
    model_name: str,
    filepath: str,
):
    """
    Computes and stores confidence scores for model responses
    """
    global calibrated  # Access the global variable

    # Determine prefix based on whether we're using a calibrated model
    model_prefix = "calibrated_model_" if calibrated else "model_"

    # Load the data dictionary
    data_dict = load_data_dict(filepath)

    # Create a deep copy to avoid modifying the original data while processing
    updated_dict = deepcopy(data_dict)

    save_every_k = 10
    count = 0
    # Process each sample in the dictionary
    for sample_id, sample_data in tqdm(updated_dict.items()):
        model_cs_key = f"{model_prefix}{model_name}_confidence_scores"
        cs_dict = {}

        if model_cs_key in sample_data:
            if len(sample_data[model_cs_key]["CCTP"]) == len(
                sample_data["all_model_responses_correctness"]
            ):

                if not FORCE_RERUN_OBTAIN_CS:
                    continue

        # Collect required info for computation
        question = sample_data["question"]
        all_model_resp = sample_data["all_model_responses"]

        # Answer Level Confidence Scores
        # CCTP
        cs_dict["CCTP"] = [
            correctness_classification_confidence_score(
                model, tokenizer, question, curr_ans
            )
            for curr_ans in all_model_resp
        ]

        # VCAA
        vcaa, vcaa_full_res = verb_conf_scores(question, all_model_resp, VCS_itr_instr)
        vcaa = [normalize_confidence_score(vcs) for vcs in vcaa]
        cs_dict["VCAA"] = vcaa

        # Model answer correctness
        cs_dict["Question Correctness"] = compute_correctness(
            vcaa, cs_dict["CCTP"], sample_data["all_model_responses_correctness"]
        )

        # Update model-specific confidence scores dict
        updated_dict[sample_id][model_cs_key] = cs_dict

        count += 1
        if count % save_every_k == 0:
            save_data_dict(updated_dict, filepath)

    save_data_dict(updated_dict, filepath)
    return updated_dict


def main(
    llm_model_name: str,
    data_file_path: str,
    num_responses: int,
    action: str,
    save_dir: Path,  # New parameter
):
    short_model_name = llm_model_name.rsplit("/", 1)[-1]

    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        trust_remote_code=True,
    )
    quant_method = None
    if "gptq".lower() in short_model_name.lower():
        quant_method = "gptq"
    elif "bnb".lower() in short_model_name.lower():
        quant_method = "bitsandbytes"
    elif "awq".lower() in short_model_name.lower():
        quant_method = "awq"

    with initialize_llm(
        llm_model_name, quantization=quant_method, save_dir=save_dir
    ) as model:
        logging.info(
            f"LLM initialized. Starting inference with model: {llm_model_name}, quantization: {quant_method}"
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        if action == "obtain_responses":
            Obtain_model_answers(
                short_model_name,
                data_file_path,
                num_responses,
                stage1_QA_format_instr,
                stage2_final_ans_format_instr,
            )
        elif action == "obtain_confidence_scores":
            Obtain_confidence_scores(short_model_name, data_file_path)
            # 1. Delete the model explicitly
            del model
            if "model" in globals():
                del globals()["model"]

            # 2. Clear CUDA cache if you're using PyTorch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 3. Force garbage collection to clean up any remaining references
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("llm_model", type=str, help="Name of the LLM model to use")
    parser.add_argument(
        "data_file_path", type=str, help="File path to load and save the data"
    )
    parser.add_argument(
        "num_responses", type=int, help="Number of responses to generate"
    )
    parser.add_argument(
        "action",
        type=str,
        help="Action to be taken (obtain_responses or obtain_confidence_scores)",
    )
    # Add new argument for save_dir
    parser.add_argument(
        "--save_dir",
        type=str,
        default="calibrated_models",
        help="Directory path to save calibrated models",
    )

    args = parser.parse_args()
    avail_actions = [
        "obtain_responses",
        "obtain_confidence_scores",
    ]

    if args.action not in avail_actions:
        raise ValueError(f"Action must be in {avail_actions}")

    main(
        args.llm_model,
        args.data_file_path,
        args.num_responses,
        args.action,
        Path(args.save_dir),  # Convert string to Path object
    )
