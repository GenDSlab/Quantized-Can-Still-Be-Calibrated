import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import list_repo_files
from auto_gptq import AutoGPTQForCausalLM


from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)

from peft import (
    PeftModel,
    PeftType,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_config,
    get_peft_model,
)
from accelerate import Accelerator


DEFAULT_SAVE_DIR = Path("calibrated_models")


class CalibrationDataset(Dataset):
    """Dataset class for calibration training."""

    def __init__(
        self,
        data_dict: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        # instruction: str,
        eval: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_dict: Dictionary containing the training data
            tokenizer: Tokenizer for processing text
            instruction: Instruction prompt to prepend to inputs
        """
        self.data = data_dict
        self.tokenizer = tokenizer
        self.instruction_simple = """Review the provided answer against the question and indicate the correctness by outputting a binary value: 1 for a concise and correct answer or 0 for an incorrect answer. Please evaluate the response's accuracy based solely on whether it concisely and appropriately addresses the given question. Your response must consist solely of your confidence probability value without any additional text, explanations, spaces, or formatting."""
        self.instruction_double_bracket = """Review the provided answer against the question and indicate the correctness by outputting a binary value: 1 for a concise and correct answer or 0 for an incorrect answer. Please evaluate the response's accuracy based solely on whether it concisely and appropriately addresses the given question. Your response must consist solely of your confidence probability value enclosed in double brackets (e.g., [[0.75]]), without any additional text, explanations, spaces, or formatting."""
        self.sample_ids = list(data_dict.keys())
        self.eval = eval

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        Args:
            idx: Index of the example to fetch
        Returns:
            Dictionary containing input_ids and labels
        """
        sample_id = self.sample_ids[idx]
        sample = self.data[sample_id]

        sft_rand = random.random()
        cs_model_ans_rand = random.random()
        case_rand = random.random()
        format_rand = random.random()

        if eval:
            sft_rand = 0
            cs_model_ans_rand = 0
            case_rand = 0

        # Determine model response accuracy
        model_ans = sample["all_model_responses"]
        model_ans_correctness = sample["all_model_responses_correctness"]
        model_ans_correct_bin = [1 if num > 0.5 else 0 for num in model_ans_correctness]
        model_acc = sum(model_ans_correct_bin) / len(model_ans_correct_bin)

        if format_rand > 0.5:
            final_instr = self.instruction_double_bracket
        else:
            final_instr = self.instruction_simple

        if sft_rand > 0.8:
            input_text = f"Please only provide a single correct and concise answer to the following question. Enclose the final answer in your response in double brackets, like this: [[YOUR ANSWER]]. Question: {sample['question']}"
            target = "[[" + str(random.sample(sample["gt_ans"], 1)[0]) + "]]"
        else:
            if cs_model_ans_rand > 0.5:
                correctness_sample = random.random()
                if correctness_sample > model_acc:
                    response_idx = random.randint(0, len(sample["gt_ans"]) - 1)
                    input_text = f"{final_instr} Question: {sample['question']} Answer: {sample['gt_ans'][response_idx]}"
                    target = str(1)

                else:
                    response_idx = random.randint(0, len(sample["incorrect_ans"]) - 1)
                    input_text = f"{final_instr} Question: {sample['question']} Answer: {sample['incorrect_ans'][response_idx]}"
                    target = str(0)
            else:
                # Get a random index for sampling answer and correctness
                response_idx = random.randint(0, len(sample["all_model_responses"]) - 1)

                # Construct input text with randomly sampled answer
                input_text = f"{final_instr} Question: {sample['question']} Answer: {model_ans[response_idx]}"
                target = str(model_ans_correct_bin[response_idx])

        if format_rand > 0.5:
            target = "[[" + target + "]]"

        if case_rand > 0.6:
            if random.random() < 0.5:  # 50/50 chance between upper and lower
                input_text = input_text.upper()
                # target = target.upper()
            else:
                input_text = input_text.lower()
                # target = target.lower()

        # Tokenize input
        # Format conversation using model's chat template
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": target},
        ]

        # Tokenize everything together according to the chat template
        encodings = self.tokenizer(
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )

        # Get length of the user input for masking
        input_len = len(
            self.tokenizer(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": input_text}],
                    tokenize=False,
                    # add_generation_prompt=True,
                    add_generation_prompt=False,
                )
            )["input_ids"]
        )

        # Create labels by masking the input portion with -100
        labels = encodings["input_ids"].clone()
        labels[..., :input_len] = -100

        return {
            "input_ids": encodings["input_ids"][0],  # Remove .squeeze()
            "attention_mask": encodings["attention_mask"][0],  # Remove .squeeze()
            "labels": labels[0],  # Remove .squeeze()
        }


def get_model_save_path(
    base_model_name: str,
    save_dir: Optional[Path] = None,
    eval_loss: Optional[float] = None,
) -> Path:
    """
    Get the path where the calibrated model should be saved or loaded from.

    Args:
        base_model_name: Name or path of the base model (e.g. 'hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4')
        save_dir: Optional custom save directory
        eval_loss: Optional evaluation loss to include in the model name

    Returns:
        Path object pointing to the model directory
    """
    save_dir = save_dir or DEFAULT_SAVE_DIR

    # Remove any organization prefix (e.g., 'hugging-quants/')
    model_base_name = base_model_name.split("/")[-1]

    # Clean the model name - replace special characters with underscores
    model_base_name = model_base_name.replace("-", "_").replace(".", "_")

    # Add eval loss to the name if provided
    if eval_loss is not None:
        calibrated_name = f"calibrated_{model_base_name}_eval_loss_{eval_loss:.4f}"
    else:
        calibrated_name = f"calibrated_{model_base_name}"

    full_path = save_dir / calibrated_name

    return full_path


def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    model_path: str,
    save_dir: Optional[Path] = None,
    eval_loss: Optional[float] = None,
) -> None:
    """
    Save the trained model and tokenizer to a local directory.
    Handles both prompt tuning weights and modified norm weights.
    Args:
        model: The trained model
        tokenizer: The tokenizer
        model_path: Name or path of the base model
        save_dir: Optional directory to save the model in
        eval_loss: Optional evaluation loss to include in the model name
    """
    save_path = get_model_save_path(model_path, save_dir, eval_loss)
    save_path.mkdir(parents=True, exist_ok=True)
    # Save the prompt tuning adapter
    model.save_pretrained(save_path)
    # Save the modified layer norm weights from the base model
    norm_weights = {}
    for name, param in model.get_base_model().named_parameters():
        if "norm" in name.lower() and param.requires_grad:
            norm_weights[name] = param.data.cpu()
    if norm_weights:  # Only save if we have layer norm weights
        torch_save_path = save_path / "norm_weights.pt"
        torch.save(norm_weights, torch_save_path)
        print(f"Normalization weights saved to {torch_save_path}")
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved successfully to {save_path}")


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


def check_flash_attention(model, verbose=True):
    """
    Thoroughly check if Flash Attention is enabled in a loaded model.

    Args:
        model: The HuggingFace model to check
        verbose: If True, print detailed information about the checks

    Returns:
        bool: True if Flash Attention is enabled
    """

    def log(msg):
        if verbose:
            print(msg)

    try:
        # Get base model if it's a PEFT model
        if hasattr(model, "get_base_model"):
            base_model = model.get_base_model()
            log("Found PEFT model, checking base model...")
        else:
            base_model = model
            log("Checking model directly...")

        # Check config settings
        config = base_model.config
        config_checks = {
            "attn_implementation": getattr(config, "attn_implementation", None),
            "_attn_implementation": getattr(config, "_attn_implementation", None),
            "use_flash_attention": getattr(config, "use_flash_attention", None),
            "_flash_attn_2_enabled": getattr(config, "_flash_attn_2_enabled", None),
        }

        log("\nConfig settings:")
        for key, value in config_checks.items():
            log(f"- {key}: {value}")

        # Find an attention layer to check implementation
        def get_first_attention_layer(model):
            # Common model architectures
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return model.model.layers[0].self_attn
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                return model.transformer.h[0].attn
            elif hasattr(model, "layers"):
                return model.layers[0].self_attn
            elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
                return model.encoder.layers[0].self_attn
            return None

        attention_layer = get_first_attention_layer(base_model)

        if attention_layer is None:
            log("\nCould not find attention layer")
            return False

        log(f"\nAttention layer type: {type(attention_layer).__name__}")

        # Check various attributes that might indicate Flash Attention
        attention_checks = {
            "_flash_attn_2_enabled": getattr(
                attention_layer, "_flash_attn_2_enabled", None
            ),
            "_use_flash_attention_2": getattr(
                attention_layer, "_use_flash_attention_2", None
            ),
            "flash_attention": getattr(attention_layer, "flash_attention", None),
        }

        log("\nAttention layer attributes:")
        for key, value in attention_checks.items():
            log(f"- {key}: {value}")

        # Check if class name contains Flash
        has_flash_in_name = "Flash" in type(attention_layer).__name__
        log(f"\nAttention class has 'Flash' in name: {has_flash_in_name}")

        # More stringent check for Flash Attention being enabled
        # Now requires either Flash in the class name OR actual Flash Attention attributes set
        is_enabled = has_flash_in_name or any(
            v is True for v in attention_checks.values()
        )

        log(
            f"\nFlash Attention appears to be: {'ENABLED' if is_enabled else 'DISABLED'}"
        )

        if not is_enabled:
            log("\nReasons Flash Attention appears to be disabled:")
            if not has_flash_in_name:
                log("- Attention layer class doesn't contain 'Flash' in its name")
            if not any(attention_checks.values()):
                log(
                    "- No Flash Attention attributes are enabled on the attention layer"
                )
            if config_checks["attn_implementation"] != "flash_attention_2":
                log(
                    "- Main attention implementation config is not set to flash_attention_2"
                )

        return is_enabled

    except Exception as e:
        log(f"\nError during Flash Attention check: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    save_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the model and tokenizer, applying PEFT configuration and enabling training
    for soft prompts and normalization layers only.

    Args:
        model_name: Name or path of the base model
        device: Device to load the model on ("cuda" or "cpu")
        save_dir: Directory containing saved PEFT weights, if any
        verbose: If True, print detailed information about the model loading process

    Returns:
        Tuple of (model, tokenizer)
    """
    # Get the path for saving/loading the model
    local_model_path = get_model_save_path(model_name, save_dir)
    # if verbose:
    print(f"Loading base model from HuggingFace: {model_name}")

    if "parasail-ai/Mistral-7B-Instruct-v0.3-GPTQ-4bit" in model_name:
        identified_basename = get_model_basename(model_name)
        # print(f"{identified_basename = }")
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            model_basename=identified_basename,  # This is recognized by AutoGPTQ
            use_safetensors=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,  # or "auto"
            device="cuda:0",  # or "cpu" if no GPU
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map=device,
            device_map="auto",
            # use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    def unfreeze_specific_params(model_to_unfreeze: nn.Module) -> int:
        """
        Freeze all parameters except those related to normalization layers and prompts.
        Returns the count of unfrozen parameters.
        """
        unfrozen_count = 0

        # First freeze everything
        for param in model_to_unfreeze.parameters():
            param.requires_grad = False

        # Then selectively unfreeze based on parameter names
        for name, param in model_to_unfreeze.named_parameters():
            # Check for normalization layers
            if any(norm_name in name.lower() for norm_name in ["layernorm"]):
                param.requires_grad = True
                unfrozen_count += param.numel()
                if verbose:
                    print(f"Unfrozen norm parameter: {name}")

            # Check for prompt-related parameters
            if any(prompt_name in name.lower() for prompt_name in ["prompt", "prefix"]):
                param.requires_grad = True
                unfrozen_count += param.numel()
                if verbose:
                    print(f"Unfrozen prompt parameter: {name}")

        return unfrozen_count

    # Configure PEFT for soft prompts
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="Before formulating your response, take the necessary time to thoroughly read and analyze the provided instructions and the question. Ensure that you fully comprehend all requirements, constraints, and nuances of the request. Carefully consider how to structure your response to address each component effectively while adhering to any specified formatting or style guidelines. Once you have achieved a complete understanding of the question, construct a comprehensive and well-structured response that reflects your careful consideration of all aspects of the task. Pay close attention to the context of the question, recall all relevant information, and use it appropriately into your answer. Both the answer to the question and your response should be accurate, concise, and directly address the question without unnecessary elaboration. If any of these criteria are not met-such as incorrect information, excessive verbosity, or failure to directly answer the question-the response should be considered incorrect or inadequate and revised accordingly.",
        num_virtual_tokens=256,
        tokenizer_name_or_path=model_name,
    )

    # Load or create PEFT model
    if local_model_path.exists():
        if verbose:
            print(f"Found PEFT adapter weights and config at {local_model_path}")
        try:
            # Load prompt tuning adapter
            model = PeftModel.from_pretrained(model, local_model_path)
            # if verbose:
            print(f"Successfully loaded adapter weights from {local_model_path}")

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

                if verbose:
                    if missing_keys:
                        print(
                            f"Warning: Could not find these norm parameters: {missing_keys}"
                        )
                    else:
                        print("Successfully loaded all norm weights")
            else:
                print("No saved norm weights found, using initial values")

        except Exception as e:
            # if verbose:
            print(f"Error loading weights from {local_model_path}: {str(e)}")
            print("Creating fresh PEFT configuration")
            model = get_peft_model(model, peft_config)
    else:
        # if verbose:
        print(f"No existing weights found at {local_model_path}")
        print("Creating fresh PEFT configuration")
        model = get_peft_model(model, peft_config)

    # Unfreeze specific parameters
    unfrozen_params = unfreeze_specific_params(model)
    if verbose:
        print(f"\nTotal number of unfrozen parameters: {unfrozen_params:,}")

    # Set padding token if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def print_trainable_parameters(model: PreTrainedModel) -> None:
        """Print details about trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        prompt_params = 0
        norm_params = 0

        if verbose:
            print("\nTrainable layers:")

        for name, param in model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                if verbose:
                    print(f"Trainable layer: {name}, Parameters: {num_params:,}")

                # Track specific parameter types
                if any(
                    prompt_name in name.lower() for prompt_name in ["prompt", "prefix"]
                ):
                    prompt_params += num_params
                if any(
                    norm_name in name.lower()
                    for norm_name in ["norm", "ln", "layer_norm"]
                ):
                    norm_params += num_params

        # Print parameter counts
        print(f"\nTotal trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {all_params:,}")
        print(
            f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%"
        )

        # if verbose:
        print(f"\nBreakdown of trainable parameters:")
        print(f"Prompt parameters: {prompt_params:,}")
        print(f"Normalization parameters: {norm_params:,}")

        # Verify trainability
        prompts_trainable = prompt_params > 0
        norms_trainable = norm_params > 0

        print("\nVerifying trainable layers:")
        print(f"Soft prompts trainable: {prompts_trainable}")
        print(f"Normalization layers trainable: {norms_trainable}")

        if not norms_trainable:
            print("\nWARNING: No normalization parameters appear to be trainable!")
        if not prompts_trainable:
            print("\nWARNING: No prompt parameters appear to be trainable!")

    if verbose:
        print("\nFinal trainable parameters summary:")
    print_trainable_parameters(model)

    # check_flash_attention(model)

    return model, tokenizer


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


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 2,
    lr: float = 1e-5,
    gradient_accumulation_steps: int = 32,
    device: str = "cuda",
    save_dir: Optional[Path] = None,
    model_path: str = None,
) -> PreTrainedModel:
    """Improved training function with Accelerate integration"""

    # Setup optimizer with fused operations
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, fused=True
    )

    # Cosine schedule with warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_dataloader) * 0.1),
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    # Initialize accelerator with mixed precision and logging
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
        # log_with="tensorboard",
    )
    accelerator.init_trackers("model_training")

    # Prepare all components with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # Setup progress tracking
    total_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps
    progress_bar = tqdm(total=total_steps, desc="Training Progress")
    effective_step = 0
    min_eval_loss = float("inf")

    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            step_count = 0

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)

                    # Only track loss for logging when we actually update
                    if accelerator.sync_gradients:
                        step_count += 1
                        total_loss += accelerator.gather(loss).mean().item()
                        clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        # Update progress
                        effective_step += 1
                        current_avg_loss = total_loss / step_count

                        # Log metrics
                        accelerator.log(
                            {
                                "train_loss": current_avg_loss,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            step=effective_step,
                        )

                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            {
                                "epoch": f"{epoch+1}/{num_epochs}",
                                "avg_loss": f"{current_avg_loss:.4f}",
                            }
                        )

                        # Periodic memory cleanup
                        if effective_step % 100 == 0:
                            torch.cuda.empty_cache()

            # Evaluation phase
            if eval_dataloader and epoch % 2 == 0:
                model.eval()
                print("Running Eval...")
                eval_loss = 0
                eval_step_count = 0

                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)  # No need to move to device

                    loss = outputs.loss
                    eval_loss += accelerator.gather(loss).mean().item()
                    eval_step_count += 1

                eval_epoch_loss = eval_loss / eval_step_count
                accelerator.log({"eval_loss": eval_epoch_loss}, step=effective_step)
                print(f"Evaluation Loss: {eval_epoch_loss:.3f}")

                # Save best model
                if eval_epoch_loss < min_eval_loss:
                    min_eval_loss = eval_epoch_loss
                    if save_dir and model_path:
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model_and_tokenizer(
                            unwrapped_model,
                            tokenizer,
                            model_path,
                            save_dir,
                            eval_loss=float(eval_epoch_loss),
                        )
                        print("New checkpoint saved")

            # Regular checkpoint saving
            if epoch % 2 == 0 and save_dir and model_path:
                unwrapped_model = accelerator.unwrap_model(model)
                save_model_and_tokenizer(
                    unwrapped_model, tokenizer, model_path, save_dir
                )

        progress_bar.close()
        accelerator.end_training()
        return accelerator.unwrap_model(model)

    except Exception as e:
        print(f"Training error: {str(e)}")
        progress_bar.close()
        accelerator.end_training()
        raise


def create_collate_fn(tokenizer):
    """Creates a collate function with access to the tokenizer"""

    def collate_fn(batch):
        """Efficient collate function for variable-length sequences."""
        # Stack all tensors in the batch
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Pad to the max length in this batch
        max_length = max(len(ids) for ids in input_ids)

        # Create padded tensors
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            padding_length = max_length - len(ids)

            if padding_length > 0:
                # Pad input_ids with pad_token_id
                padded_input_ids.append(
                    torch.cat(
                        [
                            ids,
                            torch.full(
                                (padding_length,),
                                tokenizer.pad_token_id,
                                dtype=ids.dtype,
                            ),
                        ]
                    )
                )
                # Pad attention_mask with 0
                padded_attention_mask.append(
                    torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])
                )
                # Pad labels with -100
                padded_labels.append(
                    torch.cat(
                        [lab, torch.full((padding_length,), -100, dtype=lab.dtype)]
                    )
                )
            else:
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
                padded_labels.append(lab)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

    return collate_fn


def main(
    model_path: str,
    train_data_path: str,
    valid_data_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    num_epochs: int = 2,
) -> None:
    """
    Main function to run the calibration training pipeline.

    Args:
        model_path: Path or name of the base model
        data_path: Path to the training data JSON file
        save_dir: Optional directory to save the trained model
        num_epochs: Number of training epochs
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir_path = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR

    # Initialize tokenizer and model
    model, tokenizer = load_model_and_tokenizer(model_path, device, save_dir_path)

    # print(model.print_trainable_parameters())

    # Load and prepare data

    train_data = load_data_dict(train_data_path)

    # Create dataset and dataloader
    bs = 1
    collate_fn = create_collate_fn(tokenizer)
    train_dataset = CalibrationDataset(train_data, tokenizer, eval=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=default_data_collator,
        # collate_fn=collate_fn,
    )

    if valid_data_path:
        valid_data = load_data_dict(valid_data_path)
        # valid_dataset = CalibrationDataset(
        #     valid_data, tokenizer, instruction, eval=True
        # )
        valid_dataset = CalibrationDataset(valid_data, tokenizer, eval=True)
        eval_dataloader = DataLoader(
            valid_dataset,
            batch_size=bs,
            shuffle=True,
            collate_fn=default_data_collator,
            # collate_fn=collate_fn,
        )

    # Train the model
    trained_model = train_model(
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
        num_epochs=num_epochs,
        save_dir=save_dir_path,
        model_path=model_path,
    )

    # Save the final trained model
    save_model_and_tokenizer(trained_model, tokenizer, model_path, save_dir_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrating a model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path or name of the base model"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training data JSON file",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=True,
        help="Path to the valid data JSON file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Optional directory to save the trained model (default: calibrated_models/)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)",
    )

    args = parser.parse_args()
    main(
        args.model_path,
        args.train_data_path,
        args.valid_data_path,
        args.save_dir,
        args.num_epochs,
    )
