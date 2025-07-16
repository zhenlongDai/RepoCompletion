# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tree_sitter import Language, Parser
from utils.eval_utils import (
    postprocess_code_lines_for_train,
    extract_identifiers,
    cal_edit_sim,
    remove_comments,
    postprocess_python_code_lines
)
from evaluation.eval_metric import compute_id_match
from utils.code_util import extract_content, extract_content_in_answer, count_non_empty_lines


def accuracy_code_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    input_code = kwargs['input_code']
    language = kwargs['language']
    language_object = Language(f"build/{language[0]}-lang-parser.so", language[0])
    parser_util = Parser()
    parser_util.set_language(language_object)

    for content, sol, code, lan in zip(contents, solution, input_code, language):

        # Extract code from the content
        content = extract_content_in_answer(content, lan)
        # Postprocess the code to remove comments and format it
        content = postprocess_code_lines_for_train(code, content, parser_util, lan)
        content = remove_comments(content)
        pred_lines = [l.strip() for l in content.split("\n") if l.strip()]
        
        sol = remove_comments(sol)
        gt_lines = [l.strip() for l in sol.split("\n") if l.strip()]
        reward = int(pred_lines == gt_lines)
        rewards.append(reward)

    return rewards

def code_identifiers_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    input_code = kwargs['input_code']
    language = kwargs['language']
    language_object = Language(f"build/{language[0]}-lang-parser.so", language[0])
    parser_util = Parser()
    parser_util.set_language(language_object)

    for content, sol, code, lan in zip(contents, solution, input_code, language):

        # Extract code from the content
        content = extract_content_in_answer(content, lan)
        # Postprocess the code to remove comments and format it
        content = postprocess_code_lines_for_train(code, content, parser_util, lan)
        content = remove_comments(content)

        pred_ids = extract_identifiers(content, lan)
        sol = remove_comments(sol)
        target_ids = extract_identifiers(sol, lan)
        reward = int(pred_ids == target_ids)
        rewards.append(reward)

    return rewards

def code_identifiers_precision_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    input_code = kwargs['input_code']
    language = kwargs['language']
    language_object = Language(f"build/{language[0]}-lang-parser.so", language[0])
    parser_util = Parser()
    parser_util.set_language(language_object)

    for content, sol, code, lan in zip(contents, solution, input_code, language):

        # Extract code from the content
        content = extract_content_in_answer(content, lan)
        # Postprocess the code to remove comments and format it
        content = postprocess_code_lines_for_train(code, content, parser_util, lan)
        content = remove_comments(content)

        pred_ids = extract_identifiers(content, lan)
        sol = remove_comments(sol)
        target_ids = extract_identifiers(sol, lan)
        id_tp, id_fp, id_fn = compute_id_match(pred_ids, target_ids)
        reward = id_tp / (id_tp + id_fp) if (id_tp + id_fp) != 0 else 0
        rewards.append(reward)

    return rewards

def code_identifiers_Recall_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    input_code = kwargs['input_code']
    language = kwargs['language']
    language_object = Language(f"build/{language[0]}-lang-parser.so", language[0])
    parser_util = Parser()
    parser_util.set_language(language_object)

    for content, sol, code, lan in zip(contents, solution, input_code, language):

        # Extract code from the content
        content = extract_content_in_answer(content, lan)
        # Postprocess the code to remove comments and format it
        content = postprocess_code_lines_for_train(code, content, parser_util, lan)
        content = remove_comments(content)

        pred_ids = extract_identifiers(content, lan)
        sol = remove_comments(sol)
        target_ids = extract_identifiers(sol, lan)
        id_tp, id_fp, id_fn = compute_id_match(pred_ids, target_ids)
        reward = id_tp / (id_tp + id_fn) if (id_tp + id_fn) != 0 else 0.0
        rewards.append(reward)

    return rewards

def code_identifiers_F_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    input_code = kwargs['input_code']
    language = kwargs['language']
    language_object = Language(f"build/{language[0]}-lang-parser.so", language[0])
    parser_util = Parser()
    parser_util.set_language(language_object)

    for content, sol, code, lan in zip(contents, solution, input_code, language):

        # Extract code from the content
        content = extract_content_in_answer(content, lan)
        # Postprocess the code to remove comments and format it
        
        #content = postprocess_code_lines(code, content, parser_util, lan)
        content = remove_comments(content)

        pred_ids = extract_identifiers(content, lan)
        sol = remove_comments(sol)
        target_ids = extract_identifiers(sol, lan)
        id_tp, id_fp, id_fn = compute_id_match(pred_ids, target_ids)
        reward = 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) != 0 else 0
        rewards.append(reward)

    return rewards

def len_reward_by_code_identifiers(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len) 
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = code_identifiers_reward(
        completions, solution, **kwargs
    )

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = (length - min_len) / (max_len - min_len) - 0.5
        if length > 256:
            lambda_val = 0.0
        if is_correct == 1:
            reward = max(0, lambda_val)
        else:
            reward = 0

        rewards.append(float(reward))

    return rewards


def codelines_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len) 
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    # correctness = code_identifiers_reward(
    #     completions, solution, **kwargs
    # )

    # Extract code lines from the contents and calculate line counts
    line_counts = []
    for content in contents:
        content = extract_content_in_answer(content, kwargs['language'][0])
        content = remove_comments(content)
        # Count non-empty lines
        line_count = count_non_empty_lines(content)
        line_counts.append(line_count)

    # Calculate lengths (use line counts instead of character counts)
    lengths = line_counts
    min_len = min(lengths)
    max_len = max(lengths)

    rewards = []
    for length in lengths:
        if length == 1:
            reward = 1.0
        else:
            lambda_val = 0 #min(0, 0.5 - (length - min_len) / (max_len - min_len))
            reward = lambda_val

        rewards.append(float(reward))

    return rewards

def code_smi_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    input_code = kwargs['input_code']
    language = kwargs['language']
    language_object = Language(f"build/{language[0]}-lang-parser.so", language[0])
    parser_util = Parser()
    parser_util.set_language(language_object)

    for content, sol, code, lan in zip(contents, solution, input_code, language):
        # Extract code from the content
        content = extract_content_in_answer(content, lan)
        # Postprocess the code to remove comments and format it
        content = postprocess_code_lines_for_train(code, content, parser_util, lan)
        content = remove_comments(content)
        sol = remove_comments(sol)
        reward = round(cal_edit_sim([sol], [content]) /100.0,5)
        rewards.append(reward)

    return rewards

def getmix_code_identifiers_precision2smi_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    CodeID_precision_reward = code_identifiers_precision_reward(completions, solution, **kwargs)
    smi_reward = code_smi_reward(completions, solution, **kwargs)
    rewards = []
    for codeId_r, smi_r in zip(CodeID_precision_reward, smi_reward):
        if codeId_r > 0:
            reward = codeId_r + smi_r
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards

def getmix_code_identifiers_F2smi_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    CodeID_precision_reward = code_identifiers_F_reward(completions, solution, **kwargs)
    smi_reward = code_smi_reward(completions, solution, **kwargs)
    rewards = []
    for codeId_r, smi_r in zip(CodeID_precision_reward, smi_reward):
        
        reward = smi_r *(1+codeId_r)
        #else:   
        #    reward = 0.0
        #reward = smi_r *(1+codeId_r) #smi_r *(1+codeId_r)
        rewards.append(reward)

    return rewards

def getmix_s2EcodeF1_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    CodeID_precision_reward = code_identifiers_F_reward(completions, solution, **kwargs)
    smi_reward = code_smi_reward(completions, solution, **kwargs)
    rewards = []
    for codeId_r, smi_r in zip(CodeID_precision_reward, smi_reward):
        
        reward = smi_r * math.exp(codeId_r)
        #else:   
        #    reward = 0.0
        #reward = smi_r *(1+codeId_r) #smi_r *(1+codeId_r)
        rewards.append(reward)

    return rewards

def getmix_codeF1expS_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    CodeID_precision_reward = code_identifiers_F_reward(completions, solution, **kwargs)
    smi_reward = code_smi_reward(completions, solution, **kwargs)
    rewards = []
    for codeId_r, smi_r in zip(CodeID_precision_reward, smi_reward):
        
        reward = (1+codeId_r) * math.exp(smi_r)
        #else:   
        #    reward = 0.0
        #reward = smi_r *(1+codeId_r) #smi_r *(1+codeId_r)
        rewards.append(reward)

    return rewards

def getmix_code_identifiers_EM_2smi_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    CodeID_precision_reward = code_identifiers_reward(completions, solution, **kwargs)
    smi_reward = code_smi_reward(completions, solution, **kwargs)
    rewards = []
    for codeId_r, smi_r in zip(CodeID_precision_reward, smi_reward):
        if codeId_r > 0:
            reward = smi_r
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards

def get_code_format_reward(completions, **kwargs):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```.*?```.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def get_context_code_format_reward(completions, **kwargs):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<context>\n.*?\n</context>\n<intent>\n.*?\n</intent>\n<answer>\n.*?```.*?```.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.20
        if text.count("\n</think>\n") == 1:
            count += 0.20
        if text.count("\n<answer>\n") == 1:
            count += 0.20
        if text.count("\n</answer>") == 1:
            count += 0.20
        if text.count("\n```") == 2:
            count += 0.20
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward



def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer




def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_code_reward,
        "code_identifiers": code_identifiers_reward,
        "code_id_precision": code_identifiers_precision_reward,
        "code_id_F": code_identifiers_F_reward,
        "code_similarity": code_smi_reward,
        "mix_codeIDES": getmix_code_identifiers_precision2smi_reward,
        "judge_codeID2ES": getmix_code_identifiers_EM_2smi_reward,
        "mix_codeIDF12ES": getmix_code_identifiers_F2smi_reward,
        "codeF1expS_reward": getmix_codeF1expS_reward,
        "s2EcodeF1_reward":getmix_s2EcodeF1_reward,
        "format": format_reward,
        "codelines_reward":codelines_reward,
        "context_format": get_context_code_format_reward,
        "code_format": get_code_format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "thought_length_by_CodeID": len_reward_by_code_identifiers,
        "tag_count": tag_count_reward,
        
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs


def exmpale3():
    input_code = "import abc\nimport binascii\nimport calendar\nimport copy\nimport hashlib\nimport os\nimport re\nimport six\nfrom datetime import datetime\nfrom cryptography.hazmat.primitives import constant_time\nfrom cryptography.hazmat.primitives.asymmetric import padding\nfrom .fields import DSAPriv, DSAPub, DSASignature\nfrom .fields import ECDSAPub, ECDSAPriv, ECDSASignature\nfrom .fields import ECDHPub, ECDHPriv, ECDHCipherText\nfrom .fields import ElGCipherText, ElGPriv, ElGPub\nfrom .fields import OpaquePubKey\nfrom .fields import OpaquePrivKey\nfrom .fields import RSACipherText, RSAPriv, RSAPub, RSASignature\nfrom .fields import String2Key\nfrom .fields import SubPackets\nfrom .fields import UserAttributeSubPackets\nfrom .types import Packet\nfrom .types import Primary\nfrom .types import Private\nfrom .types import Public\nfrom .types import Sub\nfrom .types import VersionedPacket\nfrom ..constants import CompressionAlgorithm\nfrom ..constants import HashAlgorithm\nfrom ..constants import PubKeyAlgorithm\nfrom ..constants import SignatureType\nfrom ..constants import SymmetricKeyAlgorithm\nfrom ..constants import TrustFlags\nfrom ..constants import TrustLevel\nfrom ..decorators import sdproperty\nfrom ..errors import PGPDecryptionError\nfrom ..symenc import _decrypt\nfrom ..symenc import _encrypt\nfrom ..types import Fingerprint\n    def pubalg(self):\n        return self._pubalg\n\n    @pubalg.register(int)\n    @pubalg.register(PubKeyAlgorithm)\n    def pubalg_int(self, val):\n        self._pubalg = PubKeyAlgorithm(val)\n        if self._pubalg in [PubKeyAlgorithm.RSAEncryptOrSign, PubKeyAlgorithm.RSAEncrypt, PubKeyAlgorithm.RSASign]:\n            self.signature = RSASignature()\n\n        elif self._pubalg == PubKeyAlgorithm.DSA:\n            self.signature = DSASignature()\n\n    @sdproperty\n    def halg(self):\n        return self._halg\n\n    @halg.register(int)\n    @halg.register(HashAlgorithm)\n    def halg_int(self, val):\n        try:\n            self._halg = HashAlgorithm(val)\n\n        except ValueError:  # pragma: no cover\n            self._halg = val\n\n    @sdproperty\n    def signer(self):\n        return self._signer\n\n    @signer.register(str)\n    @signer.register(six.text_type)\n    def signer_str(self, val):\n        self._signer = val\n\n    @signer.register(bytearray)\n    def signer_bin(self, val):\n        self._signer = binascii.hexlify(val).upper().decode('latin-1')\n\n    def __init__(self):\n        super(OnePassSignatureV3, self).__init__()\n        self._sigtype = None\n        self._halg = None\n        self._pubalg = None\n        self._signer = b'\\x00' * 8\n        self.nested = False\n\n    def __bytearray__(self):\n        _bytes = bytearray()\n        _bytes += super(OnePassSignatureV3, self).__bytearray__()\n        _bytes += bytearray([self.sigtype])\n        _bytes += bytearray([self.halg])\n        _bytes += bytearray([self.pubalg])\n        _bytes += binascii.unhexlify(six.b(self.signer))\n        _bytes += bytearray([int(self.nested)])\n        return _bytes\n\n    def parse(self, packet):\n        super(OnePassSignatureV3, self).parse(packet)\n        self.sigtype = packet[0]\n        del packet[0]\n\n        self.halg = packet[0]\n        del packet[0]\n\n        self.pubalg = packet[0]\n        del packet[0]\n\n        self.signer = packet[:8]\n        del packet[:8]\n\n        self.nested = (packet[0] == 1)\n        del packet[0]\n\n\nclass PrivKey(VersionedPacket, Primary, Private):\n    __typeid__ = 0x05\n    __ver__ = 0\n\n\nclass PubKey(VersionedPacket, Primary, Public):\n    __typeid__ = 0x06\n    __ver__ = 0\n\n    @abc.abstractproperty\n    def fingerprint(self):\n        \"\"\"compute and return the fingerprint of the key\"\"\"\n\n\nclass PubKeyV4(PubKey):\n    __ver__ = 4\n\n    @sdproperty\n    def created(self):\n        return self._created\n\n    @created.register(datetime)\n    def created_datetime(self, val):\n        self._created = val\n\n    @created.register(int)\n    def created_int(self, val):\n        self.created = datetime.utcfromtimestamp(val)\n\n    @created.register(bytes)\n    @created.register(bytearray)\n    def created_bin(self, val):\n        self.created = self.bytes_to_int(val)\n\n    @sdproperty\n    def pkalg(self):\n        return self._pkalg\n\n    @pkalg.register(int)\n    @pkalg.register(PubKeyAlgorithm)\n    def pkalg_int(self, val):\n        self._pkalg = PubKeyAlgorithm(val)\n\n        _c = {\n            # True means public\n\n"
    next_line = "<answer>```python\n            PubKeyAlgorithm.RSAEncryptOrSign: True,\n            PubKeyAlgorithm.RSAEncrypt: True,\n            PubKeyAlgorithm.RSASign: True,\n            PubKeyAlgorithm.DSA: True,\n            PubKeyAlgorithm.ECDSA: True,\n            PubKeyAlgorithm.EdDSA: True,\n```</answer>"
    language = "python"
    content = extract_content_in_answer(next_line, language)
    print("Extracted Code:", content)

 
    language_object = Language(f"build/{language}-lang-parser.so", language)
    parser_util = Parser()
    parser_util.set_language(language_object)
    content_1 = postprocess_python_code_lines(input_code, content, parser_util, language)
    len = count_non_empty_lines(content_1)
    print("Postprocessed Python Code:", content_1, len)
    
    content_2 = postprocess_code_lines_for_train(input_code, content, parser_util, language)
    len = count_non_empty_lines(content_2)
    print("Postprocessed Code:", content, len)
if __name__ == "__main__":
  # get_code_format_reward example
    # pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```.*?```.*?\n</answer>$"
    # completions = [[{"content": "<think>\nStep 1: Do something\n</think>\n<answer>\n```python\nprint('Hello World')\n```\n</answer>\n\n```\n```\n```\n```\n```\n```\n```\n```\n```"}]]
    # rewards = get_code_format_reward(completions)
    # print("Code Format Reward:", rewards)
    # completions = [[{"content": "<context>\nThis is a context.\n</context>\n<intent>\nWhat is the answer?\n</intent>\n<answer>\n```python\nprint('Hello World')\n```\n"}]]
    # rewards = get_code_format_reward(completions)
    # print("Context Code Format Reward:", rewards)
    exmpale3()