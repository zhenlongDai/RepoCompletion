import json
from functools import partial
import torch.multiprocessing as mp
from tqdm import tqdm
from tree_sitter import Language, Parser
import argparse
from utils.json_util import save_list_to_json, load_list_from_json
from utils.dataset_util import load_ground_truth
from utils.code_util import extract_content

from utils.eval_utils import (
    postprocess_code_lines,
    extract_identifiers,
    cal_edit_sim,
    remove_comments
)
import os

def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn


def compute_edit_sim(samples):
    refs, hyps = [], []
    for s in samples:
        refs.append(s["target"])
        hyps.append(s["pred"])
    return cal_edit_sim(refs, hyps)


def process_examples(language, parser, args):
    sample, ex = args
    code_text = extract_content(sample["generated_text"], language)
    prediction = postprocess_code_lines(ex["in_file_prompt"], code_text, parser, language)
    prediction = remove_comments(prediction)
    target = ex["ground_truth"]
    target = remove_comments(target)

    pred_lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    em_label = int(pred_lines == gt_lines)

    pred_ids = extract_identifiers(prediction, language)
    target_ids = extract_identifiers(target, language)

    trunc_s = {
        "task_id": sample["id"],
        "pred": prediction,
        "target": target,
        "pred_ids": pred_ids,
        "target_ids": target_ids
    }
    return trunc_s, em_label


def compute_metric(args):
    groundtruth = load_ground_truth(args.eval_dataset_name, args.data_dir_path)
    prediction_list = load_list_from_json(args.prediction_file)
    groundtruth_map = {}
    for ex in groundtruth:
        groundtruth_map[ex["id"]] = {
            "in_file_prompt": ex["in_file_prompt"],
            "ground_truth": ex["ground_truth"]
        }
    assert len(prediction_list) == len(groundtruth_map), f"{len(prediction_list)} != {len(groundtruth_map)}"

    ts_lang =  args.language
    language = Language(args.ts_lib, ts_lang)
    parser_util = Parser()
    parser_util.set_language(language)
    truncated_samples = []
    em_labels = []

    print("post-processing samples ...")
    worker = partial(process_examples, args.language, parser_util)

    with tqdm(total=len(prediction_list)) as pbar:
        for sample, example in zip(prediction_list, [groundtruth_map[s["id"]] for s in prediction_list]):
            output = worker((sample, example))
            trunc_s, em_label = output
            em_labels.append(em_label)
            truncated_samples.append(trunc_s)
            pbar.update()

    exact_match = 0
    for em_label in em_labels:
        if em_label == 1:
            exact_match += 1

    ### Score calculation

    id_em = []
    edit_similarities = []
    detailed_results = []

    for idx, trunc_s in enumerate(truncated_samples):
        identifier_em = int(trunc_s["pred_ids"] == trunc_s["target_ids"])
        es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        id_tp, id_fp, id_fn = compute_id_match(trunc_s["pred_ids"], trunc_s["target_ids"])
        id_em.append(identifier_em)
        edit_similarities.append(es)

        detailed_results.append({
            "task_id": trunc_s["task_id"],
            "em": em_labels[idx],
            "es": es,
            "id_em": identifier_em,
            "id_precision": id_tp / (id_tp + id_fp) if (id_tp + id_fp) != 0 else 0,
            "id_recall": id_tp / (id_tp + id_fn) if (id_tp + id_fn) != 0 else 0,
            "id_f1": 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) != 0 else 0,
        })

    em_ratio = round(exact_match / len(prediction_list) * 100, 2)
    edit_sim = round(sum(edit_similarities) / len(edit_similarities), 2)

    id_em_ratio = round(
        sum(detailed_results[idx]['id_em'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)
    id_precision = round(sum(detailed_results[idx]['id_precision'] for idx in range(len(detailed_results))) / len(
        detailed_results) * 100, 2)
    id_recall = round(
        sum(detailed_results[idx]['id_recall'] for idx in range(len(detailed_results))) / len(detailed_results) * 100,
        2)
    id_f1 = round(
        sum(detailed_results[idx]['id_f1'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)

    print(
        f"Code Matching: "
        f"EM {em_ratio:.2f}, "
        f"ES {edit_sim:.2f}"
    )

    print(
        f"ID matching: "
        f"EM {id_em_ratio}, "
        f"Precision {id_precision}, "
        f"Recall {id_recall}, "
        f"F1 {id_f1}"
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_path",
                        type=str,
                        default="None",
                        help="the data path of ground truth")
    parser.add_argument("--eval_dataset_name",
                        type=str,
                        default="None",
                        help="the eval dataset name")
    parser.add_argument("--prediction_file",
                        type=str,
                        default="None",
                        help="the file of prediction")
    parser.add_argument("--language",
                        type=str,
                        default="java",
                        help="Language")
    parser.add_argument("--ts_lib",
                        type=str,
                        default="build/java-lang-parser.so",
                        help="Tree-sitter library path")
    args = parser.parse_args()

    compute_metric(args)

if __name__ == '__main__':
    main()