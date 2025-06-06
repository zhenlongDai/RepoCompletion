import argparse
from evaluation.eval_metric import process_examples
from utils.dataset_util import load_ground_truth
from utils.json_util import load_list_from_json
from tree_sitter import Language, Parser
from functools import partial
from collections import Counter, defaultdict

def get_em_map(prediction_list, groundtruth_map, language, parser):
    em_map = {}
    worker = partial(process_examples, language, parser)
    for sample in prediction_list:
        ex = groundtruth_map[sample["id"]]
        trunc_s, em_label = worker((sample, ex))
        em_map[sample["id"]] = em_label
    return em_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_path", type=str, required=True)
    parser.add_argument("--eval_dataset_name", type=str, required=True)
    parser.add_argument("--pred1", type=str, required=True, help="First prediction file")
    parser.add_argument("--pred2", type=str, required=True, help="Second prediction file")
    parser.add_argument("--language", type=str, default="java")
    parser.add_argument("--ts_lib", type=str, default="build/java-lang-parser.so")
    args = parser.parse_args()

    groundtruth = load_ground_truth(args.eval_dataset_name, args.data_dir_path, 'test')
    groundtruth_map = {ex["id"]: {"in_file_prompt": ex["in_file_prompt"], "ground_truth": ex["ground_truth"]} for ex in groundtruth}

    pred1_list = load_list_from_json(args.pred1)
    pred2_list = load_list_from_json(args.pred2)

    assert len(pred1_list) == len(pred2_list) == len(groundtruth_map)

    # Build mapping from id to tag
    id2tag = {}
    for pred in pred1_list + pred2_list:
        if "tag" in pred:
            id2tag[pred["id"]] = pred["tag"]

    language_obj = Language(args.ts_lib, args.language)
    parser_util = Parser()
    parser_util.set_language(language_obj)

    em_map1 = get_em_map(pred1_list, groundtruth_map, args.language, parser_util)
    em_map2 = get_em_map(pred2_list, groundtruth_map, args.language, parser_util)

    ids = list(groundtruth_map.keys())
    drop_ids = [i for i in ids if em_map1[i] == 1 and em_map2[i] == 0]
    gain_ids = [i for i in ids if em_map1[i] == 0 and em_map2[i] == 1]

    em1 = sum(em_map1.values()) / len(em_map1)
    em2 = sum(em_map2.values()) / len(em_map2)

    # Count tag distribution
    drop_tag_counter = Counter()
    gain_tag_counter = Counter()
    drop_tag_ids = defaultdict(list)
    gain_tag_ids = defaultdict(list)

    for i in drop_ids:
        tag = id2tag.get(i, "None")
        drop_tag_counter[tag] += 1
        drop_tag_ids[tag].append(i)
    for i in gain_ids:
        tag = id2tag.get(i, "None")
        gain_tag_counter[tag] += 1
        gain_tag_ids[tag].append(i)

    print(f"pred1 EM: {em1:.4f}")
    print(f"pred2 EM: {em2:.4f}")
    print(f"Number of cases that passed before but not now: {len(drop_ids)}")
    print(f"ID: {drop_ids}")
    print("Statistics by tag (passed before but not now):")
    for tag, cnt in drop_tag_counter.items():
        print(f"  tag={tag}: {cnt} cases, ID: {drop_tag_ids[tag]}")
    print(f"Number of cases that did not pass before but pass now: {len(gain_ids)}")
    print(f"ID: {gain_ids}")
    print("Statistics by tag (did not pass before but pass now):")
    for tag, cnt in gain_tag_counter.items():
        print(f"  tag={tag}: {cnt} cases, ID: {gain_tag_ids[tag]}")

    tag_diff = {tag: (drop_tag_counter[tag], gain_tag_counter[tag]) for tag in set(drop_tag_counter) | set(gain_tag_counter)}
    print("Statistics by tag (passed before but not now, did not pass before but pass now):")
    print(f"  total: -{len(drop_ids)}|+{len(gain_ids)}") 
    for tag, (cnt1, cnt2) in tag_diff.items():
        print(f"  tag={tag}: -{cnt1}|+{cnt2}") 

if __name__ == "__main__":
    main()