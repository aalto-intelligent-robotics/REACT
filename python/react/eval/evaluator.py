from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple, Dict, Set

from react.matching.ground_truth import GroundTruth
from react.matching.match_results import MatchResults


@dataclass
class ReactEvaluator:
    ground_truth: GroundTruth
    match_results: MatchResults
    results_cnt: Dict[str, int] = field(default_factory=dict)
    old_checked_id: Set = field(default_factory=set)
    new_checked_id: Set = field(default_factory=set)

    def is_correct_match(self, pred_match: Tuple[int, int]):
        old_node_id = pred_match[0]
        new_node_id = pred_match[1]
        for gt_match in self.ground_truth.matches.values():
            if (
                old_node_id in gt_match[self.ground_truth.old_scan_id]
                and new_node_id in gt_match[self.ground_truth.new_scan_id]
            ):
                return True
        return False

    def is_completely_absent(self, pred: int):
        for gt_match in self.ground_truth.matches.values():
            old_gt_cluster = gt_match[self.ground_truth.old_scan_id]
            if pred in old_gt_cluster:
                return False  # Could be potential match with something else
        return True

    def is_completely_new(self, pred: int):
        for gt_match in self.ground_truth.matches.values():
            new_gt_cluster = gt_match[self.ground_truth.new_scan_id]
            if pred in new_gt_cluster:
                return False
        return True

    def count_matched_results(self):
        # Matched
        tp_m = 0
        fp_m = 0
        tp_a = 0
        fp_a = 0
        fn_a = 0
        tp_n = 0
        fp_n = 0
        fn_n = 0
        for match in self.match_results.matches:
            self.old_checked_id.add(match[0])
            self.new_checked_id.add(match[1])
            if self.is_correct_match(match):
                tp_m += 1
            else:
                fp_m += 1
                if self.is_completely_absent(match[0]):
                    fn_a += 1
                if self.is_completely_new(match[1]):
                    fn_n += 1
        fn_m = self.ground_truth.get_num_matches() - tp_m

        # Check completely new / absent objects (not part of a cluster that is 
        # present in both dsg)
        for new in self.match_results.new:
            if self.is_completely_new(new):
                tp_n += 1
        for absent in self.match_results.absent:
            if self.is_completely_absent(absent):
                tp_a += 1
        # Calculate conflicts
        tmp_gt = deepcopy(self.ground_truth)
        for match in tmp_gt.matches.values():
            for i in self.old_checked_id:
                if i in match[tmp_gt.old_scan_id]:
                    match[tmp_gt.old_scan_id].remove(i)
            for j in self.new_checked_id:
                if j in match[tmp_gt.new_scan_id]:
                    match[tmp_gt.new_scan_id].remove(j)
        for cid, cluster in tmp_gt.matches.items():
            old_c = cluster[tmp_gt.old_scan_id]
            new_c = cluster[tmp_gt.new_scan_id]

            pred_num_abs = len(old_c)
            gt_num_abs = tmp_gt.get_cluster_num_abs(cid)
            if pred_num_abs < gt_num_abs:
                fn_a += gt_num_abs - pred_num_abs
                tp_a += pred_num_abs
            else:
                tp_a += gt_num_abs
                fp_a += pred_num_abs - gt_num_abs

            pred_num_new = len(new_c)
            gt_num_new = tmp_gt.get_cluster_num_new(cid)
            if pred_num_new < gt_num_new:
                fn_n += gt_num_new - pred_num_new
                tp_n += pred_num_new
            else:
                tp_n += gt_num_new
                fp_n += pred_num_new - gt_num_new
        self.results_cnt = {
            "tp_m": tp_m,
            "fp_m": fp_m,
            "fn_m": fn_m,
            "tp_a": tp_a,
            "fp_a": fp_a,
            "fn_a": fn_a,
            "tp_n": tp_n,
            "fp_n": fp_n,
            "fn_n": fn_n,
        }
        print(self.results_cnt)

    def count_new_results(self) -> Dict[str, int]:
        tp = 0
        fp = 0

        fn = self.ground_truth.get_num_new() - tp
        return {"tp": tp, "fp": fp, "fn": fn}

    def calculate_pr(self, tp, fp, fn):
        if tp == 0:
            pre = 0
            rec = 0
        else:
            pre = tp / (tp + fp)
            rec = tp / (tp + fn)
        return {"pre": pre, "rec": rec}

    def get_precision_recall(self) -> Dict[str, Dict[str, int]]:
        pr = {}
        pr["m"] = {}
        pr["m"] = self.calculate_pr(
            self.results_cnt["tp_m"], self.results_cnt["fp_m"], self.results_cnt["fn_m"]
        )
        pr["a"] = {}
        pr["a"] = self.calculate_pr(
            self.results_cnt["tp_a"], self.results_cnt["fp_a"], self.results_cnt["fn_a"]
        )
        pr["n"] = {}
        pr["n"] = self.calculate_pr(
            self.results_cnt["tp_n"], self.results_cnt["fp_n"], self.results_cnt["fn_n"]
        )
        return pr
