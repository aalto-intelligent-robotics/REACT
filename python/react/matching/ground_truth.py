from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class GroundTruth:
    """
    Class to store ground truth information

    Attributes:
        old_scan_id: the reference scan id
        new_scan_id: the current scan id
        matches: list of matching node ids, mapped as
            {cluster_id -> {scan_id -> node_id}}
    """

    old_scan_id: int
    old_num_nodes: int
    new_scan_id: int
    new_num_nodes: int
    matches: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)

    def get_cluster_num_abs(self, cluster_id: int):
        cluster = self.matches[cluster_id]
        return max(0, len(cluster[self.old_scan_id]) - len(cluster[self.new_scan_id]))

    def get_cluster_num_new(self, cluster_id: int):
        cluster = self.matches[cluster_id]
        return max(0, len(cluster[self.new_scan_id]) - len(cluster[self.old_scan_id]))

    def get_num_matches(self) -> int:
        num_matches = 0
        for match in self.matches.values():
            num_matches += min(
                len(match[self.old_scan_id]), len(match[self.new_scan_id])
            )
        return num_matches

    def get_num_absent(self) -> int:
        num_matches = self.get_num_matches()
        num_absent = self.old_num_nodes - num_matches
        return num_absent

    def get_num_new(self) -> int:
        num_matches = self.get_num_matches()
        num_new = self.new_num_nodes - num_matches
        return num_new
