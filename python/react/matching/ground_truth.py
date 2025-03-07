from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class GroundTruth:
    """Class to store ground truth instance matching information.

    :param old_scan_id: The reference scan ID.
    :param old_num_nodes: The number of nodes in the old scan.
    :param new_scan_id: The current scan ID.
    :param new_num_nodes: The number of nodes in the new scan.
    :param matches: List of matching node IDs, mapped as a dictionary
        {cluster_id -> {scan_id -> node_id}}.
    """

    old_scan_id: int
    old_num_nodes: int
    new_scan_id: int
    new_num_nodes: int
    matches: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)

    def get_cluster_num_absent(self, cluster_id: int):
        """Get the number of absent nodes in a cluster.

        This method calculates the number of nodes present in the old
        scan but absent in the new scan for a specific cluster.

        :param cluster_id: The ID of the cluster.
        :return: The number of absent nodes in the cluster.
        """
        cluster = self.matches[cluster_id]
        return max(0, len(cluster[self.old_scan_id]) - len(cluster[self.new_scan_id]))

    def get_cluster_num_new(self, cluster_id: int):
        """Get the number of new nodes in a cluster.

        This method calculates the number of nodes present in the new
        scan but absent in the old scan for a specific cluster.

        :param cluster_id: The ID of the cluster.
        :return: The number of new nodes in the cluster.
        """
        cluster = self.matches[cluster_id]
        return max(0, len(cluster[self.new_scan_id]) - len(cluster[self.old_scan_id]))

    def get_num_matches(self) -> int:
        """Get the number of matching nodes between the old and new scans.

        This method calculates the total number of matching nodes
        between the old and new scans across all clusters.

        :return: The number of matching nodes.
        """
        num_matches = 0
        for match in self.matches.values():
            num_matches += min(
                len(match[self.old_scan_id]), len(match[self.new_scan_id])
            )
        return num_matches

    def get_num_absent(self) -> int:
        """Get the number of absent nodes in the old scan.

        This method calculates the total number of nodes present in the
        old scan but absent in the new scan.

        :return: The number of absent nodes in the old scan.
        """
        num_matches = self.get_num_matches()
        num_absent = self.old_num_nodes - num_matches
        return num_absent

    def get_num_new(self) -> int:
        """Get the number of new nodes in the new scan.

        This method calculates the total number of nodes present in the
        new scan but absent in the old scan.

        :return: The number of new nodes in the new scan.
        """
        num_matches = self.get_num_matches()
        num_new = self.new_num_nodes - num_matches
        return num_new
