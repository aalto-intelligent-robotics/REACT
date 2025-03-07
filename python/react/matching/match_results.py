from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class MatchResults:
    """Class to store instance matching results.

    :param old_scan_id: the reference scan id
    :param new_scan_id: the current scan id
    :param matches: list of matching node id pairs
    :param absent: list of absent node ids
    :param new: list of new node ids
    :param travel_distance: total travel distances
    """

    old_scan_id: int
    new_scan_id: int
    matches: List[Tuple] = field(default_factory=list)
    absent: List[int] = field(default_factory=list)
    new: List[int] = field(default_factory=list)
    travel_distance: float = 0.0
