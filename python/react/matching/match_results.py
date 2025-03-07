from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class MatchResults:
    """Class to store instance matching results.

    Attributes:
        old_scan_id: the reference scan id
        new_scan_id: the current scan id
        matches: list of matching node id pairs
        absent: list of absent node ids
        new: list of new node ids
        travel_distance: total travel distances
    """

    old_scan_id: int
    new_scan_id: int
    matches: List[Tuple] = field(default_factory=list)
    absent: List[int] = field(default_factory=list)
    new: List[int] = field(default_factory=list)
    travel_distance: float = 0.0
