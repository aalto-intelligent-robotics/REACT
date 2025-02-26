from typing import Dict, Tuple, List
import numpy as np
from scipy.optimize import linear_sum_assignment
import logging

from react.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="hungarian_algorithm.log",
)


def hungarian_algorithm(
    old_inst_positions: Dict[int, np.ndarray],
    new_inst_positions: Dict[int, np.ndarray],
    include_z: bool = False,
) -> Tuple[List, List]:
    """Perform Hungarian algorithm to minimize objects' travelled distance
    between 2 scans.

    :param old_inst_positions: Position at time t {global_inst_id -> xyz
        position}
    :param new_inst_positions: Position at time t+1 {global_inst_id ->
        xyz position}
    :param include_z: Include z axis when calculating travelled distance
    :return: Lists of matching global ids for the reference and current
        scan
    """
    old_inst_ids = list(old_inst_positions.keys())
    new_inst_ids = list(new_inst_positions.keys())

    old_positions = list(old_inst_positions.values())
    new_positions = list(new_inst_positions.values())
    logger.debug(f"old positions {old_positions}")
    logger.debug(f"new positions {new_positions}")
    if not include_z:
        old_positions = [pos[:-1] for pos in old_positions]
        new_positions = [pos[:-1] for pos in new_positions]
    num_col = len(old_positions)
    num_row = len(new_positions)
    cost_matrix = np.zeros([num_row, num_col])
    for c in range(num_col):
        for r in range(num_row):
            cost_matrix[r, c] = np.linalg.norm(new_positions[r] - old_positions[c])
    logger.debug(f"Cost matrix {cost_matrix}")
    new_ind, old_ind = linear_sum_assignment(cost_matrix)
    logger.debug(f"Matching old ids {old_ind}")
    logger.debug(f"Matching new ids {new_ind}")
    matching_old_ids = [old_inst_ids[i] for i in old_ind]
    matching_new_ids = [new_inst_ids[i] for i in new_ind]
    return matching_old_ids, matching_new_ids
