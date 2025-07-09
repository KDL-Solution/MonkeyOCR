from typing import Dict, List


def dict_to_list(
    input_dict: Dict,
) -> List:
    items_list = []
    for _, item in input_dict.items():
        items_list.append(item)
    return items_list
