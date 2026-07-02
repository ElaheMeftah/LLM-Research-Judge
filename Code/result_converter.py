"""Helpers for switching between row-style and column-style results."""

def convert_dict_of_lists_to_list_of_dicts(results_dict):
    """Convert column-style results into row-style records."""
    if not results_dict:
        return []
    
    # Get the length from the first non-empty list
    num_abstracts = 0
    for key, values in results_dict.items():
        if isinstance(values, list) and len(values) > 0:
            num_abstracts = len(values)
            break
    
    if num_abstracts == 0:
        return []
    
    # Convert to list of dictionaries
    list_of_dicts = []
    for i in range(num_abstracts):
        abstract_dict = {}
        for key, values in results_dict.items():
            if isinstance(values, list) and i < len(values):
                abstract_dict[key] = values[i]
            else:
                abstract_dict[key] = None
        list_of_dicts.append(abstract_dict)
    
    return list_of_dicts


def convert_list_of_dicts_to_dict_of_lists(list_of_dicts):
    """Convert row-style records into column-style results."""
    if not list_of_dicts:
        return {}
    
    # Get all possible keys
    all_keys = set()
    for item in list_of_dicts:
        all_keys.update(item.keys())
    
    # Convert to dictionary of lists
    dict_of_lists = {}
    for key in all_keys:
        dict_of_lists[key] = [item.get(key) for item in list_of_dicts]
    
    return dict_of_lists


# Convenience function for immediate use
def get_results_as_list_of_dicts(results_dict):
    """Return results as row-style records."""
    return convert_dict_of_lists_to_list_of_dicts(results_dict)
