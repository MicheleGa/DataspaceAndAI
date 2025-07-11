from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


def count_keys_and_nested_blocks(json_data, level=0, key_counts=None, nested_blocks=None):
    """
    Recursively navigates a JSON-like data structure and counts:
    - The number of keys at each level of nesting.
    - The number of nested blocks (dicts or lists) at each level.

    Args:
        json_data (dict or list): The JSON data to analyze.
        level (int): The current level of nesting (starts at 0).
        key_counts (dict): A dictionary to store the key counts per level.
        nested_blocks (dict): A dictionary to store the nested block counts per level.

    Returns:
        tuple: (key_counts, nested_blocks)
            - key_counts: dict of nesting level → number of keys
            - nested_blocks: dict of nesting level → number of dict/list blocks
    """
    if key_counts is None:
        key_counts = {}
    if nested_blocks is None:
        nested_blocks = {}

    if isinstance(json_data, dict):
        key_counts[level] = key_counts.get(level, 0) + len(json_data)
        nested_blocks[level] = nested_blocks.get(level, 0) + 1  # this dict is a nested block
        for value in json_data.values():
            count_keys_and_nested_blocks(value, level + 1, key_counts, nested_blocks)
    elif isinstance(json_data, list):
        nested_blocks[level] = nested_blocks.get(level, 0) + 1  # this list is a nested block
        for item in json_data:
            count_keys_and_nested_blocks(item, level + 1, key_counts, nested_blocks)
    
    return key_counts, nested_blocks


def extract_flat_keys(json_data):
    """
    Recursively extracts all keys from a JSON-like data structure
    and returns them in a flat list.
    
    Args:
        json_data (dict or list): The JSON data to analyze.
    
    Returns:
        list: A flat list of keys extracted from the JSON data.    
    """
    
    keys = []
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            keys.append(key)
            keys.extend(extract_flat_keys(value))
    elif isinstance(json_data, list):
        for item in json_data:
            keys.extend(extract_flat_keys(item))
    return keys


def extract_key_paths(json_data, prefix=''):
    """
    Recursively extracts all key paths from a JSON-like data structure
    and returns them in a flat list. Each key path is represented as a string
    with dot notation for nested keys.
    
    Args:
        json_data (dict or list): The JSON data to analyze.
        prefix (str): The current prefix for nested keys, used for recursion.
    
    Returns:
        list: A flat list of key paths extracted from the JSON data.    
    """
    
    keys = []
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            keys.extend(extract_key_paths(value, full_key))
    elif isinstance(json_data, list):
        for item in json_data:
            full_key = f"{prefix}[]" if prefix else "[]"
            keys.extend(extract_key_paths(item, full_key))
    return keys


def align_keys(source_keys, target_keys, model, similarity_threshold=0.7):
    """
    Aligns keys from a source set to a target set using a language model
    to compute semantic similarity. Returns a list of keys from the target set
    that are semantically similar to the source keys, based on a cosine similarity threshold.
    
    Args:
        source_keys (list): List of keys from the source set.
        target_keys (list): List of keys from the target set.
        model: A pre-trained language model used for encoding keys.
        similarity_threshold (float): The cosine similarity threshold for alignment.
    
    Returns:
        list: A list of keys from the target set that are aligned with the source keys.
    """ 
    emb_source = {k: model.encode(k) for k in source_keys}
    emb_target = {k: model.encode(k) for k in target_keys}
    aligned_keys = set()
    for k1, v1 in emb_source.items():
        best_match = None
        best_score = -1
        for k2, v2 in emb_target.items():
            score = cosine_similarity([v1], [v2])[0][0]
            if score > best_score:
                best_score = score
                best_match = k2
        if best_score >= similarity_threshold:
            aligned_keys.add(best_match)

    return list(aligned_keys)


def embed_keys(keys, model):
    """
    Calculates the embedding for a list of keys using the provided model.
    Keys are sorted and duplicates are removed before embedding.
    
    Args:
        keys (list): A list of keys to be embedded.
        model: A pre-trained language model used for encoding keys.
    
    Returns:
        numpy.ndarray: The embedding vector for the combined keys, or None if no keys are provided
    """
    if not keys:
        return None
    keys = sorted(set(keys))  # Normalize order and remove duplicates
    combined_text = " ".join(keys)
    return model.encode(combined_text)


def get_json_embedding(
    json_data,
    model,
    use_key_paths=False,
    reference_keys=None,
    use_key_alignment=False,
    similarity_threshold=0.6,
):
    """
    Generates an embedding for a JSON-like data structure.
    This function extracts keys from the JSON data, optionally aligns them with
    a reference set of keys, and embeds them using the provided model.
    
    Args:
        json_data (dict or list): The JSON data to analyze.
        model: A pre-trained language model used for encoding keys.
        use_key_paths (bool): If True, extracts key paths instead of flat keys.
        reference_keys (list, optional): A list of keys to align against.
        use_key_alignment (bool): If True, aligns extracted keys with reference keys.
        similarity_threshold (float): The cosine similarity threshold for alignment.
    
    Returns:
        numpy.ndarray: The embedding vector for the JSON data, or None if no keys are extracted
    """
    if use_key_paths:
        keys = extract_key_paths(json_data)
    else:
        keys = extract_flat_keys(json_data)

    if use_key_alignment and reference_keys is not None:
        keys = align_keys(reference_keys, keys, model, similarity_threshold)

    return embed_keys(keys, model)


def generate_run_name(args):
    """
    Generates a run name based on the provided arguments.

    Args:
        args: The argument parser containing the experiment settings.

    Returns:
        str: A formatted string representing the run name.
    """
    if args.experiment_name != '':
        return f"{args.model_name}_{args.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        return f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    