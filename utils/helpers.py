from datetime import datetime


def count_keys_per_level(json_data, level=0, key_counts=None):
    """
    Recursively navigates a JSON-like data structure and counts the number of
    keys at each level of nesting.

    Args:
        json_data (dict or list): The JSON data to analyze.
        level (int): The current level of nesting (starts at 0).
        key_counts (dict): A dictionary to store the key counts per level.
                           Initialized as an empty dictionary if None.

    Returns:
        dict: A dictionary where keys are the nesting levels (integers) and
              values are the number of keys found at that level.
    """
    if key_counts is None:
        key_counts = {}
        
    if isinstance(json_data, dict):
        key_counts[level] = key_counts.get(level, 0) + len(json_data.keys())
        for value in json_data.values():
            count_keys_per_level(value, level + 1, key_counts)
    elif isinstance(json_data, list):
        for item in json_data:
            count_keys_per_level(item, level + 1, key_counts)
            
    # For primitive types (string, number, boolean, None), we don't have keys,
    # so we simply return and don't increment the level further.
    
    return key_counts


def extract_text_content(json_data):
    """
    Recursively extracts all *key* strings from JSON data.
    Values are not included in the extracted text for embedding.

    Args:
        json_data (dict, list, str, int, float, bool, None): The JSON data to process.

    Returns:
        list: A list of strings containing all the keys found in the JSON.
    """
    text_list = []
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            text_list.append(str(key))  # Always append the key
            # Recursively call for nested structures (dict or list) to get their keys
            text_list.extend(extract_text_content(value))
    elif isinstance(json_data, list):
        for item in json_data:
            # Recursively call for items in list; they could be nested dicts
            text_list.extend(extract_text_content(item))
    # If json_data is a base type (str, int, float, bool, None),
    # we do NOT append its value, as the goal is to extract only keys.
    # The function simply returns the current (empty or partially filled) text_list
    # for these base cases.
    return text_list

def get_json_embedding(json_data, model):
    """
    Generates an embedding for the combined text content of a JSON.

    Args:
        json_data (dict, list, str, int, float, bool, None): The JSON data to embed.
        model: the embedding model (e.g. transformer)

    Returns:
        numpy.ndarray or None: A dense vector representing the semantic embedding
                               of the JSON's text content, or None if no text
                               content is found.
    """
    text_content = extract_text_content(json_data)
    if not text_content:
        return None  # Handle empty JSON

    # Option 1: Concatenate all text and get one embedding
    combined_text = " ".join(text_content)
    embedding = model.encode(combined_text)

    # Option 2: Get embeddings for each text and average them (commented out)
    # embeddings = model.encode(text_content)
    # embedding = np.mean(embeddings, axis=0) if embeddings.size > 0 else None

    return embedding

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
    