import json

def load_json(path : str = '', mode : str = 'r'):
    with open(path, mode) as f:
        return json.load(f)
    

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
    Recursively extracts all string values (keys and string-based values) from JSON data.

    Args:
        json_data (dict, list, str, int, float, bool, None): The JSON data to process.

    Returns:
        list: A list of strings containing all the textual content found in the JSON.
    """
    text_list = []
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            text_list.append(str(key))  # Treat keys as text too
            text_list.extend(extract_text_content(value))
    elif isinstance(json_data, list):
        for item in json_data:
            text_list.extend(extract_text_content(item))
    elif isinstance(json_data, (str, int, float, bool, type(None))):
        text_list.append(str(json_data))
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