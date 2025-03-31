from sklearn.metrics.pairwise import cosine_similarity


def calculate_structure_similarity(dict1, dict2):
    """
    Calculates the Jaccard Similarity between the key structure of two JSON-like
    data structures based on the number of keys at each nesting level.

    Args:
        dict1 (dict): A dictionary where keys are nesting levels and values are
                      the number of keys at that level (obtained from
                      count_keys_per_level for the first JSON).
        dict2 (dict): A dictionary where keys are nesting levels and values are
                      the number of keys at that level (obtained from
                      count_keys_per_level for the second JSON).

    Returns:
        float: The Jaccard Similarity score between 0.0 and 1.0, representing
               the structural similarity based on key counts per level.
               Returns 0.0 if the union of features is empty.
    """

    features1 = set((level, count) for level, count in dict1.items())
    features2 = set((level, count) for level, count in dict2.items())

    intersection = features1.intersection(features2)
    union = features1.union(features2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def calculate_semantic_similarity(embedding1, embedding2):
    """
    Calculates the normalized cosine similarity between two embedding vectors.

    Args:
        embedding1 (numpy.ndarray or None): The embedding vector of the first JSON.
        embedding2 (numpy.ndarray or None): The embedding vector of the second JSON.

    Returns:
        float: The normalized cosine similarity between the two embeddings, ranging
               from 0.0 (no similarity) to 1.0 (perfect similarity). Returns 0.0
               if either embedding is None.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    
    return similarity