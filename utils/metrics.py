from sklearn.metrics.pairwise import cosine_similarity


def calculate_structure_similarity(dict1, dict2):
    """
    Calculates the Jaccard Similarity between the key structure of two JSON-like
    data structures, giving equal weight to each nesting level.
    The similarity is computed as the average of per-level Jaccard similarities.

    Args:
        dict1 (dict): A dictionary where keys are nesting levels and values are
                      the number of keys at that level (obtained from
                      count_keys_per_level for the first JSON).
        dict2 (dict): A dictionary where keys are nesting levels and values are
                      the number of keys at that level (obtained from
                      count_keys_per_level for the second JSON).

    Returns:
        float: The Jaccard Similarity score between 0.0 and 1.0, representing
               the structural similarity based on key counts per level,
               where each level contributes equally.
               Returns 1.0 if both input dictionaries are empty (perfectly similar empty structures).
    """

    all_levels = set(dict1.keys()).union(set(dict2.keys()))

    if not all_levels:
        # If both input dictionaries are empty (e.g., from an empty JSON or only primitive types),
        # they are considered perfectly similar.
        return 1.0

    total_level_similarity = 0.0

    for level in all_levels:
        count1 = dict1.get(level, 0)
        count2 = dict2.get(level, 0)

        # Calculate local Jaccard for this level
        if max(count1, count2) == 0:
            # If both counts are 0 for this level, it's a perfect match for an empty level
            level_jaccard = 1.0
        else:
            level_jaccard = min(count1, count2) / max(count1, count2)
        
        total_level_similarity += level_jaccard

    # Average the similarities across all unique levels
    return total_level_similarity / len(all_levels)


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