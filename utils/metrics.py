from sklearn.metrics.pairwise import cosine_similarity


def calculate_structure_similarity(keys1, blocks1, keys2, blocks2):
    """
    Calculates the structural similarity between two JSON-like data structures
    by comparing both:
    - The number of keys per nesting level
    - The number of nested blocks per nesting level (dicts or lists)

    Similarity is computed using a per-level Jaccard-style score and averaged
    over all levels.

    Args:
        keys1 (dict): Key counts per level for JSON 1.
        blocks1 (dict): Nested block counts per level for JSON 1.
        keys2 (dict): Key counts per level for JSON 2.
        blocks2 (dict): Nested block counts per level for JSON 2.

    Returns:
        float: Structural similarity score between 0.0 and 1.0.
    """

    all_levels = set(keys1.keys()) | set(keys2.keys()) | set(blocks1.keys()) | set(blocks2.keys())

    if not all_levels:
        return 1.0  # Perfect similarity for empty structures

    key_sim_total = 0.0
    block_sim_total = 0.0

    for level in all_levels:
        # --- Key similarity per level ---
        k1 = keys1.get(level, 0)
        k2 = keys2.get(level, 0)
        if max(k1, k2) == 0:
            key_sim = 1.0
        else:
            key_sim = min(k1, k2) / max(k1, k2)
        key_sim_total += key_sim

        # --- Block similarity per level ---
        b1 = blocks1.get(level, 0)
        b2 = blocks2.get(level, 0)
        if max(b1, b2) == 0:
            block_sim = 1.0
        else:
            block_sim = min(b1, b2) / max(b1, b2)
        block_sim_total += block_sim

    # Average over levels
    avg_key_sim = key_sim_total / len(all_levels)
    avg_block_sim = block_sim_total / len(all_levels)

    # Final similarity is the average of both components
    final_similarity = round((avg_key_sim + avg_block_sim) / 2, 2)
    return final_similarity


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
    
    return round(similarity, 2)