import os
import pandas as pd
import json

def process_directory(directory_path):
    """
    Reads CSV/TSV files from a directory, collects all unique column names,
    and returns a dictionary of column names and their inferred formats
    with an example value from the first instance.
    
    Args:
        directory_path (str): The path to the directory containing CSV/TSV files.
    
    Returns:
        dict: A dictionary where keys are column names (from all processed files)
              and values are example data values for those columns,
              inferred from the first non-null instance.
    """
    column_metadata = {}
    
    # Determine the ID column name based on the directory
    id_column_name = None
    if 'aurora_db' in directory_path:
        id_column_name = 'pid'
    elif 'mimic-iii' in directory_path:
        id_column_name = 'subject_id'
    elif 'vital_db' in directory_path:
        id_column_name = 'subjectid'

    # Add the ID column to metadata initially, if identified.
    # Its example value will be populated from data if found.
    if id_column_name:
        column_metadata[id_column_name] = "Example ID value (will be populated from data)"

    # Iterate through files in the specified directory
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        df = None

        # Process CSV/TSV files
        # Read only a small number of rows (e.g., 100) to infer types and get example values.
        # Pandas is quite inefficient in merging with large datasets.
        # Still, we need only some sample rows as we ar einterested in their schema rather than content
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=100) 
        elif filename.endswith('.tsv'):
            df = pd.read_csv(filepath, sep='\t', nrows=100)
        
        if df is not None:
            # Handle the ID column: update its example value from the first record
            if id_column_name and id_column_name in df.columns:
                if pd.notna(df[id_column_name].iloc[0]):
                    column_metadata[id_column_name] = str(df[id_column_name].iloc[0])

            # Infer column formats and get the first non-null example values for other columns
            for col_name, dtype in df.dtypes.items():
                # Skip the ID column as it's handled separately to ensure its presence
                if col_name == id_column_name:
                    continue

                # Determine the inferred type of the column for schema representation
                inferred_type = 'unknown'
                if pd.api.types.is_integer_dtype(dtype):
                    inferred_type = 'integer'
                elif pd.api.types.is_float_dtype(dtype):
                    inferred_type = 'float'
                elif pd.api.types.is_bool_dtype(dtype):
                    inferred_type = 'boolean'
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    inferred_type = 'datetime'
                elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                    inferred_type = 'string'
                else:
                    inferred_type = str(dtype) # Fallback for any other data types

                # Add the column to metadata only if it hasn't been seen before.
                # The first encountered non-null value is used as an example.
                if col_name not in column_metadata:
                    example_value = None
                    # Find the index of the first non-null value in the column
                    first_valid_idx = df[col_name].first_valid_index()
                    if first_valid_idx is not None:
                        val = df[col_name].loc[first_valid_idx]
                        
                        # Convert value based on inferred type to ensure JSON compatibility
                        if inferred_type == 'datetime':
                            example_value = pd.to_datetime(val).isoformat()
                        elif inferred_type == 'integer' or inferred_type == 'float':
                            try:
                                example_value = val.item() 
                            except (ValueError, AttributeError):
                                example_value = str(val) 
                        elif inferred_type == 'boolean':
                            example_value = bool(val)
                        else:
                            example_value = str(val)
                    
                    column_metadata[col_name] = example_value

    return column_metadata

def create_nested_json(flat_data):
    """
    Transforms a flat dictionary of column metadata (col_name: example_value)
    into a nested JSON structure. Keys are nested based on underscores.
    A prefix will form a nested dictionary only if multiple columns share that prefix.
    
    Args:
        flat_data (dict): A flat dictionary containing column names as keys and
                          their example values.
    
    Returns:
        dict: A nested dictionary representing the JSON structure with grouped
              columns.
    """
    nested_data = {}
    
    def set_nested_value(d, path_parts, val):
        """
        Helper function to set a value deeply within a nested dictionary,
        creating intermediate dictionaries as needed.
        
        Args:
            d (dict): The dictionary to modify.
            path_parts (list): A list of strings representing the nested path
                                (e.g., ['dilution', 'text']).
            val (any): The value to set at the end of the path.
        """
        current = d
        for i, part in enumerate(path_parts):
            if i == len(path_parts) - 1: 
                current[part] = val
            else: 
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part] # Move deeper into the nested structure

    # First Pass: Identify prefixes that appear in multiple column names.
    # These prefixes will become keys for nested dictionaries.
    prefixes_to_group = {} 
    for col_name in flat_data.keys():
        # Split only on the first underscore to identify the top-level prefix.
        parts = col_name.split('_', 1)
        if len(parts) > 1: # Only consider columns with at least one underscore
            prefix = parts[0]
            if prefix not in prefixes_to_group:
                prefixes_to_group[prefix] = []
            prefixes_to_group[prefix].append(col_name)

    # Determine the set of column names that belong to a 'group' and should be nested.
    # A column is part of a group if its first prefix is shared by more than one column.
    grouped_columns_set = set()
    for prefix, col_names_list in prefixes_to_group.items():
        if len(col_names_list) > 1: 
            for col_name in col_names_list:
                grouped_columns_set.add(col_name)
    
    # Second Pass: Populate the nested_data structure based on identified groups.
    for col_name, value in flat_data.items():
        if col_name in grouped_columns_set:
            # Neste key case
            parts = col_name.split('_', 1)
            prefix = parts[0]
            
            remaining_key_parts = parts[1].split('_') 

            # Ensure the top-level 'prefix' key in nested_data is a dictionary.
            # If it doesn't exist, or if it exists but is not a dictionary, initialize it.
            if prefix not in nested_data or not isinstance(nested_data[prefix], dict):
                nested_data[prefix] = {}
            
            # Set the value at the correct nested path.
            set_nested_value(nested_data[prefix], remaining_key_parts, value)
        else:
            # This column remains a top-level key in the final JSON.
            nested_data[col_name] = value
            
    return nested_data

def simplify_single_key_nesting(data):
    """
    Recursively simplifies a nested dictionary by merging a parent key with its
    child key if the nested dictionary contains only one key-value pair.
    This helps in flattening unnecessary single-level nesting.
    
    Args:
        data (dict or any): The dictionary (or sub-dictionary) to simplify.
    
    Returns:
        dict or any: The simplified dictionary, or the original value if not a dict.
    """
    if not isinstance(data, dict):
        return data  
    
    new_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            simplified_value = simplify_single_key_nesting(value)
            
            if isinstance(simplified_value, dict) and len(simplified_value) == 1:
                inner_key, inner_value = list(simplified_value.items())[0]
                new_key = f"{key}_{inner_key}" 
                new_data[new_key] = inner_value
            else:
                new_data[key] = simplified_value
        else:
            new_data[key] = value
    
    return new_data


if __name__ == "__main__":

    # Process each specified directory to get flat column metadata
    dir_one_formats = process_directory('./aurora_db')
    dir_two_formats = process_directory('./mimic-iii-clinical-database-demo-1.4')
    dir_three_formats = process_directory('./vital_db')
    
    # Apply initial nesting logic to the collected metadata
    nested_aurora_formats = create_nested_json(dir_one_formats)
    nested_mimic_formats = create_nested_json(dir_two_formats)
    nested_vital_formats = create_nested_json(dir_three_formats)

    # Apply the simplification logic for single-key nested dictionaries recursively
    final_aurora_formats = simplify_single_key_nesting(nested_aurora_formats)
    final_mimic_formats = simplify_single_key_nesting(nested_mimic_formats)
    final_vital_formats = simplify_single_key_nesting(nested_vital_formats)
    
    # Output the final JSON files with the new nested and simplified structure
    with open('aurora_db.json', 'w') as f:
        json.dump(final_aurora_formats, f, indent=4)

    with open('mimic_iii_db.json', 'w') as f:
        json.dump(final_mimic_formats, f, indent=4)
    
    with open('vital_db.json', 'w') as f:
        json.dump(final_vital_formats, f, indent=4)

    # Print the content of the generated JSON files for verification
    print("Processing complete. JSON files generated with simplified nesting.")
    print("\nContent of 'aurora_db.json':")
    print(json.dumps(final_aurora_formats, indent=4))
    print("\nContent of 'mimic_iii_db.json':")
    print(json.dumps(final_mimic_formats, indent=4))
    print("\nContent of 'vital_db.json':")
    print(json.dumps(final_vital_formats, indent=4))
