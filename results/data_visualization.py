import os
import argparse
import pprint
from distutils.util import strtobool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd


def plot_similarity_graph(csv_file_path, save_path='./', output_file_name='similarity_graph.png'):
    """
    Plots a scatter graph showing structural vs. semantic similarity for
    pairwise comparisons from a single CSV file, with different colors for each pair.
    """
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    try:
        df = pd.read_csv(csv_file_path)
        df = df.rename(columns={
            'structure': 'Structure Similarity',
            'semantic': 'Semantic Similarity'
        })
        df['Pair'] = df['x'] + ' vs ' + df['y']
    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}")
        return

    all_pairs = sorted(df['Pair'].unique())
    palette = sns.color_palette("pastel", n_colors=len(all_pairs))
    pair_to_color = {pair: palette[i] for i, pair in enumerate(all_pairs)}

    pair_handles = {}
    for pair in all_pairs:
        pair_df = df[df['Pair'] == pair]
        if not pair_df.empty:
            ax.scatter(
                pair_df['Structure Similarity'],
                pair_df['Semantic Similarity'],
                label=pair,
                marker='o',
                s=200,
                alpha=0.8,
                edgecolor='w',
                linewidth=1,
                color=pair_to_color[pair]
            )
            if pair not in pair_handles:
                pair_handles[pair] = plt.Line2D([0], [0], marker='o', color='w',
                                                label=pair,
                                                markerfacecolor=pair_to_color[pair],
                                                markersize=12, markeredgecolor='k')

    plt.title('Before Harmonization - Pairwise DBs Structure vs. Semantic Similarity', fontsize=16)
    plt.xlabel('Structure Similarity', fontsize=12)
    plt.ylabel('Semantic Similarity', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(handles=list(pair_handles.values()), title="DBs Pair (Color)", loc='upper right')
    plt.tight_layout()

    output_file_path = os.path.join(save_path, output_file_name)
    plt.savefig(output_file_path, dpi=300)
    print(f"Saved plot to {output_file_path}")


def plot_different_llms_similarity(csv_file_paths, similarity_type, save_path='./', output_file_name='similarity_graph.png'):
    """
    Plots a grouped bar plot showing the specified similarity type of JSON messages vs Harmonized schema
    for each LLM (x-axis). Each LLM has a set of bars, one for each dataset in the CSV file.
    Adds a second legend with mean (μ) and variance (σ²) for each LLM.
    """

    records = []
    dataset_names = set()
    llm_stats = {}

    for csv_path in csv_file_paths:
        try:
            llm_name = csv_path.split('/')[1].split('_')[0]
            df = pd.read_csv(csv_path)
            similarity_values = []
            for _, row in df.iterrows():
                dataset = row['x']
                similarity = row[similarity_type]
                records.append({'LLM': llm_name, 'Dataset': dataset, f'{similarity_type.capitalize()} Similarity': similarity})
                dataset_names.add(dataset)
                similarity_values.append(similarity)
            if similarity_values:
                llm_stats[llm_name] = {
                    'mean': np.mean(similarity_values),
                    'var': np.std(similarity_values)
                }
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    if not records:
        print("No data to plot.")
        return

    plot_df = pd.DataFrame(records)
    dataset_names = sorted(dataset_names)
    palette = sns.color_palette("pastel", n_colors=len(dataset_names))
    dataset_to_color = {ds: palette[i] for i, ds in enumerate(dataset_names)}

    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        data=plot_df,
        x='LLM',
        y=f'{similarity_type.capitalize()} Similarity',
        hue='Dataset',
        palette=dataset_to_color,
        errorbar=None
    )

    plt.ylabel(f"{similarity_type.capitalize()} Similarity", fontsize=12)
    plt.xlabel("LLM", fontsize=12)
    plt.title(f"After Harmonization - {similarity_type.capitalize()} Similarity per DB for Each LLM", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # First legend (datasets)
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles, labels, title="Dataset", loc='upper left')

    # Second legend (mean and variance)
    stats_patches = []
    for llm, stats in llm_stats.items():
        label = f"{llm}: μ={stats['mean']:.2f}, σ²={stats['var']:.3f}"
        patch = mpatches.Patch(color='none', label=label)
        stats_patches.append(patch)
    
    legend2 = plt.legend(handles=stats_patches, title="LLM Stats", loc='upper right', frameon=True)
    ax.add_artist(legend1)  # Keep the first legend

    output_file_path = os.path.join(save_path, output_file_name)
    plt.savefig(output_file_path, dpi=300)
    print(f"Saved grouped bar plot to {output_file_path}")


def plot_similarity_heatmaps(csv_paths, similarity_type, save_path='./', output_file_name='similarity_graph.png'):
    """
    Reads semantic similarity data from CSV files and plots heatmaps.

    The first CSV in the list is assumed to contain pairwise similarities
    among the initial four JSONs. Subsequent CSVs contain similarities
    between the initial four JSONs and a 'final_harmonized_schema' JSON
    for different LLMs.

    Args:
        csv_paths (list[str]): A list of file paths to the CSV files.
                               The first path should be the base pairwise CSV.
                               Subsequent paths are for LLM-specific similarities.
        similarity_type (str): The type of similarity to plot ('semantic' or 'structural').
    """

    if not csv_paths:
        print("No CSV paths provided. Please provide at least one CSV file.")
        return

    # Define all five unique JSON names that will form the axes of our heatmaps.
    json_names = [
        '1_aurora_db.json',
        '2_mimic_iii_db.json',
        '3_mimic_iv_db.json',
        '4_vital_db.json', 
        'final_harmonized_schema'
    ]

    # --- Step 1: Create and populate the base similarity matrix ---
    # This matrix will hold all the initial pairwise similarities.
    # It's initialized with NaN (Not a Number) to clearly show missing values before they are filled from the CSVs.
    base_similarity_matrix = pd.DataFrame(np.nan, index=json_names, columns=json_names)

    # Set diagonal elements to 1.0, as a JSON is perfectly similar to itself.
    for name in json_names:
        base_similarity_matrix.loc[name, name] = 1.0

    # Read the first CSV file, which contains the pairwise similarities among the initial four JSONs.
    base_csv_path = csv_paths[0]
    print(f"Processing base CSV for initial pairwise similarities: {base_csv_path}")
    try:
        df_base = pd.read_csv(base_csv_path)

        for _, row in df_base.iterrows():
            json_x = row['x']
            json_y = row['y']
            similarity = row[similarity_type]

            if json_x in json_names and json_y in json_names:
                base_similarity_matrix.loc[json_x, json_y] = similarity
                base_similarity_matrix.loc[json_y, json_x] = similarity
            else:
                print(f"Warning: JSON '{json_x}' or '{json_y}' from '{base_csv_path}' "
                      f"is not in the predefined list of JSON names. Skipping this entry.")

    except FileNotFoundError:
        print(f"Error: Base CSV file not found at '{base_csv_path}'. Please check the path.")
        return
    except KeyError as e:
        print(f"Error: Missing expected column in '{base_csv_path}'. "
              f"Please ensure 'x', 'y', and '{similarity_type}' columns exist. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading '{base_csv_path}': {e}")
        return

    # --- Step 2: Plot the base heatmap ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        base_similarity_matrix,
        annot=True,
        cmap='viridis',
        fmt=".2f",    
        linewidths=.5,
        linecolor='black',
        cbar_kws={'label': f'{similarity_type.capitalize()} Similarity'} 
    )
    plt.title(f'Pairwise {similarity_type.capitalize()} Similarities (Initial JSONs) - Base Data', fontsize=14)
    plt.xlabel('JSONs', fontsize=12)
    plt.ylabel('JSONs', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)             
    plt.tight_layout() 
    output_file_path = os.path.join(save_path, f'base_{output_file_name}')
    plt.savefig(output_file_path, dpi=300)
    print(f"Saved base {similarity_type} similarity heatmap to base_{output_file_path}")

    # --- Step 3: Process and plot LLM-specific heatmaps ---
    for i, llm_csv_path in enumerate(csv_paths[1:]):
        
        llm_name = llm_csv_path.split('/')[1].split('_')[0]

        print(f"Processing LLM-specific CSV for {llm_name}: {llm_csv_path}")

        llm_similarity_matrix = base_similarity_matrix.copy()

        try:
            df_llm = pd.read_csv(llm_csv_path)

            for _, row in df_llm.iterrows():
                json_initial = row['x']
                json_harmonized = row['y']
                similarity = row[similarity_type]

                if json_harmonized == 'final_harmonized_schema' and \
                   json_initial in json_names and json_harmonized in json_names:
                    llm_similarity_matrix.loc[json_initial, json_harmonized] = similarity
                    llm_similarity_matrix.loc[json_harmonized, json_initial] = similarity
                else:
                    print(f"Warning: Unexpected row format in '{llm_csv_path}': {row.to_dict()}. "
                          f"Expected 'y' to be 'final_harmonized_schema' and 'x' to be an initial JSON.")

            # Plot the heatmap for the current LLM.
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                llm_similarity_matrix,
                annot=True,
                cmap='viridis',
                fmt=".2f",
                linewidths=.5,
                linecolor='black',
                cbar_kws={'label': f'{similarity_type.capitalize()} Similarity'}
            )
            plt.title(f'{similarity_type.capitalize()} Similarities with Final Harmonized Schema ({llm_name})', fontsize=14)
            plt.xlabel('JSONs', fontsize=12)
            plt.ylabel('JSONs', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            output_file_path = os.path.join(save_path, f'{llm_name}_{output_file_name}')
            plt.savefig(output_file_path, dpi=300)
            print(f"Saved {llm_name} {similarity_type} similarity heatmap to {llm_name}_{output_file_name}")

        except FileNotFoundError:
            print(f"Error: LLM CSV file not found at '{llm_csv_path}'. Please check the path.")
            continue 
        except KeyError as e:
            print(f"Error: Missing expected column in '{llm_csv_path}'. "
                  f"Please ensure 'x', 'y', and '{similarity_type}' columns exist. Details: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while reading '{llm_csv_path}': {e}")
            continue


def parseargs():
    parser = argparse.ArgumentParser(description="Dataspace and AI: Schema Harmonization ~ Data Visualization")
    
    parser.add_argument('--input_data', nargs='+', default=['./results.csv'], type=str,
                        help='paths to the results CSV files (space-separated)')
    parser.add_argument('--save_path', default='./', type=str,
                        help='where to save the output figure')
    parser.add_argument('--output_file_name', default='similarity_graph.png', type=str,
                        help='name of the output file')
    parser.add_argument('--plot_different_llms', default='False', type=lambda x: bool(strtobool(x)), 
                        help='plot different LLMs results together')
    parser.add_argument('--plot_llms_heatmaps', default='False', type=lambda x: bool(strtobool(x)), 
                        help='plot different LLMs semantic similarity heatmaps, if true, the first CSV file of --input_data reports the initial pairwise similarities between dbs, while the following CSV file paths are LLM-specific')
    parser.add_argument('--similarity_type', default='semantic', type=str, choices=['semantic', 'structure'],
                        help='Type of similarity to plot (semantic or structural). Only applicable for heatmaps and grouped bar plots.')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parseargs()
    
    # Pretty print the parsed arguments
    print("Run Configuration:")
    pprint.pprint(vars(args))
    print()
    
    if args.plot_different_llms:
        if args.plot_llms_heatmaps:
            # Grouped bar plot for different LLMs, choosing similarity type
            plot_different_llms_similarity(
                csv_file_paths=args.input_data[1:], 
                similarity_type=args.similarity_type,
                save_path=args.save_path, 
                output_file_name=args.output_file_name
            )
            
            # Heatmaps for LLM harmonization results, choosing similarity type
            plot_similarity_heatmaps(
                csv_paths=args.input_data, 
                similarity_type=args.similarity_type,
                save_path=args.save_path, 
                output_file_name=args.output_file_name
            )
        else:
            # Grouped bar plot for different LLMs, choosing similarity type
            plot_different_llms_similarity(
                csv_file_paths=args.input_data, 
                similarity_type=args.similarity_type,
                save_path=args.save_path, 
                output_file_name=args.output_file_name
            )
    else:
        # Plot initial similarity graph (structural vs semantic, so no change here)
        plot_similarity_graph(
            csv_file_path=args.input_data[0],
            save_path=args.save_path,  
            output_file_name=args.output_file_name
        )