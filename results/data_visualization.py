import os
import argparse
from distutils.util import strtobool
import matplotlib.pyplot as plt
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
            'structure': 'Structural Similarity',
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
                pair_df['Structural Similarity'],
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

    plt.title('Before Harmonization - Pairwise DBs Structural vs. Semantic Similarity', fontsize=16)
    plt.xlabel('Structural Similarity', fontsize=12)
    plt.ylabel('Semantic Similarity', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(handles=list(pair_handles.values()), title="Dataset (Color)", loc='upper right')
    plt.tight_layout()

    output_file_path = os.path.join(save_path, output_file_name)
    plt.savefig(output_file_path, dpi=300)
    print(f"Saved plot to {output_file_path}")
    
    
def plot_different_llms_semnatic_similarity(csv_file_paths, save_path='./', output_file_name='similarity_graph.png'):
    """
    Plots a grouped bar plot showing the semantic similarity of JSON messages vs Harmonized schema
    for each LLM (x-axis). Each LLM has a set of bars, one for each dataset in the CSV file.
    """
    
    # Collect all data in a long-form DataFrame
    records = []
    dataset_names = set()
    for csv_path in csv_file_paths:
        try:
            llm_name = csv_path.split('/')[1].split('_')[0]
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                dataset = row['x']
                semantic = row['semantic']
                records.append({'LLM': llm_name, 'Dataset': dataset, 'Semantic Similarity': semantic})
                dataset_names.add(dataset)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    if not records:
        print("No data to plot.")
        return

    plot_df = pd.DataFrame(records)
    # Sort datasets for consistent color mapping
    dataset_names = sorted(dataset_names)
    palette = sns.color_palette("pastel", n_colors=len(dataset_names))
    dataset_to_color = {ds: palette[i] for i, ds in enumerate(dataset_names)}

    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        data=plot_df,
        x='LLM',
        y='Semantic Similarity',
        hue='Dataset',
        palette=dataset_to_color
    )

    plt.ylabel("Semantic Similarity", fontsize=12)
    plt.xlabel("LLM", fontsize=12)
    plt.title("After Harmonization - Semantic Similarity per DB for Each LLM", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Dataset", loc='upper left')
    plt.tight_layout()

    output_file_path = os.path.join(save_path, output_file_name)
    plt.savefig(output_file_path, dpi=300)
    print(f"Saved grouped bar plot to {output_file_path}")
    

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
    
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parseargs()
    
    if args.plot_different_llms:
        # Plot the semantic similarity between the harmonized schema obtained 
        # with different LLMs vs original dataset 
        # Importantly, the input data argparse element will be a list with the path 
        # to the llm-specific csv file, with the semantic similarity.
        plot_different_llms_semnatic_similarity(
            csv_file_paths=args.input_data, 
            save_path=args.save_path, 
            output_file_name=args.output_file_name
        )
    else:
        # We assume that all the compared LLMs are tested on the same dataset.
        # Therefore, each LLM results folder in the results folder have the same initial_similarity_results.csv
        # Plot initial similarity: no need for the LLM legend, 
        # just choose one random CSV file containing the initial pairwise similarities 
        # between datasets, from one of the LLMs' folders.
        # Importantly, the input data argparse element will not be a list but just the path to a single csv file.
        plot_similarity_graph(
            csv_file_path=args.input_data[0],
            save_path=args.save_path,  
            output_file_name=args.output_file_name
        )