import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

def plot_similarity_graph(csv_file_paths, save_path='./'):
    """
    Plots scatter graphs showing structural vs. semantic similarity for
    pairwise comparisons from multiple CSV files, using different markers for each file
    and different colors for each pair (consistent across files).
    """
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Collect all pairs across all files for consistent coloring
    all_pairs = set()
    dfs = []
    for csv_file_path in csv_file_paths:
        try:
            df = pd.read_csv(csv_file_path)
            df = df.rename(columns={
                'structure': 'Structural Similarity',
                'semantic': 'Semantic Similarity'
            })
            df['Pair'] = df['x'] + ' vs ' + df['y']
            all_pairs.update(df['Pair'].unique())
            dfs.append((csv_file_path, df))
        except Exception as e:
            print(f"Error reading {csv_file_path}: {e}")

    # Assign a color to each pair
    all_pairs = sorted(all_pairs)
    palette = sns.color_palette("pastel", n_colors=len(all_pairs))
    pair_to_color = {pair: palette[i] for i, pair in enumerate(all_pairs)}

    # Plot each file with its marker
    file_handles = []
    pair_handles = {}
    for idx, (csv_file_path, df) in enumerate(dfs):
        marker = MARKERS[idx % len(MARKERS)]
        
        # For file legend (only once per file)
        file_handles.append(
            plt.Line2D([0], [0], marker=marker, color='w',
                        label=csv_file_path.split('/')[0].split('_')[0],
                        markerfacecolor='gray', markersize=12, markeredgecolor='k')
        )
        
        for pair in all_pairs:
            pair_df = df[df['Pair'] == pair]
            if not pair_df.empty:
                sc = ax.scatter(
                    pair_df['Structural Similarity'],
                    pair_df['Semantic Similarity'],
                    label=f"{os.path.basename(csv_file_path)} - {pair}",
                    marker=marker,
                    s=200,
                    alpha=0.8,
                    edgecolor='w',
                    linewidth=1,
                    color=pair_to_color[pair]
                )
                # For pair legend (only once per pair)
                if pair not in pair_handles:
                    pair_handles[pair] = plt.Line2D([0], [0], marker='o', color='w',
                                                    label=pair,
                                                    markerfacecolor=pair_to_color[pair],
                                                    markersize=12, markeredgecolor='k')

    plt.title('Pairwise Structural vs. Semantic Similarity', fontsize=16)
    plt.xlabel('Structural Similarity', fontsize=12)
    plt.ylabel('Semantic Similarity', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    # Two legends: one for files (shapes), one for pairs (colors)
    legend1 = plt.legend(handles=file_handles, title="LLM (Shape)", loc='lower right')
    plt.gca().add_artist(legend1)
    plt.legend(handles=list(pair_handles.values()), title="Dataset (Color)", loc='upper right')
    plt.tight_layout()

    output_file_name = 'different_models_similarity_graph.png'
    output_file_path = os.path.join(save_path, output_file_name)
    plt.savefig(output_file_path, dpi=300)
    print(f"Saved combined plot to {output_file_path}")

def parseargs():
    parser = argparse.ArgumentParser(description="Dataspace and AI: Schema Harmonization")
    parser.add_argument('--input_data', nargs='+', default=['./results.csv'], type=str,
                        help='paths to the results CSV files (space-separated)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()
    plot_similarity_graph(args.input_data)