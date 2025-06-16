import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to plot the similarity graph from a CSV file
def plot_similarity_graph(csv_file_path, save_path='./'):
    """
    Plots a scatter graph showing structural vs. semantic similarity for
    pairwise comparisons, reading data from a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file containing the data.
                             Expected columns: 'structure', 'semantic', 'x', 'y'.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    # Rename columns for consistent plotting and labeling
    df = df.rename(columns={
        'structure': 'Structural Similarity',
        'semantic': 'Semantic Similarity'
    })

    # Create the 'Pair' column from 'x' and 'y' for labeling and hue
    df['Pair'] = df['x'] + ' vs ' + df['y']

    # Get unique elements for the title
    all_elements = pd.concat([df['x'], df['y']]).unique()
    num_elements = len(all_elements)

    # Set the pastel color palette
    sns.set_palette("pastel")

    plt.figure(figsize=(12, 10)) # Set figure size for better readability

    # Create the scatter plot using seaborn
    ax = sns.scatterplot(
        data=df,
        x='Structural Similarity',
        y='Semantic Similarity',
        hue='Pair',       # Differentiate points by pair
        s=200,            # Adjust point size
        alpha=0.8,        # Adjust transparency
        edgecolor='w',    # Add white edge to points
        linewidth=1
    )

    # --- Labels on points have been removed as per your request ---
    # The loop for adding ax.text() is no longer present here.

    # Set plot title and labels
    plt.title(f'Pairwise Structural vs. Semantic Similarity for {num_elements} Elements', fontsize=16)
    plt.xlabel('Structural Similarity', fontsize=12)
    plt.ylabel('Semantic Similarity', fontsize=12)

    # Set limits for axes to ensure they start at 0 and go up to 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for better readability
    plt.legend(loc='upper left') # Move legend outside
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    # Save the figure as a PNG file
    # Build the output name file path by removing the '.csv' extension and adding '.png'
    output_file_name = os.path.splitext(os.path.basename(csv_file_path))[0] + '_similarity_graph.png'
    output_file_path = os.path.join(save_path, output_file_name)
    plt.savefig(output_file_path, dpi=300) # Save the figure as a PNG file
    plt.show()
    

def parseargs():
    parser = argparse.ArgumentParser(description="Dataspace and AI: Schema Harmonization")
    
    parser.add_argument('--input_data', default='./results.csv', type=str, help='path to the results CSV file')
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    # Parse command line arguments
    args = parseargs()
    
    # You can change the CSV file path here if your file is named differently or located elsewhere.
    plot_similarity_graph(args.input_data)