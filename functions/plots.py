import matplotlib.pyplot as plt
import numpy as np

def plot_pie(ax, class_counts, labels, colors, explode, title):
    """
    Plot a pie chart to visualize class distribution or proportions.

    Parameters:
        ax: Matplotlib axis object on which the pie chart will be plotted.
        class_counts: A list or array containing the values representing the size of each class or category.
        labels: A list of strings representing the names of each class or category.
        colors: A list of colors to use for each segment of the pie chart.
        explode: A list indicating how much each segment should be "exploded" (separated from the center). 
                Use 0 for no separation and values > 0 for increased separation.
        title: Title of the pie chart.

    Functionality:
        - Creates a pie chart with the specified class counts, labels, and colors.
    """

    wedges, texts, autotexts = ax.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode, textprops={'fontsize': 12})
    
    # Customize the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)

    ax.set_title(title, fontsize=14)

def plot_metrics(df, color_map):
    """
    Plot the metrics for the model as a function of the balancing method.

    Parameters:
        df: the DataFrame containing the performance metrics for the model.
        color_map: a list of colors to use for plotting the metrics.

    Returns:
        - Displays two separate plots:
          1. A figure with two subplots: Accuracy and ROC-AUC as bar plots.
          2. A bar plot with three subplots: Precision, Recall, and F1 scores.
    """

    # Extract metrics for plotting
    metrics = ['Accuracy', 'ROC-AUC']

    # Computing the number of classes
    num_classes = len([col for col in df.columns if 'Class' in col and 'Precision' in col])

    # Define metrics based on the number of classes
    precision_metrics = [f'Precision Class {i}' for i in range(num_classes)]
    recall_metrics = [f'Recall Class {i}' for i in range(num_classes)]
    f1_metrics = [f'F1 Class {i}' for i in range(num_classes)]

    x_labels = df['Balancing']

    # Figure 1: Accuracy and ROC-AUC as bar plots
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    fig1.tight_layout(pad=7.0)

    bar_width = 0.35
    x = np.arange(len(x_labels))

    for i, (metric, ax) in enumerate(zip(metrics, axes1)):
        ax.bar(x, df[metric], bar_width, color=color_map[i], label=metric)
        
        # Add the values on top of each bar
        for idx, value in enumerate(df[metric]):
            ax.text(x[idx], value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(metric if metric == 'Accuracy' else 'ROC-AUC Score', fontsize=12, pad=15)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Balancing Strategies', fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        ax.tick_params(axis='x', rotation = 45)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, 1.1)

    plt.show()

    # Figure 2: Bar plots for Precision, Recall, and F1
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12), sharey=True)
    fig2.tight_layout(h_pad=15.0)
    
    metric_groups = [precision_metrics, recall_metrics, f1_metrics]
    metric_titles = ['Precision', 'Recall', 'F1']
    
    bar_width = 0.3

    for i, (metric_group, ax) in enumerate(zip(metric_groups, axes2)):
        for j, metric in enumerate(metric_group):
            bars = ax.bar(x + (j - (num_classes - 1) / 2) * bar_width, df[metric], bar_width, color=color_map[j+2], label=f"Class {j}" if i == 0 else None)
            
            # Add the values on top of each bar
            for bar in bars:
                yval = bar.get_height()
                ax.text( bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

        # Set x-axis and title for each subplot
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_title(f'{metric_titles[i]} - Class-wise', fontsize=15)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Balancing Techniques', fontsize=12)
        ax.grid(alpha = 0.3)
        ax.legend(labels=[f"Class {i}" for i in range(num_classes)], loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 0.5))

    plt.ylim(0, 1.1)
    plt.subplots_adjust(right=0.95, hspace=0.3)
    plt.show()