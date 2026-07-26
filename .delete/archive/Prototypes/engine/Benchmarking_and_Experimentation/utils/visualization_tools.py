"""
Visualization Tools for Benchmarking Framework

This module contains all visualization and analysis functions used in the benchmarking notebook.
Functions here handle dataset analysis, performance visualizations, and result plotting.

Functions:
- analyze_dataset_distribution(): Analyzes and plots class distribution in the dataset
- create_better_confusion_visualizations(): Creates detailed performance analysis charts
- load_results(): Loads previous experiment results from CSV files
- visualize_results(): Creates comparison visualizations for experiment results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


def analyze_dataset_distribution(audio_dir="Audio_data"):
    """
    Analyze the class distribution in the audio dataset.
    
    Args:
        audio_dir (str): Path to the audio dataset directory
        
    Returns:
        dict: Dictionary containing class counts and statistics
    """
    def count_files_per_class(audio_dir):
        class_counts = {}
        for class_name in os.listdir(audio_dir):
            class_path = os.path.join(audio_dir, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len([
                    f for f in os.listdir(class_path) 
                    if os.path.isfile(os.path.join(class_path, f))
                ])
        return class_counts

    # Analyze the dataset
    class_counts = count_files_per_class(audio_dir)

    # Display class distribution
    print("Class Distribution:")
    total_files = sum(class_counts.values())
    print(f"Total classes: {len(class_counts)}")
    print(f"Total files: {total_files}")
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{class_name}: {count} files")
    print("... (showing top 10)")

    # Calculate variance in class distribution
    class_counts_values = list(class_counts.values())
    variance = np.var(class_counts_values)
    mean_files = np.mean(class_counts_values)
    print(f"\nStatistics:")
    print(f"Mean files per class: {mean_files:.2f}")
    print(f"Variance in class distribution: {variance:.2f}")
    print(f"Standard deviation: {np.std(class_counts_values):.2f}")

    # Sort classes by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    # Top 20 classes
    top_20 = sorted_classes[:20]
    # Bottom 20 classes
    bottom_20 = sorted_classes[-20:]

    # Plot top 20 classes
    plt.figure(figsize=(12, 6))
    plt.bar([x[0] for x in top_20], [x[1] for x in top_20], color='green', alpha=0.7)
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Number of Files")
    plt.title("Top 20 Classes by Number of Files")
    plt.tight_layout()
    plt.show()

    # Plot bottom 20 classes
    plt.figure(figsize=(12, 6))
    plt.bar([x[0] for x in bottom_20], [x[1] for x in bottom_20], color='red', alpha=0.7)
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Number of Files")
    plt.title("Bottom 20 Classes by Number of Files")
    plt.tight_layout()
    plt.show()
    
    return {
        'class_counts': class_counts,
        'total_files': total_files,
        'total_classes': len(class_counts),
        'variance': variance,
        'mean_files': mean_files
    }


def create_better_confusion_visualizations(y_true, y_pred, class_names, exp_name, timestamp, models_dir):
    """
    Create comprehensive performance visualizations instead of a massive confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        exp_name: Experiment name
        timestamp: Timestamp for file naming
        models_dir: Directory to save visualizations
        
    Returns:
        tuple: (performance_filename, confusion_filename, confusion_matrix)
    """
    # Generate confusion matrix data
    cm = confusion_matrix(y_true, y_pred)
    
    # 1. Per-class performance chart
    plt.figure(figsize=(15, 10))
    
    # Calculate per-class metrics
    class_report_dict = classification_report(y_true, y_pred, 
                                             labels=np.unique(np.concatenate([y_true, y_pred])),
                                             target_names=[class_names[i] for i in np.unique(np.concatenate([y_true, y_pred]))], 
                                             output_dict=True, zero_division=0)
    
    # Extract per-class metrics
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for class_name, metrics in class_report_dict.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(class_name)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1-score'])
            supports.append(metrics['support'])
    
    # Create subplot for per-class performance
    plt.subplot(2, 2, 1)
    x_pos = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x_pos - width, precisions, width, label='Precision', alpha=0.8)
    plt.bar(x_pos, recalls, width, label='Recall', alpha=0.8)
    plt.bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title(f'Per-Class Performance - {exp_name}')
    plt.xticks(x_pos, classes, rotation=90, fontsize=8)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 2. Top performing classes
    plt.subplot(2, 2, 2)
    sorted_indices = np.argsort(f1_scores)[::-1][:10]  # Top 10
    top_classes = [classes[i] for i in sorted_indices]
    top_f1_scores = [f1_scores[i] for i in sorted_indices]
    
    plt.barh(range(len(top_classes)), top_f1_scores, color='green', alpha=0.7)
    plt.xlabel('F1-Score')
    plt.title('Top 10 Performing Classes')
    plt.yticks(range(len(top_classes)), top_classes)
    plt.grid(axis='x', alpha=0.3)
    
    # 3. Worst performing classes
    plt.subplot(2, 2, 3)
    worst_indices = np.argsort(f1_scores)[:10]  # Bottom 10
    worst_classes = [classes[i] for i in worst_indices]
    worst_f1_scores = [f1_scores[i] for i in worst_indices]
    
    plt.barh(range(len(worst_classes)), worst_f1_scores, color='red', alpha=0.7)
    plt.xlabel('F1-Score')
    plt.title('Bottom 10 Performing Classes')
    plt.yticks(range(len(worst_classes)), worst_classes)
    plt.grid(axis='x', alpha=0.3)
    
    # 4. Class distribution in test set
    plt.subplot(2, 2, 4)
    unique, counts = np.unique(y_true, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1][:15]  # Top 15 most frequent
    top_classes_freq = [class_names[unique[i]] for i in sorted_indices]
    top_counts = [counts[i] for i in sorted_indices]
    
    plt.bar(range(len(top_classes_freq)), top_counts, alpha=0.7)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Top 15 Most Frequent Classes in Test Set')
    plt.xticks(range(len(top_classes_freq)), top_classes_freq, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive performance chart
    perf_filename = f"{exp_name}_{timestamp}_performance_analysis.png"
    perf_path = os.path.join(models_dir, perf_filename)
    plt.savefig(perf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate figure for most common confusions
    plt.figure(figsize=(12, 8))
    
    # Find top confusions (off-diagonal elements)
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # Remove diagonal (correct predictions)
    
    # Get top 20 confusions
    top_confusions = []
    for i in range(cm_copy.shape[0]):
        for j in range(cm_copy.shape[1]):
            if cm_copy[i, j] > 0:
                unique_classes = np.unique(np.concatenate([y_true, y_pred]))
                true_class = class_names[unique_classes[i]]
                pred_class = class_names[unique_classes[j]]
                count = cm_copy[i, j]
                top_confusions.append((true_class, pred_class, count))
    
    # Sort by confusion count and take top 20
    top_confusions.sort(key=lambda x: x[2], reverse=True)
    top_confusions = top_confusions[:20]
    
    if top_confusions:
        confusion_labels = [f"{tc[0]} → {tc[1]}" for tc in top_confusions]
        confusion_counts = [tc[2] for tc in top_confusions]
        
        plt.barh(range(len(confusion_labels)), confusion_counts, alpha=0.7, color='orange')
        plt.xlabel('Number of Confusions')
        plt.title(f'Top 20 Most Common Confusions - {exp_name}')
        plt.yticks(range(len(confusion_labels)), confusion_labels)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, count in enumerate(confusion_counts):
            plt.text(count + 0.1, i, str(count), va='center')
    
    plt.tight_layout()
    
    # Save the confusion analysis chart
    conf_filename = f"{exp_name}_{timestamp}_confusion_analysis.png"
    conf_path = os.path.join(models_dir, conf_filename)
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return perf_filename, conf_filename, cm


def load_results(output_dir):
    """
    Load previous experiment results from metadata.json files in output/models directory.
    
    Args:
        output_dir (str): Directory containing result files
        
    Returns:
        pandas.DataFrame or None: Results dataframe if found, None otherwise
    """
    import pandas as pd
    import json
    from IPython.display import HTML
    
    # Check if results directory exists
    models_dir = os.path.join(output_dir, "models")
    if not os.path.exists(models_dir):
        print(f"Models directory does not exist: {models_dir}")
        return None
    
    # Look for metadata.json files
    metadata_files = [f for f in os.listdir(models_dir) if f.endswith("_metadata.json")]
    
    if not metadata_files:
        print("No metadata files found. Run experiments first.")
        return None
    
    # Load all metadata files
    results_data = []
    for metadata_file in metadata_files:
        metadata_path = os.path.join(models_dir, metadata_file)
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create links to PNG files if they exist
            perf_analysis_link = ""
            confusion_analysis_link = ""
            
            if "performance_analysis_file" in metadata:
                perf_file = metadata["performance_analysis_file"]
                perf_path = os.path.join(models_dir, perf_file)
                if os.path.exists(perf_path):
                    perf_analysis_link = f'<a href="{perf_path}" target="_blank">Performance Analysis</a>'
            
            if "confusion_analysis_file" in metadata:
                conf_file = metadata["confusion_analysis_file"]
                conf_path = os.path.join(models_dir, conf_file)
                if os.path.exists(conf_path):
                    confusion_analysis_link = f'<a href="{conf_path}" target="_blank">Confusion Analysis</a>'
            
            # Extract key metrics
            result_row = {
                "Experiment": metadata.get("experiment_name", "Unknown"),
                "Model": metadata.get("model_architecture", "Unknown"),
                "Accuracy": f"{metadata.get('test_accuracy', 0.0):.4f}",
                "Precision": f"{metadata.get('precision_macro', 0.0):.4f}",
                "Recall": f"{metadata.get('recall_macro', 0.0):.4f}",
                "F1 Score": f"{metadata.get('f1_score_macro', 0.0):.4f}",
                "Training Time (min)": f"{metadata.get('training_time_minutes', 0.0):.1f}",
                "Audio Aug": metadata.get("audio_augmentation", "none"),
                "Image Aug": metadata.get("image_augmentation", "none"),
                "Performance Analysis": perf_analysis_link,
                "Confusion Analysis": confusion_analysis_link,
                "Timestamp": metadata.get("timestamp", "Unknown")
            }
            
            results_data.append(result_row)
            
        except Exception as e:
            print(f"Error loading {metadata_file}: {e}")
    
    if not results_data:
        print("No valid metadata found.")
        return None
    
    # Create DataFrame and sort by timestamp (most recent first)
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values("Timestamp", ascending=False)
    
    print(f"Loaded {len(results_df)} experiment results from metadata files")
    return results_df


def visualize_results(results_df):
    """
    Create various visualizations to compare experiment results.
    
    Args:
        results_df (pandas.DataFrame): Results dataframe from load_results()
    """
    import pandas as pd
    import seaborn as sns
    
    if results_df is None or len(results_df) == 0:
        print("No results available to visualize.")
        return
    
    # Set the figure size for better visibility
    plt.figure(figsize=(14, 8))
    
    # Create accuracy comparison bar chart
    plt.subplot(2, 2, 1)
    sns.barplot(x='Experiment', y='Test Accuracy', data=results_df)
    plt.title('Test Accuracy by Experiment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Create F1 score comparison bar chart
    plt.subplot(2, 2, 2)
    sns.barplot(x='Experiment', y='F1 Score', data=results_df)
    plt.title('F1 Score by Experiment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Training time comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x='Experiment', y='Training Time (min)', data=results_df)
    plt.title('Training Time by Experiment (minutes)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Model comparison
    plt.subplot(2, 2, 4)
    model_comparison = results_df.groupby('Model')['Test Accuracy'].mean().reset_index()
    sns.barplot(x='Model', y='Test Accuracy', data=model_comparison)
    plt.title('Average Accuracy by Model')
    plt.tight_layout()
    
    plt.tight_layout(pad=3.0)
    plt.show()
