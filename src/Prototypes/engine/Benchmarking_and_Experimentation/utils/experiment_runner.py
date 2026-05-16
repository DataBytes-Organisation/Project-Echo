"""
Experiment Runner for Benchmarking Framework

This module contains the core experiment execution logic.
Functions here handle running experiments, training models, and saving results.

Functions:
- run_experiments_pipeline(): Main function to run selected experiments
- setup_experiment_callback(): Sets up the callback function for the run button
"""

import os
import time
import datetime
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def run_experiments_pipeline(interface_components, experiments):
    """
    Set up and bind the experiment runner to the interface.
    
    Args:
        interface_components (dict): Interface components with widgets
        experiments (list): List of available experiments
    """
    def run_selected_experiments(button):
        """Main experiment execution function."""
        output_area = interface_components['output_area']
        data_dir_widget = interface_components['data_dir_widget']
        cache_dir_widget = interface_components['cache_dir_widget']
        output_dir_widget = interface_components['output_dir_widget']
        experiment_widget = interface_components['experiment_widget']
        
        output_area.clear_output(wait=True)
        with output_area:
            print("🚀 Starting experiment run...")
            print("=" * 50)

            # Get new directory paths from widgets
            new_data_dir = data_dir_widget.value
            new_cache_dir = cache_dir_widget.value
            new_output_dir = output_dir_widget.value

            # Update system configuration
            from config.system_config import SC
            SC['AUDIO_DATA_DIRECTORY'] = new_data_dir
            SC['CACHE_DIRECTORY'] = new_cache_dir
            SC['OUTPUT_DIRECTORY'] = new_output_dir
            
            print("✅ System configuration updated:")
            print(f"   📁 Data Dir: {SC['AUDIO_DATA_DIRECTORY']}")
            print(f"   💾 Cache Dir: {SC['CACHE_DIRECTORY']}")
            print(f"   📊 Output Dir: {SC['OUTPUT_DIRECTORY']}")

            # Ensure directories exist
            for path in [new_data_dir, new_cache_dir, new_output_dir]:
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                    print(f"   ➕ Created directory: {path}")

            # Ensure models directory exists
            models_dir = os.path.join(new_output_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            print(f"   🤖 Models will be saved to: {models_dir}")

            # Get selected experiments
            selected_experiments = list(experiment_widget.value)
            if not selected_experiments:
                print("❌ No experiment selected. Please select at least one.")
                return

            print(f"\n🎯 Running {len(selected_experiments)} experiment(s):")
            for exp_name in selected_experiments:
                print(f"   • {exp_name}")
            print("\n" + "=" * 50)

            # Run each selected experiment
            for i, exp_name in enumerate(selected_experiments, 1):
                exp_config = next((exp for exp in experiments 
                                 if exp["name"] == exp_name), None)
                if exp_config is None:
                    print(f"❌ Experiment {exp_name} not found.")
                    continue

                print(f"\n🔬 [{i}/{len(selected_experiments)}] Running: "
                      f"{exp_config['name']}")
                print(f"   Model: {exp_config['model']}")
                print(f"   Epochs: {exp_config.get('epochs', 'default')}")
                print(f"   Batch Size: {exp_config.get('batch_size', 'default')}")
                
                try:
                    success = _run_single_experiment(exp_config, models_dir, SC)
                    if success:
                        print(f"✅ Experiment {exp_config['name']} completed!")
                    else:
                        print(f"❌ Experiment {exp_config['name']} failed!")
                        
                except Exception as e:
                    print(f"💥 Error in experiment {exp_config['name']}: {str(e)}")
                    import traceback
                    traceback.print_exc()

                print("-" * 40)

            print(f"\n🎉 All experiments completed!")
            print("=" * 50)

    # Bind the function to the run button
    interface_components['run_selected_button'].on_click(run_selected_experiments)


def _run_single_experiment(exp_config, models_dir, system_config):
    """
    Run a single experiment configuration.
    
    Args:
        exp_config (dict): Experiment configuration
        models_dir (str): Directory to save models
        system_config (dict): System configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import training function
        from utils.optimised_engine_pipeline import train_model
        from utils.visualization_tools import create_better_confusion_visualizations
        
        # Generate timestamp for this experiment run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Record training start time
        training_start_time = time.time()
        print(f"   ⏰ Training started at {datetime.datetime.now()}")
        
        # Train the model
        model, history = train_model(
            model_name=exp_config['model'],
            epochs=exp_config.get('epochs'),
            batch_size=exp_config.get('batch_size')
        )
        
        # Calculate training time
        training_end_time = time.time()
        training_time_seconds = training_end_time - training_start_time
        training_time_minutes = training_time_seconds / 60
        
        print(f"   ✅ Training completed!")
        print(f"   ⏱️  Training time: {training_time_minutes:.2f} minutes")
        
        if model:
            # Save the model
            model_filename = (f"{exp_config['name']}_{exp_config['model']}_"
                            f"{timestamp}_best.h5")
            model_path = os.path.join(models_dir, model_filename)
            model.save(model_path)
            print(f"   💾 Model saved: {model_filename}")
            
            # Try to evaluate the model
            try:
                _evaluate_and_save_results(model, exp_config, timestamp, 
                                         models_dir, system_config,
                                         training_time_seconds, 
                                         training_time_minutes, model_filename)
            except Exception as eval_error:
                print(f"   ⚠️  Evaluation failed: {str(eval_error)}")
                return True  # Training succeeded even if evaluation failed
            
            return True
        else:
            print(f"   ❌ Model training failed - no model returned")
            return False
            
    except Exception as e:
        print(f"   💥 Training failed: {str(e)}")
        return False


def _evaluate_and_save_results(model, exp_config, timestamp, models_dir, 
                              system_config, training_time_seconds, 
                              training_time_minutes, model_filename):
    """
    Evaluate model and save comprehensive results.
    
    Args:
        model: Trained model
        exp_config: Experiment configuration
        timestamp: Timestamp string
        models_dir: Directory for saving results
        system_config: System configuration
        training_time_seconds: Training duration in seconds
        training_time_minutes: Training duration in minutes
        model_filename: Saved model filename
    """
    print(f"   📊 Evaluating model performance...")
    
    # Import data pipeline functions
    from utils.data_pipeline import create_datasets, build_datasets
    from utils.visualization_tools import create_better_confusion_visualizations
    
    # Create test dataset
    train_ds_init, val_ds_init, test_ds_init, class_names_from_data = \
        create_datasets(system_config['AUDIO_DATA_DIRECTORY'])
    
    _, _, test_dataset = build_datasets(
        train_ds_init, val_ds_init, test_ds_init, 
        class_names_from_data, model_name=exp_config['model']
    )
    
    # Record testing start time
    testing_start_time = time.time()
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    
    # Get predictions for detailed metrics
    y_true = []
    y_pred = []
    
    for batch_images, batch_labels in test_dataset:
        # Get true labels
        true_labels = np.argmax(batch_labels.numpy(), axis=1)
        y_true.extend(true_labels)
        
        # Get predictions
        predictions = model.predict(batch_images, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        y_pred.extend(pred_labels)
    
    # Calculate testing time
    testing_end_time = time.time()
    testing_time_seconds = testing_end_time - testing_start_time
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', 
                                    zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', 
                              zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Create visualizations
    perf_filename, conf_filename, cm = create_better_confusion_visualizations(
        y_true, y_pred, class_names_from_data, exp_config['name'], 
        timestamp, models_dir
    )
    
    # Generate classification report
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    class_report = classification_report(
        y_true, y_pred, 
        labels=unique_classes,
        target_names=[class_names_from_data[i] for i in unique_classes], 
        output_dict=True, 
        zero_division=0
    )
    
    print(f"   📈 Performance Metrics:")
    print(f"      • Test Accuracy: {test_accuracy:.4f}")
    print(f"      • F1 Score (macro): {f1_macro:.4f}")
    print(f"      • Precision (macro): {precision_macro:.4f}")
    print(f"      • Recall (macro): {recall_macro:.4f}")
    print(f"   ⏱️  Testing time: {testing_time_seconds:.2f} seconds")
    
    # Save comprehensive metadata
    metadata = {
        "experiment_name": exp_config['name'],
        "model_architecture": exp_config['model'],
        "epochs": exp_config.get('epochs'),
        "batch_size": exp_config.get('batch_size'),
        "audio_augmentation": exp_config.get('audio_augmentation'),
        "image_augmentation": exp_config.get('image_augmentation'),
        "timestamp": timestamp,
        "model_file": model_filename,
        "num_classes": len(class_names_from_data),
        "num_classes_in_predictions": len(unique_classes),
        
        # Performance metrics
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_score_macro": float(f1_macro),
        
        # Timing metrics
        "training_time_seconds": float(training_time_seconds),
        "training_time_minutes": float(training_time_minutes),
        "testing_time_seconds": float(testing_time_seconds),
        
        # Additional files
        "performance_analysis_file": perf_filename,
        "confusion_analysis_file": conf_filename,
        "confusion_matrix_shape": cm.shape,
        "classification_report": class_report
    }
    
    # Save metadata file
    metadata_filename = (f"{exp_config['name']}_{exp_config['model']}_"
                        f"{timestamp}_metadata.json")
    metadata_path = os.path.join(models_dir, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   💾 Saved metadata: {metadata_filename}")
    print(f"   📊 Saved visualizations: {perf_filename}, {conf_filename}")
