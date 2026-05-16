"""
Notebook Interface Components for Benchmarking Framework

This module contains all widget creation and UI setup functions for the notebook.
Functions here handle the interactive interface components and layout.

Functions:
- create_directory_widgets(): Creates widgets for directory configuration
- create_experiment_widgets(): Creates experiment selection widgets
- setup_experiment_interface(): Sets up the complete interactive interface
"""

from ipywidgets import widgets
from IPython.display import display


def create_directory_widgets(data_dir, cache_dir, output_dir):
    """
    Create widgets for directory configuration.
    
    Args:
        data_dir (str): Default data directory path
        cache_dir (str): Default cache directory path
        output_dir (str): Default output directory path
        
    Returns:
        dict: Dictionary containing directory widgets
    """
    data_dir_widget = widgets.Text(
        value=data_dir,
        description='Data Directory:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    cache_dir_widget = widgets.Text(
        value=cache_dir,
        description='Cache Directory:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    output_dir_widget = widgets.Text(
        value=output_dir,
        description='Output Directory:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    # Group directory widgets
    dir_widgets_box = widgets.VBox([data_dir_widget, cache_dir_widget, 
                                   output_dir_widget])

    return {
        'data_dir_widget': data_dir_widget,
        'cache_dir_widget': cache_dir_widget,
        'output_dir_widget': output_dir_widget,
        'dir_widgets_box': dir_widgets_box
    }


def create_experiment_widgets(experiments):
    """
    Create widgets for experiment selection.
    
    Args:
        experiments (list): List of experiment configurations
        
    Returns:
        dict: Dictionary containing experiment widgets
    """
    # Create widget for experiment selection
    experiment_options = [(exp["name"], exp["name"]) for exp in experiments]
    experiment_widget = widgets.SelectMultiple(
        options=experiment_options,
        description='Select Experiments:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%', height='200px')
    )

    return {
        'experiment_widget': experiment_widget,
        'experiment_options': experiment_options
    }


def create_action_buttons():
    """
    Create action buttons for experiment control.
    
    Returns:
        dict: Dictionary containing action buttons and layout
    """
    # Buttons for actions
    run_selected_button = widgets.Button(
        description='Run Selected Experiments',
        button_style='primary',
        tooltip='Run the selected experiments'
    )

    run_all_button = widgets.Button(
        description='Run All Experiments',
        tooltip='Run all experiments'
    )

    generate_report_button = widgets.Button(
        description='Generate Report Only',
        button_style='info',
        tooltip='Generate a report from existing results'
    )

    # Group buttons
    buttons_box = widgets.HBox([run_selected_button, run_all_button, 
                               generate_report_button])

    return {
        'run_selected_button': run_selected_button,
        'run_all_button': run_all_button,
        'generate_report_button': generate_report_button,
        'buttons_box': buttons_box
    }


def setup_experiment_interface(data_dir, cache_dir, output_dir, experiments):
    """
    Set up the complete interactive interface for experiments.
    
    Args:
        data_dir (str): Default data directory path
        cache_dir (str): Default cache directory path
        output_dir (str): Default output directory path
        experiments (list): List of experiment configurations
        
    Returns:
        dict: Dictionary containing all widgets and interface components
    """
    # Create all widget components
    dir_widgets = create_directory_widgets(data_dir, cache_dir, output_dir)
    exp_widgets = create_experiment_widgets(experiments)
    action_widgets = create_action_buttons()

    # Output area for logs
    output_area = widgets.Output(
        layout={'border': '1px solid black', 'width': '90%', 'height': '300px'}
    )

    # Main container for all control widgets
    controls_box = widgets.VBox([
        widgets.HTML("<h3>Directory Configuration:</h3>"),
        dir_widgets['dir_widgets_box'],
        widgets.HTML("<hr><h3>Experiment Selection:</h3>"),
        exp_widgets['experiment_widget'],
        widgets.HTML("<hr><h3>Actions:</h3>"),
        action_widgets['buttons_box']
    ])

    # Combine all components
    interface_components = {
        **dir_widgets,
        **exp_widgets,
        **action_widgets,
        'output_area': output_area,
        'controls_box': controls_box
    }

    return interface_components


def display_experiment_interface(interface_components):
    """
    Display the experiment interface components.
    
    Args:
        interface_components (dict): Interface components from 
                                   setup_experiment_interface()
    """
    display(interface_components['controls_box'])
    display(interface_components['output_area'])
