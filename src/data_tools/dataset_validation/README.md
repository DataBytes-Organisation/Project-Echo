## Overview

This folder contains the pipelines, notebooks, and documentation used to assess and improve dataset quality before the data is used for analysis or downstream processes.

The data QA (validation) pipeline focuses on assessing data accuracy, completeness, consistency, integrity, and overall dataset quality. The validation pipeline is designed to be reproducible and reusable across datasets and projects where the same quality checks are required.

The data cleaning pipeline follows a similar workflow by identifying and addressing data-quality issues such as invalid, unreadable, duplicated, or inconsistent records. The cleaning process is also reproducible and reusable, while the specific cleaning rules and transformations are tailored to the Project Echo dataset and its requirements.

## Pipeline Notebooks
data_qa — Contains the dataset quality assurance and validation pipeline.

Performs automated data-quality checks.
Generates validation reports.
Designed to be reproducible and reusable.

data_cleaning — Contains the dataset cleaning pipeline.

Identifies and addresses data-quality issues.
Applies cleaning and transformation steps required for the dataset.
Designed to be reproducible and reusable, with cleaning logic tailored specifically to the Project Echo dataset.

Both pipelines are notebook-based and can be rerun to reproduce the generated outputs and reports.

