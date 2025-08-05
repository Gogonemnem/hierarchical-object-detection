# Hierarchical Object Detection

This repository contains tools and scripts for hierarchical object detection using advanced deep learning models. The project is structured to facilitate dataset preparation, model training, evaluation, and analysis.

## Table of Contents
- [Hierarchical Object Detection](#hierarchical-object-detection)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
    - [Step 1: Download the Dataset](#step-1-download-the-dataset)
    - [Step 2: Convert the Dataset](#step-2-convert-the-dataset)
  - [Training and Evaluation](#training-and-evaluation)
  - [Contributing](#contributing)
  - [License](#license)

## Overview
Hierarchical object detection involves detecting objects while considering their hierarchical relationships. This repository provides tools to:
- Download datasets using the Kaggle API.
- Convert datasets to include hierarchical relationships.
- Train and evaluate models using MMDetection.

## Setup

### Prerequisites
- Docker and Docker Compose
- Kaggle API credentials

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/hierarchical-object-detection.git
   cd hierarchical-object-detection
   ```
2. Start Docker Compose:
   ```bash
   docker-compose build
   docker-compose up
   ```
3. Access the container:
   ```bash
   docker exec -it <container-name> bash
   ```

## Dataset Preparation

### Step 1: Download the Dataset
1. Ensure you have the Kaggle API key. If not, follow [these instructions](https://www.kaggle.com/docs/api) to generate one.
2. Export your Kaggle API key as an environment variable:
   ```bash
   export KAGGLE_USERNAME=<your-username>
   export KAGGLE_KEY=<your-api-key>
   ```
3. Use the dataset downloader script to fetch the dataset:
   ```bash
   bash tools/misc/download_aircraft_dataset.sh
   ```

### Step 2: Convert the Dataset
1. Use the dataset converter to add hierarchical relationships. The converter supports two taxonomies:
   - **Functional Taxonomy**: Groups aircraft based on their function (e.g., Fighters, Bombers, UAVs).
   - **Area Taxonomy**: Groups aircraft based on their region of origin (e.g., US Aircraft, Russian Aircraft).

   Example command:
   ```bash
   python tools/dataset_converters/aircraft.py --csv-file data/aircraft/labels_with_split.csv --out-dir data/aircraft
   ```

2. The dataset splits (e.g., `train`, `val`, `test`) are already defined in the CSV file. The converter will automatically generate COCO JSON files for each split.

3. The output will include:
   - `aircraft_train.json`
   - `aircraft_val.json`
   - `aircraft_test.json`
   - Hierarchical versions:
     - `aircraft_hierarchy_function_train.json`
     - `aircraft_hierarchy_function_val.json`
     - `aircraft_hierarchy_function_test.json`
     - `aircraft_hierarchy_area_train.json`
     - `aircraft_hierarchy_area_val.json`
     - `aircraft_hierarchy_area_test.json`

## Training and Evaluation

1. Train a model using the provided training scripts:
   ```bash
   bash tools/train.sh configs/hod/<config-file>.py
   ```
2. Evaluate the model:
   ```bash
   bash tools/test.sh configs/hod/<config-file>.py <checkpoint-file>
   ```

For more details about training and evaluation, refer to the [MMDetection documentation](https://github.com/open-mmlab/mmdetection).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
