#!/bin/bash
# make sure that directory exists if not then create it
mkdir -p ./data/aircrafts

# download the dataset from kaggle
kaggle datasets download a2015003713/militaryaircraftdetectiondataset -p ./data/aircraft --unzip
