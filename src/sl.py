
import os
import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\tagged_boards"
OUTPUT_DIR = f"{Assignment3Dir}\\documentation\\data"
PROCESSED_DIR = f"{Assignment3Dir}\\documentation\\processed_data"


def read_processed_data():
    num_tiles = []
    blanks = []
    manhattan_h_val = []
    euclidean_h_val = []
    displaced_tiles = []
    effort = []

    with open(f"{PROCESSED_DIR}\\blanks.csv", "r") as blanks_file:
        csv_reader = csv.reader(blanks_file, delimiter=",")
        for row in csv_reader:
            blanks.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\displaced_tiles.csv", "r") as displaced_tiles_file:
        csv_reader = csv.reader(displaced_tiles_file, delimiter=",")
        for row in csv_reader:
            displaced_tiles.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\effort.csv", "r") as effort_file:
        csv_reader = csv.reader(effort_file, delimiter=",")
        for row in csv_reader:
            effort.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\euclidean_h_val.csv", "r") as euclidean_h_val_file:
        csv_reader = csv.reader(euclidean_h_val_file, delimiter=",")
        for row in csv_reader:
            euclidean_h_val.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\manhattan_h_val.csv", "r") as manhattan_h_val_file:
        csv_reader = csv.reader(manhattan_h_val_file, delimiter=",")
        for row in csv_reader:
            manhattan_h_val.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\num_tiles.csv", "r") as num_tiles_file:
        csv_reader = csv.reader(num_tiles_file, delimiter=",")
        for row in csv_reader:
            num_tiles.append(float(row[0]))

    return num_tiles, blanks, manhattan_h_val, euclidean_h_val, displaced_tiles, effort


num_tiles, blanks, manhattan_h_val, euclidean_h_val, displaced_tiles, effort = read_processed_data()
print(f"Num Tiles: {num_tiles[-1]}")
print(f"Blanks: {blanks[-1]}")
print(f"Manhattan H Val: {manhattan_h_val[-1]}")
print(f"Euclidean H Val: {euclidean_h_val[-1]}")
print(f"Displaced Tiles: {displaced_tiles[-1]}")
print(f"Effort: {effort[-1]}")


