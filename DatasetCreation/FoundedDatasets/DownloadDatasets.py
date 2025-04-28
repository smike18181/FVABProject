import kagglehub
import os

# Download latest version "gtsrb-german-traffic-sign"
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print("Path to dataset files:", path)

# Download latest version "lisa-traffic-light-dataset"
path2 = kagglehub.dataset_download("mbornoe/lisa-traffic-light-dataset")
print("Path to dataset files:", path2)