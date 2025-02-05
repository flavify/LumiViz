import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Define dataset path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMG_DIR = BASE_DIR / "data" / "images" / "train"
META_PATH = BASE_DIR / "data" / "metadata" / "ISIC_2020_Training_GroundTruth.csv"

class	DataLoader:
  "Handles loading images and metadata"

  def __init__(self, img_dir: Path = IMG_DIR, meta_path: Path = META_PATH):
    self.img_dir = img_dir
    self.meta_path = meta_path

    self.metadata = self.load_metadata()

  def load_metadata(self, raise_error=True):
    """Loads metadata from CSV into a pandas DataFrame."""
    
    if not self.meta_path.exists():
      message = f"❌ Metadata file not found at {self.meta_path}"
      if raise_error:
        raise FileNotFoundError(message)
      print(message)
      return None

    try:
      df = pd.read_csv(self.meta_path)
      if df.empty:
        message = f"⚠ Metadata file at {self.meta_path} is empty"
        if raise_error:
          raise ValueError(message)
        print(message)
        return None	

      print(f"✅ Loaded metadata with {len(df)} samples, {df.shape[1]} columns")
      return df

    except pd.errors.ParserError:
      raise ValueError(f"❌ Error: Metadata file at {self.meta_path} is corrupted or malformed")
    except Exception as e:
      raise RuntimeError(f"❌ Unexpected error loading metadata: {e}")

  def list_images(self):
    """Lists all image files in the dataset folder."""

    images = list(self.img_dir.glob("*.jpg"))
    print(f"📸 Found {len(images)} images in dataset")
    return images


  def load_image(self, image_path: Path):
    pass

  def get_image_label(self, image_name: str):
    pass

  def split_dataset(self, train_ratio: 0.8):
    pass


if __name__ == "__main__":
  # Initialize DataLoader
  loader = DataLoader()

  # List images
  images = loader.list_images()

  # Display a sample image
  if images:
    print(f"Displaying sample image: {images[0]}")
    loader.load_image(images[0])

  # Get label for the first image
  sample_image = os.path.basename(str(images[0]))
  label = loader.get_image_label(sample_image)
  print(f"Label for {sample_image}: {label}")

  # Split dataset
  train_data, val_data = loader.split_dataset()