import pandas as pd
import os
from PIL import Image

def clean_metadata(csv_path, image_dir, output_csv_path):
    """
    Loads metadata, removes duplicates, drops rows with missing key values,
    and validates that the corresponding image file actually exists on disk.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Metadata file {csv_path} not found.")
        return None
        
    df = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
    
    # Drop rows missing critical metrics
    df = df.dropna(subset=['id', 'category', 'gender'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['id'])
    
    # Cast ID to int for standard formatting
    df['id'] = df['id'].astype(int)
    
    valid_rows = []
    print("Validating image paths...")
    for idx, row in df.iterrows():
        img_name = f"{row['id']}.jpg"
        img_path = os.path.join(image_dir, img_name)
        
        # Verify file exists and is a valid image
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_rows.append(row)
            except Exception:
                # Corrupted image file
                continue
                
    cleaned_df = pd.DataFrame(valid_rows)
    cleaned_df.to_csv(output_csv_path, index=False)
    print(f"Data cleaning complete. Saved {len(cleaned_df)} valid entries to {output_csv_path}")
    return cleaned_df
