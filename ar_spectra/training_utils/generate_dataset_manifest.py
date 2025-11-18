import os
import json

def generate_manifest(base_path, output_file):
    partitions = ['train', 'eval', 'test']
    manifest = {}

    print(f"Generating manifest from base path: {base_path}")

    for partition in partitions:
        dir_path = os.path.join(base_path, partition)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist. Skipping.")
            manifest[partition] = []
            continue
        
        print(f"Scanning {dir_path}...")
        # List all files in the directory
        try:
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            
            # Sort for reproducibility
            files.sort()
            
            manifest[partition] = files
            print(f"Found {len(files)} files in {partition}.")
        except Exception as e:
            print(f"Error scanning {dir_path}: {e}")
            manifest[partition] = []

    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    print(f"Manifest saved to {output_file}")

if __name__ == "__main__":
    # Path specificato dall'utente
    DEFAULT_DATA_PATH = "/home/cerovaz/repos/data/jamendo_full"
    OUTPUT_FILE = "jamendo_partition_manifest.json"
    
    generate_manifest(DEFAULT_DATA_PATH, OUTPUT_FILE)
