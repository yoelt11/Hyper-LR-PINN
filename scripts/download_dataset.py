#!/usr/bin/env python
"""
Download datasets from eff-physics-learn-dataset.

This script is a wrapper around the eff-physics-learn-dataset download functionality.
It uses the same download utilities as the repository's scripts/download_datasets.py.

Dataset links are from:
https://github.com/yoelt11/eff-physics-learn-dataset/blob/main/configs/datasets/dataset_links.toml
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from eff_physics_learn_dataset.download import download_dataset, load_dataset_links
    import gdown
except ImportError as e:
    print(f"Error importing download utilities: {e}")
    print("Make sure eff-physics-learn-dataset is installed.")
    sys.exit(1)


def get_default_dataset_links():
    """Get default dataset links from the repository.
    
    Dataset links are from:
    https://raw.githubusercontent.com/yoelt11/eff-physics-learn-dataset/main/configs/datasets/dataset_links.toml
    """
    return {
        "helmholtz2D": "1SLg7GUxzUIl6tWQ-xxe-HFBnN4NqaigU",
        "helmholtz3D": "1Y4iqmHWf-JnnVUoCt767VVTrIsio4tqP",
        "burgers": "1ghwmA_Pir1epE45UXjZ5FYrYX0ZpdGG9",
        "allen_cahn": "1R8ciIuib4QO_d5HtY12DPwrsBeoflj1Y",
        "flow_mixing": "1ly3v8qysghp_6pJyGe1bCCpCTd4snank",
        "convection": "14ontlZyzys_pPKQpeoIW81zKl2Scxi9R",
        "hlrp_cdr": "132eWiCbSZKienhssNDRVk4DH9PmX8zVP",
        "hlrp_convection": "1dW1mQ5Qh-CaJOWFAYF-aTUCq2HO_j5cy",
        "hlrp_diffusion": "18_1N48-e8J4Ynr1SN7B5B9SFUMPSl5bd",
        "hlrp_helmholtz": "1JFbt7yiyM0nbRXuw7Vkrh7b105uYB-w_",
        "hlrp_reaction": "1DtlEoPjElE-zPNTX0aQ5CNi-Z3Rd__HL",
    }


def list_datasets():
    """List all available datasets."""
    try:
        # First try to load from config file if available
        repo_root = Path(__file__).resolve().parent.parent.parent
        config_path = repo_root / "configs" / "datasets" / "dataset_links.toml"
        
        if config_path.exists():
            links = load_dataset_links(str(config_path))
        else:
            # Fall back to default links
            links = get_default_dataset_links()
        
        return links
    except Exception as e:
        # If loading fails, use defaults
        print(f"Note: Using default dataset links ({e})")
        return get_default_dataset_links()


def download_single_dataset(name: str, file_id: str, output_dir: Path):
    """Download a single dataset."""
    print(f"Downloading {name}...")
    print(f"  File ID: {file_id}")
    print(f"  Output: {output_dir}")
    
    success = download_dataset(name, file_id, output_dir, extract=True)
    
    if success:
        print(f"✓ Successfully downloaded {name}")
        # Verify the dataset structure
        dataset_dir = output_dir / name
        ground_truth_dir = dataset_dir / "ground_truth"
        if ground_truth_dir.exists():
            print(f"✓ Dataset structure verified: {ground_truth_dir}")
        else:
            print(f"⚠ Warning: ground_truth directory not found at {ground_truth_dir}")
    else:
        print(f"✗ Failed to download {name}")
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Download datasets from eff-physics-learn-dataset')
    parser.add_argument('--list', action='store_true',
                       help='List all available datasets')
    parser.add_argument('-d', '--dataset', type=str,
                       help='Dataset name to download (e.g., allen_cahn, convection)')
    parser.add_argument('-o', '--output', type=str, default='./data_gen/dataset',
                       help='Output directory for datasets (default: ./data_gen/dataset)')
    parser.add_argument('--file-id', type=str,
                       help='Google Drive file ID (if dataset not in config)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        print("Available datasets:")
        links = list_datasets()
        if links:
            for name in sorted(links.keys()):
                print(f"  - {name}")
        else:
            print("  (Could not load dataset list)")
            print("\nCommon dataset names:")
            print("  - allen_cahn")
            print("  - convection")
            print("  - burgers")
            print("  - helmholtz2D")
            print("  - helmholtz3D")
            print("\nTo download, you may need to:")
            print("  1. Clone the repository to get dataset_links.toml")
            print("  2. Or specify --file-id manually")
        return
    
    if args.dataset:
        # Get file ID from config or defaults
        links = list_datasets()
        file_id = args.file_id
        
        if not file_id and args.dataset in links:
            file_id = links[args.dataset]
            print(f"Found {args.dataset} in dataset links")
        elif not file_id:
            print(f"Error: {args.dataset} not found in available datasets")
            print(f"\nAvailable datasets: {', '.join(sorted(links.keys()))}")
            print("\nTo download with a custom file ID, use:")
            print(f"  python scripts/download_dataset.py -d {args.dataset} --file-id <GOOGLE_DRIVE_FILE_ID>")
            return
        
        download_single_dataset(args.dataset, file_id, output_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

