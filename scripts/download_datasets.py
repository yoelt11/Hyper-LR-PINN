#!/usr/bin/env python3
"""Download datasets from Google Drive using gdown.

This script is based on the eff-physics-learn-dataset repository's download script:
https://github.com/yoelt11/eff-physics-learn-dataset/blob/main/scripts/download_datasets.py

It reads dataset links from the TOML configuration file and downloads them to the specified output directory.
"""

import argparse
import sys
from pathlib import Path

from eff_physics_learn_dataset.download import download_dataset, load_dataset_links


def main():
    parser = argparse.ArgumentParser(
        description="Download physics learning datasets from Google Drive"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Specific dataset to download (default: all datasets)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./data_gen/dataset"),
        help="Output directory for downloaded datasets (default: ./data_gen/dataset)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to dataset links TOML configuration file (default: packaged config)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets without downloading"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not extract zip files after download"
    )
    
    args = parser.parse_args()
    
    # Load dataset links
    if args.config is not None and not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        datasets = load_dataset_links(args.config)
    except Exception as e:
        print(f"Warning: Could not load dataset links from config: {e}")
        print("Using default dataset links from repository...")
        # Fallback to default links from the repository
        datasets = {
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
    
    # List mode
    if args.list:
        print("\nAvailable datasets:")
        print("-" * 40)
        for name in sorted(datasets.keys()):
            print(f"  • {name}")
        print(f"\nTotal: {len(datasets)} datasets")
        return
    
    # Determine which datasets to download
    if args.dataset:
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not found.")
            print("Available datasets:", ", ".join(sorted(datasets.keys())))
            sys.exit(1)
        to_download = {args.dataset: datasets[args.dataset]}
    else:
        to_download = datasets
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    print(f"\nDownloading {len(to_download)} dataset(s) to: {args.output_dir.absolute()}")
    
    successful = 0
    failed = 0
    
    for name, file_id in to_download.items():
        if download_dataset(name, file_id, args.output_dir, extract=not args.no_extract):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {successful + failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()


