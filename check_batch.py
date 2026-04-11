# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai",
#     "python-dotenv"
# ]
# ///

import os
import sys
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Check OpenAI Batch API Job Status")
    parser.add_argument("--batch-id", type=str, help="Specific batch ID to check explicitly")
    parser.add_argument("--job", choices=["tikz", "image"], default=None, 
                        help="Check the latest saved job ID from tikz2uml or image2uml runs")
    parser.add_argument("--no-hpc", action="store_true", 
                        help="Look for the job id in /Tmp/kumargau/ift6765/output instead of the HPC path")
    
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable is not set. Looking for it in .env...")
        
    client = OpenAI(api_key=api_key)
    
    batch_id = args.batch_id
    
    # If a specific batch-id was not passed, dynamically map to the saved file
    if not batch_id and args.job:
        if args.no_hpc:
            output_dir = "/Tmp/kumargau/ift6765/output"
        else:
            output_dir = "/project/def-syriani/gauransh/ift6765/output"
            
        if args.job == "tikz":
            batch_file = f"{output_dir}/.batch_job_id"
        else:
            batch_file = f"{output_dir}/.vision_batch_job_id"
            
        if not os.path.exists(batch_file):
            print(f"Error: Could not find batch ID file at {batch_file}")
            sys.exit(1)
            
        with open(batch_file, "r") as f:
            batch_ids_str = f.read().strip()
            
    if batch_id:
        # If passed explicitly via --batch-id
        batch_ids = [batch_id]
    elif args.job:
        # If pulled from the saved file
        batch_ids = [b.strip() for b in batch_ids_str.split(",") if b.strip()]
    else:
        print("Error: You must provide either --batch-id explicitly, or --job [tikz|image]")
        sys.exit(1)
        
    for current_batch in batch_ids:
        print(f"\nRetrieving batch status for ID: {current_batch}")
        try:
            batch = client.batches.retrieve(current_batch)
            print("=" * 50)
            print(f" Batch ID:         {batch.id}")
            print(f" Status:           {batch.status.upper()}")
            
            if batch.request_counts:
                print(f" Total Requests:   {batch.request_counts.total}")
                print(f" Completed:        {batch.request_counts.completed}")
                print(f" Failed:           {batch.request_counts.failed}")
                
            if getattr(batch, "created_at", None):
                dt = datetime.fromtimestamp(batch.created_at)
                print(f" Created At:       {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                
            if batch.errors and batch.errors.data:
                print("\n Errors Encountered:")
                for error in batch.errors.data:
                    print(f"  - [{error.code}]: {error.message}")
                    
            print("=" * 50)
        except Exception as e:
            print(f"Failed to retrieve batch {current_batch}: {e}")

if __name__ == "__main__":
    main()
