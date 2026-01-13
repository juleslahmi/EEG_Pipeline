import sys
import mne
import numpy as np
from pathlib import Path
import collections
import re

def main():
    if len(sys.argv) < 2:
        # Tries to find a sample file if none provided
        sample_files = list(Path("Data/Control").glob("*.set"))
        if sample_files:
            file_path = sample_files[0]
            print(f"Usage: python scripts/inspect_file_events.py <path_to_set_file>")
            print(f"No file argument provided. Using found sample: {file_path}\n")
        else:
            print("Usage: python scripts/inspect_file_events.py <path_to_set_file>")
            sys.exit(1)
    else:
        file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Loading {file_path} ...")
    
    # Try loading as epochs (common in this project)
    try:
        epochs = mne.read_epochs_eeglab(str(file_path), verbose=False)
        events = epochs.events
        event_id_map = epochs.event_id
        print("-> Successfully loaded as Epochs.")
    except Exception as e_epochs:
        print(f"-> Could not load as Epochs ({e_epochs}). Trying as Raw...")
        try:
            raw = mne.io.read_raw_eeglab(str(file_path), preload=False, verbose=False)
            events, event_id_map = mne.events_from_annotations(raw, verbose=False)
            print("-> Successfully loaded as Raw data.")
        except Exception as e_raw:
            print(f"Error: Could not load file as Epochs or Raw.")
            print(f"Epochs error: {e_epochs}")
            print(f"Raw error: {e_raw}")
            sys.exit(1)

    # Count occurrences of each event code
    event_codes = events[:, -1]
    counts = collections.Counter(event_codes)
    
    # Reverse the map to look up labels by ID
    id_to_label = {v: k for k, v in event_id_map.items()}

    print("\n=== Event Summary (Raw) ===")
    print(f"{'Event ID':<10} | {'Count':<8} | {'Label'}")
    print("-" * 40)
    
    sorted_codes = sorted(counts.keys())
    for code in sorted_codes:
        count = counts[code]
        label = id_to_label.get(code, "<Unknown>")
        print(f"{code:<10} | {count:<8} | {label}")

    print("-" * 40)
    print(f"Total Events: {len(event_codes)}")
    
    # Parse using the project logic to see "Clean" events
    print("\n=== Parsed Event Summary (Cleaned) ===")
    parsed_counts = collections.defaultdict(int)
    
    for code in event_codes:
        label = id_to_label.get(int(code), str(code))
        # Logic from src/data_load.py
        m = re.search(r"\((\d+)\)", label)
        if m:
            clean_code = int(m.group(1))
            parsed_counts[clean_code] += 1
        else:
            # Fallback for labels that might not match
            parsed_counts[f"Unparseable containing '{label}'"] += 1
            
    print(f"{'Clean Code':<30} | {'Count':<8}")
    print("-" * 40)
    for code in sorted(parsed_counts.keys(), key=lambda x: str(x)):
        print(f"{str(code):<30} | {parsed_counts[code]:<8}")

if __name__ == "__main__":
    main()
