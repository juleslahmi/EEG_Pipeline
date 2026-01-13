import json
import glob
import os
import pandas as pd
import numpy as np
import itertools

def get_dropped_set(filename):
    basename = os.path.basename(filename)
    region_str = basename.replace("aggregate_", "").replace("_seeds.json", "")
    if region_str == "baseline":
        return frozenset()
    # Split by underscore to get regions
    return frozenset(region_str.split("_"))

def analyze_results():
    files = glob.glob("runs/freq=8/aggregate_*_seeds.json")
    
    # Map frozenset(dropped_regions) -> results dict
    results_map = {}
    
    for f in files:
        with open(f, 'r') as fp:
            res = json.load(fp)
        dropped = get_dropped_set(f)
        results_map[dropped] = res

    all_regions = {'FRONTAL', 'CENTRAL', 'TEMPORAL', 'PARIETAL'}
    
    # We want to calculate the marginal contribution of each region R.
    # Contribution(R) = Average over all contexts S (where R is present) of:
    #                   Acc(Model with S dropped) - Acc(Model with S + R dropped)
    # Note: "Model with S dropped" means R is PRESENT.
    #       "Model with S+R dropped" means R is ABSENT.
    
    summary_data = []
    
    print(f"{'Region':<15} | {'Epoch Benefit':<15} | {'Patient Benefit':<15}")
    print("-" * 50)

    for region in sorted(all_regions):
        other_regions = list(all_regions - {region})
        
        epoch_deltas = []
        patient_deltas = []
        
        # Iterate through all subsets of other_regions (contexts where 'region' could be present)
        for i in range(len(other_regions) + 1):
            for context_tuple in itertools.combinations(other_regions, i):
                context = frozenset(context_tuple) # These are dropped
                
                # Case 1: Region is PRESENT (only context is dropped)
                # Case 2: Region is ABSENT (context + region is dropped)
                context_plus_region = context | {region}
                
                if context in results_map and context_plus_region in results_map:
                    # Get accuracies
                    # Use 'epoch_mean_across_seeds' if available, else 'epoch_mean'
                    # The previous error showed 'epoch_mean_across_seeds' is the correct key for aggregated files
                    
                    res_present = results_map[context]
                    res_absent = results_map[context_plus_region]
                    
                    acc_present_epoch = res_present.get('epoch_mean_across_seeds', res_present.get('epoch_mean', 0))
                    acc_absent_epoch = res_absent.get('epoch_mean_across_seeds', res_absent.get('epoch_mean', 0))
                    
                    acc_present_patient = res_present.get('patient_mean_across_seeds', res_present.get('patient_mean', 0))
                    acc_absent_patient = res_absent.get('patient_mean_across_seeds', res_absent.get('patient_mean', 0))
                    
                    delta_epoch = acc_present_epoch - acc_absent_epoch
                    delta_patient = acc_present_patient - acc_absent_patient
                    
                    epoch_deltas.append(delta_epoch)
                    patient_deltas.append(delta_patient)
        
        avg_epoch_benefit = np.mean(epoch_deltas) if epoch_deltas else 0.0
        avg_patient_benefit = np.mean(patient_deltas) if patient_deltas else 0.0
        
        summary_data.append({
            "Region": region,
            "Avg Epoch Benefit": avg_epoch_benefit,
            "Avg Patient Benefit": avg_patient_benefit
        })
        
        print(f"{region:<15} | {avg_epoch_benefit:+.4f}          | {avg_patient_benefit:+.4f}")

    df = pd.DataFrame(summary_data)
    df = df.sort_values("Avg Patient Benefit", ascending=False)
    
    print("\nDetailed Summary (Sorted by Patient Benefit):")
    print(df.to_string(index=False, float_format=lambda x: "{:+.4f}".format(x)))
    
    df.to_csv("runs/freq=8/marginal_contribution_analysis.csv", index=False)
    print("\nSaved to runs/freq=8/marginal_contribution_analysis.csv")

    # --- New Section: Single Region vs Drop-One Analysis ---
    print("\n" + "="*60)
    print("SUFFICIENCY vs NECESSITY ANALYSIS")
    print("="*60)
    print(f"{'Region':<15} | {'Solo Perf (Sufficiency)':<25} | {'Drop Impact (Necessity)':<25}")
    print(f"{'':<15} | {'(Keep Only This)':<25} | {'(Drop Only This)':<25}")
    print("-" * 70)

    # Baseline performance for reference
    baseline_acc = results_map[frozenset()].get('patient_mean_across_seeds', 0)
    
    single_region_data = []

    for region in sorted(all_regions):
        # 1. Solo Performance (Sufficiency)
        # We want the result where EVERYTHING ELSE is dropped.
        others = all_regions - {region}
        dropped_others = frozenset(others)
        
        if dropped_others in results_map:
            res = results_map[dropped_others]
            solo_acc = res.get('patient_mean_across_seeds', 0)
        else:
            solo_acc = float('nan')

        # 2. Drop Impact (Necessity)
        # We want the result where ONLY THIS REGION is dropped.
        dropped_self = frozenset({region})
        
        if dropped_self in results_map:
            res = results_map[dropped_self]
            drop_acc = res.get('patient_mean_across_seeds', 0)
            # Impact is how much accuracy DROPS compared to baseline
            # Positive impact means the region was necessary
            impact = baseline_acc - drop_acc
        else:
            drop_acc = float('nan')
            impact = float('nan')
            
        single_region_data.append({
            "Region": region,
            "Solo Accuracy": solo_acc,
            "Drop Accuracy": drop_acc,
            "Necessity (Base - Drop)": impact
        })

        print(f"{region:<15} | {solo_acc:.4f}                    | -{impact:.4f} (Acc: {drop_acc:.4f})")

    print("-" * 70)
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    
    # Save this specific view
    pd.DataFrame(single_region_data).to_csv("runs/freq=8/sufficiency_necessity.csv", index=False)

if __name__ == "__main__":
    analyze_results()
