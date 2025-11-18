"""
Analyze and compare tokenized datasets to measure differences.

This script compares three datasets:
1. raileymontalan/SEA-PILE-v2-tl-tokenized (baseline)
2. raileymontalan/SEA-PILE-v2-tl-tokenized-stochastok0.1
3. raileymontalan/SEA-PILE-v2-tl-tokenized-patok0.3-0.3-0.7

Metrics:
- Average length of ids column ± standard deviation
- Percentage of different rows between dataset pairs
"""

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
from multiprocessing import cpu_count
import json
from datetime import datetime
import os


def load_dataset_safely(dataset_name, split='train'):
    """Load a dataset from HuggingFace."""
    print(f"\nLoading dataset: {dataset_name} (split: {split})")
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Successfully loaded {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def compute_length_statistics(dataset, dataset_name, num_proc=None):
    """
    Compute average length and standard deviation of ids column.
    
    Args:
        dataset: HuggingFace dataset
        dataset_name: Name for display
        num_proc: Number of processes to use (None = auto-detect)
    
    Returns:
        dict with mean, std, min, max lengths
    """
    if num_proc is None:
        num_proc = cpu_count()
    
    print(f"\nComputing length statistics for {dataset_name} (using {num_proc} workers)...")
    
    # Use map with multiprocessing to compute lengths in parallel
    def get_length(example):
        return {'length': len(example['ids'])}
    
    dataset_with_lengths = dataset.map(
        get_length,
        num_proc=num_proc,
        desc="Computing lengths"
    )
    
    lengths = np.array(dataset_with_lengths['length'])
    
    stats = {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'median': np.median(lengths)
    }
    
    return stats


def compare_datasets(dataset_a, dataset_b, name_a, name_b, num_proc=None):
    """
    Compare two datasets and compute percentage of different rows.
    
    Args:
        dataset_a: First dataset
        dataset_b: Second dataset
        name_a: Name of first dataset
        name_b: Name of second dataset
        num_proc: Number of processes to use (None = auto-detect)
    
    Returns:
        dict with comparison statistics and set of indices where rows differ
    """
    if num_proc is None:
        num_proc = cpu_count()
    
    print(f"\nComparing {name_a} vs {name_b} (using {num_proc} workers)...")
    
    # Ensure same length
    min_len = min(len(dataset_a), len(dataset_b))
    if len(dataset_a) != len(dataset_b):
        print(f"Warning: Datasets have different lengths ({len(dataset_a)} vs {len(dataset_b)})")
        print(f"Comparing first {min_len} rows only")
    
    # Trim to same length if needed
    if len(dataset_a) > min_len:
        dataset_a = dataset_a.select(range(min_len))
    if len(dataset_b) > min_len:
        dataset_b = dataset_b.select(range(min_len))
    
    # Add index column for tracking
    dataset_a = dataset_a.add_column('idx', range(len(dataset_a)))
    dataset_b = dataset_b.add_column('idx', range(len(dataset_b)))
    
    # Compare in parallel using map with batched processing
    def compare_batch(examples_a, idx):
        # Get corresponding rows from dataset_b
        examples_b = dataset_b.select(idx)
        
        results = []
        for i, (ids_a, ids_b) in enumerate(zip(examples_a['ids'], examples_b['ids'])):
            is_different = (ids_a != ids_b)
            results.append({
                'is_different': is_different,
                'original_idx': idx[i]
            })
        
        return {
            'is_different': [r['is_different'] for r in results],
            'original_idx': [r['original_idx'] for r in results]
        }
    
    comparison_results = dataset_a.map(
        compare_batch,
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        desc="Comparing rows",
        remove_columns=dataset_a.column_names
    )
    
    # Collect results
    different_indices = set()
    same_indices = set()
    
    for i in range(len(comparison_results)):
        if comparison_results[i]['is_different']:
            different_indices.add(comparison_results[i]['original_idx'])
        else:
            same_indices.add(comparison_results[i]['original_idx'])
    
    num_different = len(different_indices)
    total_compared = min_len
    percentage_different = (num_different / total_compared) * 100
    
    return {
        'total_compared': total_compared,
        'num_different': num_different,
        'num_same': total_compared - num_different,
        'percentage_different': percentage_different,
        'percentage_same': 100 - percentage_different,
        'different_indices': different_indices,
        'same_indices': same_indices
    }


def print_comparison(comparison, name_a, name_b):
    """Pretty print comparison results."""
    print(f"\n{'='*80}")
    print(f"Comparison: {name_a} vs {name_b}")
    print(f"{'='*80}")
    print(f"Total rows compared:     {comparison['total_compared']:,}")
    print(f"Number of same rows:     {comparison['num_same']:,} ({comparison['percentage_same']:.2f}%)")
    print(f"Number of different rows: {comparison['num_different']:,} ({comparison['percentage_different']:.2f}%)")


def analyze_shared_rows(dataset_a, dataset_b, dataset_c, name_a, name_b, name_c, num_proc):
    """
    Analyze which rows have identical ids across different dataset combinations.
    
    Uses set operations to find:
    - Rows shared by A & B only
    - Rows shared by A & C only
    - Rows shared by B & C only
    - Rows shared by all three (A & B & C)
    - Rows unique to each dataset
    
    Args:
        dataset_a, dataset_b, dataset_c: The three datasets
        name_a, name_b, name_c: Names for display
        num_proc: Number of processes to use
    
    Returns:
        dict with set analysis results
    """
    if num_proc is None:
        num_proc = cpu_count()
    
    print(f"\n{'='*80}")
    print("ANALYZING ROW OVERLAPS (Set Operations)")
    print(f"{'='*80}")
    print(f"Finding which rows have identical 'ids' values across datasets (using {num_proc} workers)...")
    
    # Ensure all datasets have same length
    min_len = min(len(dataset_a), len(dataset_b), len(dataset_c))
    if len(dataset_a) != len(dataset_b) or len(dataset_b) != len(dataset_c):
        print(f"Warning: Datasets have different lengths")
        print(f"  {name_a}: {len(dataset_a)}, {name_b}: {len(dataset_b)}, {name_c}: {len(dataset_c)}")
        print(f"Comparing first {min_len} rows only\n")
    
    # Trim to same length if needed
    if len(dataset_a) > min_len:
        dataset_a = dataset_a.select(range(min_len))
    if len(dataset_b) > min_len:
        dataset_b = dataset_b.select(range(min_len))
    if len(dataset_c) > min_len:
        dataset_c = dataset_c.select(range(min_len))
    
    # Add index column for tracking
    dataset_a = dataset_a.add_column('idx', range(len(dataset_a)))
    
    # Compare in parallel using batched processing
    print("Building sets of identical rows...")
    
    def compare_three_way(examples_a, idx):
        # Get corresponding rows from dataset_b and dataset_c
        examples_b = dataset_b.select(idx)
        examples_c = dataset_c.select(idx)
        
        results = {
            'ab_match': [],
            'ac_match': [],
            'bc_match': [],
            'original_idx': []
        }
        
        for i, (ids_a, ids_b, ids_c) in enumerate(zip(examples_a['ids'], examples_b['ids'], examples_c['ids'])):
            results['ab_match'].append(ids_a == ids_b)
            results['ac_match'].append(ids_a == ids_c)
            results['bc_match'].append(ids_b == ids_c)
            results['original_idx'].append(idx[i])
        
        return results
    
    comparison_results = dataset_a.map(
        compare_three_way,
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        desc="Processing rows",
        remove_columns=dataset_a.column_names
    )
    
    # Build sets from results
    set_a = set()  # A and B match
    set_b = set()  # A and C match
    set_c = set()  # B and C match
    
    for i in range(len(comparison_results)):
        idx = comparison_results[i]['original_idx']
        if comparison_results[i]['ab_match']:
            set_a.add(idx)
        if comparison_results[i]['ac_match']:
            set_b.add(idx)
        if comparison_results[i]['bc_match']:
            set_c.add(idx)
    
    # Compute set operations
    # Rows where all three match
    all_three = set_a & set_b & set_c
    
    # Rows where only two match
    only_ab = set_a - set_b  # A&B match but not C
    only_ac = set_b - set_a  # A&C match but not B
    only_bc = set_c - (set_a & set_b)  # B&C match but not A
    
    # Rows unique to each (no matches with others)
    all_rows = set(range(min_len))
    matched_rows = set_a | set_b | set_c
    no_matches = all_rows - matched_rows
    
    results = {
        'total_rows': min_len,
        'all_three_match': len(all_three),
        'only_ab_match': len(only_ab),
        'only_ac_match': len(only_ac),
        'only_bc_match': len(only_bc),
        'no_matches': len(no_matches),
        'all_three_indices': all_three,
        'only_ab_indices': only_ab,
        'only_ac_indices': only_ac,
        'only_bc_indices': only_bc,
        'no_matches_indices': no_matches
    }
    
    return results


def print_shared_rows_analysis(results, name_a, name_b, name_c):
    """Pretty print shared rows analysis."""
    total = results['total_rows']
    
    print(f"\n{'='*80}")
    print("SET OVERLAP ANALYSIS")
    print(f"{'='*80}")
    print(f"Total rows analyzed: {total:,}\n")
    
    print("Rows with identical 'ids' values:")
    print("-" * 80)
    
    all_three = results['all_three_match']
    only_ab = results['only_ab_match']
    only_ac = results['only_ac_match']
    only_bc = results['only_bc_match']
    no_matches = results['no_matches']
    
    print(f"  All three datasets match ({name_a} ∩ {name_b} ∩ {name_c}):")
    print(f"    {all_three:,} rows ({all_three/total*100:.2f}%)")
    
    print(f"\n  Only two datasets match:")
    print(f"    {name_a} & {name_b} only (not {name_c}): {only_ab:,} rows ({only_ab/total*100:.2f}%)")
    print(f"    {name_a} & {name_c} only (not {name_b}): {only_ac:,} rows ({only_ac/total*100:.2f}%)")
    print(f"    {name_b} & {name_c} only (not {name_a}): {only_bc:,} rows ({only_bc/total*100:.2f}%)")
    
    print(f"\n  No datasets match (all three are different):")
    print(f"    {no_matches:,} rows ({no_matches/total*100:.2f}%)")
    
    # Verification
    total_accounted = all_three + only_ab + only_ac + only_bc + no_matches
    print(f"\n  Verification: {total_accounted:,} / {total:,} rows accounted for")


def print_shared_analysis(results):
    """Pretty print shared rows analysis."""
    print("\n" + "="*80)
    print("SHARED ROWS ANALYSIS")
    print("="*80)
    
    total = results['total_rows']
    
    print(f"\nTotal rows analyzed: {total:,}")
    print("\n" + "-"*80)
    print("Pairwise Sharing (rows identical between two datasets):")
    print("-"*80)
    print(f"  Baseline & Stochastok:  {results['shared_ab']:>8,} ({results['shared_ab']/total*100:>6.2f}%)")
    print(f"  Baseline & Patok:       {results['shared_ac']:>8,} ({results['shared_ac']/total*100:>6.2f}%)")
    print(f"  Stochastok & Patok:     {results['shared_bc']:>8,} ({results['shared_bc']/total*100:>6.2f}%)")
    
    print("\n" + "-"*80)
    print("Exclusive Sharing (rows identical in exactly these datasets):")
    print("-"*80)
    print(f"  ONLY Baseline & Stochastok (not Patok):  {results['only_ab']:>8,} ({results['only_ab']/total*100:>6.2f}%)")
    print(f"  ONLY Baseline & Patok (not Stochastok):  {results['only_ac']:>8,} ({results['only_ac']/total*100:>6.2f}%)")
    print(f"  ONLY Stochastok & Patok (not Baseline):  {results['only_bc']:>8,} ({results['only_bc']/total*100:>6.2f}%)")
    
    print("\n" + "-"*80)
    print("Universal Sharing:")
    print("-"*80)
    print(f"  All three datasets (A & B & C):           {results['shared_all']:>8,} ({results['shared_all']/total*100:>6.2f}%)")
    
    print("\n" + "-"*80)
    print("Unique Rows (different from both other datasets):")
    print("-"*80)
    print(f"  Unique to Baseline:     {results['unique_a']:>8,} ({results['unique_a']/total*100:>6.2f}%)")
    print(f"  Unique to Stochastok:   {results['unique_b']:>8,} ({results['unique_b']/total*100:>6.2f}%)")
    print(f"  Unique to Patok:        {results['unique_c']:>8,} ({results['unique_c']/total*100:>6.2f}%)")
    
    # Sanity check
    print("\n" + "-"*80)
    print("Venn Diagram Breakdown:")
    print("-"*80)
    categorized = (
        results['shared_all'] +  # All three
        results['only_ab'] +     # Only A&B
        results['only_ac'] +     # Only A&C
        results['only_bc'] +     # Only B&C
        results['unique_a'] +    # Only A
        results['unique_b'] +    # Only B
        results['unique_c']      # Only C
    )
    print(f"  Categorized rows: {categorized:,}")
    print(f"  Total rows:       {total:,}")
    if categorized == total:
        print("  ✓ All rows accounted for!")
    else:
        print(f"  ⚠ Mismatch: {abs(categorized - total)} rows")
    
    print("\n" + "="*80)


def print_statistics(stats, dataset_name):
    """Pretty print statistics for a dataset."""
    print(f"\n{'='*80}")
    print(f"Statistics for: {dataset_name}")
    print(f"{'='*80}")
    print(f"Average length: {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"Median length:  {stats['median']:.2f}")
    print(f"Min length:     {stats['min']}")
    print(f"Max length:     {stats['max']}")


def print_comparison(comparison, name_a, name_b):
    """Pretty print comparison results."""
    print(f"\n{'='*80}")
    print(f"Comparison: {name_a} vs {name_b}")
    print(f"{'='*80}")
    print(f"Total rows compared:     {comparison['total_compared']:,}")
    print(f"Number of same rows:     {comparison['num_same']:,} ({comparison['percentage_same']:.2f}%)")
    print(f"Number of different rows: {comparison['num_different']:,} ({comparison['percentage_different']:.2f}%)")


def save_results(statistics, comparisons, shared_analysis, output_dir='analysis_results'):
    """
    Save analysis results to JSON and text files.
    
    Args:
        statistics: Dictionary of length statistics for each dataset
        comparisons: Dictionary of pairwise comparison results
        shared_analysis: Dictionary of shared rows analysis results
        output_dir: Directory to save results (default: 'analysis_results')
    
    Returns:
        Tuple of (json_path, text_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for JSON (convert sets to lists)
    json_data = {
        'timestamp': timestamp,
        'statistics': statistics,
        'comparisons': {
            key: {
                'total_compared': val['total_compared'],
                'num_different': val['num_different'],
                'num_same': val['num_same'],
                'percentage_different': val['percentage_different'],
                'percentage_same': val['percentage_same'],
                # Don't save the actual indices to keep file size manageable
                'num_different_indices': len(val['different_indices']),
                'num_same_indices': len(val['same_indices'])
            }
            for key, val in comparisons.items()
        },
        'shared_analysis': {
            'total_rows': shared_analysis['total_rows'],
            'all_three_match': shared_analysis['all_three_match'],
            'only_ab_match': shared_analysis['only_ab_match'],
            'only_ac_match': shared_analysis['only_ac_match'],
            'only_bc_match': shared_analysis['only_bc_match'],
            'no_matches': shared_analysis['no_matches'],
            # Store percentage calculations
            'percentages': {
                'all_three_match': shared_analysis['all_three_match'] / shared_analysis['total_rows'] * 100,
                'only_ab_match': shared_analysis['only_ab_match'] / shared_analysis['total_rows'] * 100,
                'only_ac_match': shared_analysis['only_ac_match'] / shared_analysis['total_rows'] * 100,
                'only_bc_match': shared_analysis['only_bc_match'] / shared_analysis['total_rows'] * 100,
                'no_matches': shared_analysis['no_matches'] / shared_analysis['total_rows'] * 100,
            }
        }
    }
    
    # Save JSON
    json_path = os.path.join(output_dir, f'analysis_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save text report
    text_path = os.path.join(output_dir, f'analysis_report_{timestamp}.txt')
    with open(text_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET COMPARISON ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # Length Statistics
        f.write("="*80 + "\n")
        f.write("LENGTH STATISTICS\n")
        f.write("="*80 + "\n\n")
        for key in ['baseline', 'stochastok', 'patok']:
            stats = statistics[key]
            name = key.capitalize()
            f.write(f"{name}:\n")
            f.write(f"  Mean:   {stats['mean']:.2f}\n")
            f.write(f"  Std:    {stats['std']:.2f}\n")
            f.write(f"  Median: {stats['median']:.2f}\n")
            f.write(f"  Min:    {stats['min']}\n")
            f.write(f"  Max:    {stats['max']}\n\n")
        
        # Pairwise Comparisons
        f.write("="*80 + "\n")
        f.write("PAIRWISE COMPARISONS\n")
        f.write("="*80 + "\n\n")
        
        comp_names = {
            'baseline_vs_stochastok': ('Baseline', 'Stochastok'),
            'baseline_vs_patok': ('Baseline', 'Patok'),
            'stochastok_vs_patok': ('Stochastok', 'Patok')
        }
        
        for key, (name_a, name_b) in comp_names.items():
            comp = comparisons[key]
            f.write(f"{name_a} vs {name_b}:\n")
            f.write(f"  Total compared:    {comp['total_compared']:,}\n")
            f.write(f"  Same rows:         {comp['num_same']:,} ({comp['percentage_same']:.2f}%)\n")
            f.write(f"  Different rows:    {comp['num_different']:,} ({comp['percentage_different']:.2f}%)\n\n")
        
        # Shared Rows Analysis
        f.write("="*80 + "\n")
        f.write("SHARED ROWS ANALYSIS (Set Overlap)\n")
        f.write("="*80 + "\n\n")
        
        total = shared_analysis['total_rows']
        f.write(f"Total rows analyzed: {total:,}\n\n")
        
        f.write("Set Overlaps:\n")
        f.write(f"  All three match:           {shared_analysis['all_three_match']:,} ({shared_analysis['all_three_match']/total*100:.2f}%)\n")
        f.write(f"  Baseline & Stochastok:     {shared_analysis['only_ab_match']:,} ({shared_analysis['only_ab_match']/total*100:.2f}%)\n")
        f.write(f"  Baseline & Patok:          {shared_analysis['only_ac_match']:,} ({shared_analysis['only_ac_match']/total*100:.2f}%)\n")
        f.write(f"  Stochastok & Patok:        {shared_analysis['only_bc_match']:,} ({shared_analysis['only_bc_match']/total*100:.2f}%)\n")
        f.write(f"  No matches (all different): {shared_analysis['no_matches']:,} ({shared_analysis['no_matches']/total*100:.2f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n{'='*80}")
    print("RESULTS SAVED")
    print(f"{'='*80}")
    print(f"JSON results: {json_path}")
    print(f"Text report:  {text_path}")
    
    return json_path, text_path


def main(split='train', sample_size=None, num_proc=None, output_dir='analysis_results'):
    """
    Main analysis function.
    
    Args:
        split: Dataset split to analyze ('train', 'validation', etc.)
        sample_size: Optional number of samples to analyze (for testing)
        num_proc: Number of processes to use for parallel processing (None = auto-detect)
        output_dir: Directory to save results (default: 'analysis_results')
    
    Returns:
        Tuple of (statistics, comparisons, shared_analysis)
    """
    if num_proc is None:
        num_proc = cpu_count()
    
    print(f"Using {num_proc} worker processes for parallel computation\n")
    
    # Dataset names
    datasets_info = {
        'baseline': 'raileymontalan/SEA-PILE-v2-tl-tokenized',
        'stochastok': 'raileymontalan/SEA-PILE-v2-tl-tokenized-stochastok0.1',
        'patok': 'raileymontalan/SEA-PILE-v2-tl-tokenized-patok0.3-0.3-0.7'
    }
    
    # Load datasets
    datasets = {}
    for key, name in datasets_info.items():
        dataset = load_dataset_safely(name, split=split)
        if dataset is None:
            print(f"Failed to load {key} dataset. Exiting.")
            return
        
        # Sample if requested
        if sample_size is not None and sample_size < len(dataset):
            print(f"Sampling {sample_size} examples from {len(dataset)} total")
            dataset = dataset.select(range(sample_size))
        
        datasets[key] = dataset
    
    # Compute length statistics for each dataset
    print("\n" + "="*80)
    print("PART 1: LENGTH STATISTICS")
    print("="*80)
    
    statistics = {}
    for key, dataset in datasets.items():
        stats = compute_length_statistics(dataset, datasets_info[key], num_proc=num_proc)
        statistics[key] = stats
        print_statistics(stats, datasets_info[key])
    
    # Compare datasets pairwise
    print("\n" + "="*80)
    print("PART 2: PAIRWISE COMPARISONS")
    print("="*80)
    
    comparisons = {}
    
    # Baseline vs Stochastok
    comp_bs = compare_datasets(
        datasets['baseline'], 
        datasets['stochastok'],
        'Baseline',
        'Stochastok',
        num_proc=num_proc
    )
    comparisons['baseline_vs_stochastok'] = comp_bs
    print_comparison(comp_bs, 'Baseline', 'Stochastok')
    
    # Baseline vs Patok
    comp_bp = compare_datasets(
        datasets['baseline'],
        datasets['patok'],
        'Baseline',
        'Patok',
        num_proc=num_proc
    )
    comparisons['baseline_vs_patok'] = comp_bp
    print_comparison(comp_bp, 'Baseline', 'Patok')
    
    # Stochastok vs Patok
    comp_sp = compare_datasets(
        datasets['stochastok'],
        datasets['patok'],
        'Stochastok',
        'Patok',
        num_proc=num_proc
    )
    comparisons['stochastok_vs_patok'] = comp_sp
    print_comparison(comp_sp, 'Stochastok', 'Patok')
    
    # Analyze shared rows across all three datasets
    print("\n" + "="*80)
    print("PART 3: SHARED ROWS ANALYSIS")
    print("="*80)
    
    shared_analysis = analyze_shared_rows(
        datasets['baseline'],
        datasets['stochastok'],
        datasets['patok'],
        'Baseline',
        'Stochastok',
        'Patok',
        num_proc=num_proc
    )
    print_shared_rows_analysis(shared_analysis, 'Baseline', 'Stochastok', 'Patok')
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nLength Statistics:")
    print(f"{'Dataset':<20} {'Mean ± Std':<25} {'Median':<10}")
    print("-" * 80)
    for key in ['baseline', 'stochastok', 'patok']:
        stats = statistics[key]
        name = key.capitalize()
        print(f"{name:<20} {stats['mean']:.2f} ± {stats['std']:.2f:<14} {stats['median']:.2f}")
    
    print("\nDifference Percentages:")
    print(f"{'Comparison':<30} {'Different Rows %':<20}")
    print("-" * 80)
    print(f"{'Baseline vs Stochastok':<30} {comp_bs['percentage_different']:.2f}%")
    print(f"{'Baseline vs Patok':<30} {comp_bp['percentage_different']:.2f}%")
    print(f"{'Stochastok vs Patok':<30} {comp_sp['percentage_different']:.2f}%")
    
    print("\nShared Rows (Set Overlap):")
    print(f"{'Category':<40} {'Count':<15} {'Percentage'}")
    print("-" * 80)
    total = shared_analysis['total_rows']
    print(f"{'All three datasets match':<40} {shared_analysis['all_three_match']:,} {shared_analysis['all_three_match']/total*100:>14.2f}%")
    print(f"{'Baseline & Stochastok only':<40} {shared_analysis['only_ab_match']:,} {shared_analysis['only_ab_match']/total*100:>14.2f}%")
    print(f"{'Baseline & Patok only':<40} {shared_analysis['only_ac_match']:,} {shared_analysis['only_ac_match']/total*100:>14.2f}%")
    print(f"{'Stochastok & Patok only':<40} {shared_analysis['only_bc_match']:,} {shared_analysis['only_bc_match']/total*100:>14.2f}%")
    print(f"{'No matches (all different)':<40} {shared_analysis['no_matches']:,} {shared_analysis['no_matches']/total*100:>14.2f}%")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    # Save results
    save_results(statistics, comparisons, shared_analysis, output_dir=output_dir)
    
    return statistics, comparisons, shared_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and compare tokenized datasets (with parallel processing)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to analyze (default: train)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to analyze (default: all)"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help=f"Number of worker processes for parallel computation (default: auto-detect, {cpu_count()} available)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results (default: analysis_results)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Dataset Comparison Analysis")
    print("="*80)
    print(f"Split: {args.split}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size}")
    else:
        print("Sample size: Full dataset")
    print(f"Output directory: {args.output_dir}")
    
    main(split=args.split, sample_size=args.sample_size, num_proc=args.num_proc, output_dir=args.output_dir)
    
    # Usage examples:
    # python analyze_dataset_differences.py
    # python analyze_dataset_differences.py --split train --sample_size 1000
    # python analyze_dataset_differences.py --split validation
    # python analyze_dataset_differences.py --num_proc 16  # Use 16 workers
    # python analyze_dataset_differences.py --split train --sample_size 10000 --num_proc 32
    # python analyze_dataset_differences.py --output_dir my_results  # Custom output directory
