"""
Necessary to be run before training to make sure all of the data is preprcessed etc.
"""
import os 
from functools import partial
import argparse
from datasets import load_dataset

from models.components.base_tokenizer import BaseTokenizer
from patok_processor import PatokProcessor
from dataset_preprocessing.utils import write_tokenized_data_as_memmap


def tokenize(example, tokenizer):
    """Tokenize a single example and append EOT token."""
    ids = tokenizer.encode(example["text"])
    ids.append(tokenizer.eot_token)
    return {"ids": ids, "len": len(ids)}


def patok_expand_contract(example, patok_processor):
    """Expand and contract tokenized example using Patok processor."""
    ids = patok_processor.affix_aware_expand_contract(example["ids"])
    return {"ids": ids, "len": len(ids)}


def get_dataset_name(hf_dataset_name, config=None, expand_prop=None, contract_prop=None, affix_preference=None):
    """Generate standardized dataset name with optional expansion suffix."""
    base_name = hf_dataset_name.split('/')[-1]
    if config is not None:
        name = f"{base_name}-{config}"
    else:
        name = base_name
    
    if expand_prop is not None and contract_prop is not None and affix_preference is not None:
        name = f"{name}-patok{expand_prop}-{contract_prop}-{affix_preference}"
    
    return name


def get_memmap_folder_path(data_dir, dataset_name):
    """Get the path to the memmap folder for a dataset."""
    return os.path.join(data_dir, "data_as_memmaps", dataset_name)


def ensure_directory_exists(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def check_memmap_exists(folder_path):
    """Check if expanded memmap data already exists."""
    return os.path.exists(folder_path) and len(os.listdir(folder_path)) != 0


def load_preexpanded_dataset(hf_dataset_name):
    """Load a pre-expanded dataset from HuggingFace."""
    hf_expanded_dataset_name = f"{hf_dataset_name}"
    print(f"Loading pre-expanded dataset from HuggingFace: {hf_expanded_dataset_name}")
    
    expanded_dataset = load_dataset(hf_expanded_dataset_name)
    print(f"Dataset info: {expanded_dataset}")
    
    assert "ids" in expanded_dataset["train"].features, "Dataset must contain 'ids' column"
    print(f'Sample tokens: {expanded_dataset["train"][0]["ids"][:10]}')
    print(f"Successfully loaded pre-expanded dataset from HuggingFace")
    
    return expanded_dataset


def load_pretokenized_and_expand(hf_dataset_name, expand_prop, contract_prop, affix_preference, affix_file):
    """Load pre-tokenized dataset and expand it with Patok."""
    hf_tokenized_dataset_name = f"{hf_dataset_name}"
    print(f"Loading pretokenized dataset from HuggingFace: {hf_tokenized_dataset_name}")
    
    tokenized_dataset = load_dataset(hf_tokenized_dataset_name)
    print(f"Dataset info: {tokenized_dataset}")
    
    assert "ids" in tokenized_dataset["train"].features, "Dataset must contain 'ids' column"
    print(f'Sample tokens: {tokenized_dataset["train"][0]["ids"][:10]}')
    print(f"Successfully loaded pretokenized dataset from HuggingFace")
    
    # Expand with Patok
    tokenizer = BaseTokenizer()
    patok_processor = PatokProcessor(tokenizer=tokenizer.tokenizer, expand_prop=expand_prop, contract_prop=contract_prop, affix_preference=affix_preference, affixes_file=affix_file)
    expand_contract_fn = partial(patok_expand_contract, patok_processor=patok_processor)
    
    print(f"Expanding dataset with Patok (expand_prop={expand_prop}, contract_prop={contract_prop}, affix_preference={affix_preference})...")
    max_procs = get_num_processors()
    
    expanded_dataset = tokenized_dataset.map(
        expand_contract_fn,
        desc="Patok expanding-contracting dataset",
        num_proc=max_procs
    )
    
    return expanded_dataset


def load_tokenize_and_expand(hf_dataset_name, config, expand_prop, contract_prop, affix_preference, affix_file):
    """Load raw dataset, tokenize, and expand with Patok."""
    print(f"Loading {hf_dataset_name} dataset from HuggingFace...")
    
    if config is not None:
        dataset = load_dataset(hf_dataset_name, data_dir=config)
    else:
        dataset = load_dataset(hf_dataset_name)
    
    print(f"Dataset info: {dataset}")
    
    # Create train/val split
    dataset = dataset["train"].train_test_split(test_size=0.01, seed=489, shuffle=True)
    dataset["val"] = dataset.pop("test")
    
    print(f"Dataset: {dataset}")
    print(f"Dataset example: {dataset['train'][0]}\n")
    
    # Initialize tokenizer and processor
    tokenizer = BaseTokenizer()
    tokenize_fn = partial(tokenize, tokenizer=tokenizer)
    patok_processor = PatokProcessor(tokenizer=tokenizer.tokenizer, expand_prop=expand_prop, contract_prop=contract_prop, affix_preference=affix_preference, affixes_file=affix_file)
    expand_contract_fn = partial(patok_expand_contract, patok_processor=patok_processor)
    
    max_procs = get_num_processors()
    
    # Tokenize the dataset
    print(f"Tokenizing dataset...")
    dataset_tokenized = dataset.map(
        tokenize_fn,
        desc="Tokenizing dataset",
        num_proc=max_procs
    )
    
    assert "ids" in dataset_tokenized["train"].features, "Dataset must contain 'ids' column"
    
    # Expand with Patok
    print(f"Expanding dataset with Patok (expand_prop={expand_prop}, contract_prop={contract_prop}, affix_preference={affix_preference})...")
    dataset_expanded = dataset_tokenized.map(
        expand_contract_fn,
        desc="Patok expanding-contracting dataset",
        num_proc=max_procs
    )
    
    return dataset_expanded


def get_num_processors():
    """Get optimal number of processors to use."""
    max_procs = max(os.cpu_count() // 2, 12)
    print(f"Using {max_procs} processors (out of {os.cpu_count()} available).")
    return max_procs


def save_expanded_to_memmap(expanded_dataset, memmap_folder):
    """Save expanded dataset as memmap bin files."""
    try:
        write_tokenized_data_as_memmap(
            tokenized=expanded_dataset,
            tokenized_data_folder=memmap_folder,
        )
        print(f"Successfully saved expanded dataset as memmap to {memmap_folder}")
        return True
    except Exception as exc:
        print(f"Error saving memmap: {exc}")
        # Clean up partial files
        for file in os.listdir(memmap_folder):
            os.remove(os.path.join(memmap_folder, file))
        raise RuntimeError("Failed to process and write data") from exc


def save_expanded_to_hf(dataset_expanded, hf_username, dataset_name, data_dir):
    """Save expanded dataset to HuggingFace or local directory as fallback."""
    hf_repo_id = f"{hf_username}/{dataset_name}"
    
    try:
        print(f"Attempting to push expanded dataset to: {hf_repo_id}")
        dataset_expanded.push_to_hub(hf_repo_id)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{hf_repo_id}")
        return True
    except Exception as e:
        print(f"Pushing to HuggingFace failed: {e}")
        
        # Fallback to local save
        try:
            print(f"Attempting to save to local directory")
            local_folder = os.path.join(data_dir, "data_as_datasets", dataset_name)
            ensure_directory_exists(local_folder)
            dataset_expanded.save_to_disk(local_folder)
            print(f"Successfully saved expanded dataset to: {local_folder}")
            return False
        except Exception as e:
            print(f"Saving to local directory failed: {e}")
            return False


def prepare_data_expand(
        hf_dataset_name,
        data_dir,
        get_pretokenized_from_hf=False,
        get_preexpanded_from_hf=False,
        save_to_hf=True,
        hf_username=None,
        expand_prop=0.3,
        contract_prop=0.3,
        affix_preference=0.7,
        affix_file="data_other/filipino_affixes.txt",
        config=None
        ):
    """
    Main function to prepare, tokenize, and expand dataset with Patok for training.
    
    Args:
        hf_dataset_name: HuggingFace dataset identifier
        data_dir: Directory to store processed data
        get_pretokenized_from_hf: Whether to load pre-tokenized data (will expand it)
        get_preexpanded_from_hf: Whether to load pre-expanded data (already tokenized and expanded)
        save_to_hf: Whether to save expanded data to HuggingFace
        hf_username: HuggingFace username for uploading
        expand_prop: Patok expansion proportion
        contract_prop: Patok contraction proportion
        affix_preference: Patok affix preference
        affix_file: Path to affixes file
        config: Optional dataset configuration
    """
    # Setup paths and check if already processed
    expanded_dataset_name = get_dataset_name(hf_dataset_name, config, expand_prop, contract_prop, affix_preference)
    memmap_folder = get_memmap_folder_path(data_dir, expanded_dataset_name)
    
    if check_memmap_exists(memmap_folder):
        print(f"Patok-expanded memmap data already exists (path={memmap_folder})")
        return
    
    ensure_directory_exists(memmap_folder)
    
    # Load or process dataset based on input type
    if get_preexpanded_from_hf:
        # Load pre-expanded dataset
        expanded_dataset = load_preexpanded_dataset(hf_dataset_name)
    elif get_pretokenized_from_hf:
        # Load pre-tokenized and expand
        expanded_dataset = load_pretokenized_and_expand(hf_dataset_name, expand_prop, contract_prop, affix_preference, affix_file)
    else:
        # Load raw, tokenize, and expand
        expanded_dataset = load_tokenize_and_expand(hf_dataset_name, config, expand_prop, contract_prop, affix_preference, affix_file)
    
    # Save as memmap (required for training)
    print(f"Saving to memmap...")
    successfully_saved_memmap = save_expanded_to_memmap(expanded_dataset, memmap_folder)
    
    # Optionally save to HuggingFace
    if save_to_hf and hf_username:
        save_expanded_to_hf(expanded_dataset, hf_username, expanded_dataset_name, data_dir)
    
    print(f"\n\nSuccessfully prepared Patok-expanded dataset for training: {successfully_saved_memmap}")
    print(f"Dataset path: {memmap_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name", default="anyasims/openwebtext-tokenized", type=str, required=False)
    parser.add_argument("--data_dir", default="./data", type=str, required=False, help="Path to the data directory")
    parser.add_argument("--get_preexpanded_from_hf", default=False, type=bool, required=False, help="Load a dataset that is already expanded from HuggingFace")
    parser.add_argument("--get_pretokenized_from_hf", default=False, type=bool, required=False, help="Load a dataset that is already tokenized from HuggingFace")
    parser.add_argument("--save_to_hf", default=False, type=bool, required=False, help="Save to HuggingFace after tokenization.")
    parser.add_argument("--hf_username", default=None, type=str, required=False, help="HuggingFace username used if save-to-hf==True.")
    parser.add_argument("--expand_prop", default=0.3, type=float, required=False, help="Patok expand proportion hyperparameter.")
    parser.add_argument("--contract_prop", default=0.3, type=float, required=False, help="Patok contract proportion hyperparameter.")
    parser.add_argument("--affix_preference", default=0.7, type=float, required=False, help="Patok affix preference hyperparameter (0-1).")
    parser.add_argument("--affix_file", default="data_other/filipino_affixes.txt", type=str, required=False, help="Path to affixes file.")
    parser.add_argument("--config", default=None, type=str, help="HuggingFace config used.")
    args = parser.parse_args()
    print(f"\nArgs: {args}\n")
    if args.save_to_hf:
        assert args.hf_username is not None, "hf_username must be provided if args.save_to_hf==True"

    prepare_data_expand(
        hf_dataset_name=args.hf_dataset_name,
        data_dir=args.data_dir,
        get_preexpanded_from_hf=args.get_preexpanded_from_hf,
        get_pretokenized_from_hf=args.get_pretokenized_from_hf,
        save_to_hf=args.save_to_hf,
        hf_username=args.hf_username,
        expand_prop=args.expand_prop,
        contract_prop=args.contract_prop,
        affix_preference=args.affix_preference,
        affix_file=args.affix_file,
        config=args.config,
    )

    # run with:
    # python dataset_preprocessing/patok_expand_dataset.py --hf_dataset_name Skylion007/openwebtext --data_dir ./data --get_preexpanded_from_hf True --save_to_hf True --hf_username XXX --expand_prop 0.1
    # python dataset_preprocessing/patok_expand_dataset.py --get_preexpanded_from_hf True --hf_username anyasims
    # python dataset_preprocessing/patok_expand_dataset.py --save_to_hf True --hf_username anyasims
    # python dataset_preprocessing/patok_expand_dataset.py --get_preexpanded_from_hf True

            

    



