import os 
from functools import partial
import argparse
from datasets import load_dataset

from models.components.base_tokenizer import BaseTokenizer
from dataset_preprocessing.utils import write_tokenized_data_as_memmap


def tokenize(example, tokenizer):
    """Tokenize a single example and append EOT token."""
    ids = tokenizer.encode(example["text"])
    ids.append(tokenizer.eot_token)
    return {"ids": ids, "len": len(ids)}


def get_dataset_name(hf_dataset_name, config=None, suffix="-tokenized"):
    """Generate standardized dataset name."""
    base_name = hf_dataset_name.split('/')[-1]
    if config is not None:
        return f"{base_name}-{config}{suffix}"
    return f"{base_name}{suffix}"


def get_memmap_folder_path(data_dir, dataset_name):
    """Get the path to the memmap folder for a dataset."""
    return os.path.join(data_dir, "data_as_memmaps", dataset_name)


def ensure_directory_exists(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def check_memmap_exists(folder_path):
    """Check if tokenized memmap data already exists."""
    return os.path.exists(folder_path) and len(os.listdir(folder_path)) != 0


def load_pretokenized_dataset(hf_dataset_name):
    """Load a pre-tokenized dataset from HuggingFace."""
    pretokenized_map = {
        "Skylion007/openwebtext": "anyasims/openwebtext-tokenized"
    }
    
    if hf_dataset_name not in pretokenized_map:
        raise ValueError(f"Pre-tokenized HF dataset for {hf_dataset_name} not found.")
    
    tokenized_hf_id = pretokenized_map[hf_dataset_name]
    print(f"Loading pretokenized dataset from HuggingFace: {tokenized_hf_id}")
    
    tokenized_dataset = load_dataset(tokenized_hf_id)
    print(f"Dataset info: {tokenized_dataset}")
    
    assert "ids" in tokenized_dataset["train"].features, "Dataset must contain 'ids' column"
    print(f'Sample tokens: {tokenized_dataset["train"][0]["ids"][:10]}')
    print(f"Successfully loaded pretokenized dataset from HuggingFace")
    
    return tokenized_dataset


def load_and_tokenize_dataset(hf_dataset_name, config=None):
    """Load raw dataset from HuggingFace and tokenize it."""
    print(f"Loading {hf_dataset_name} dataset from HuggingFace...")
    
    if config is not None:
        dataset = load_dataset(hf_dataset_name, data_dir=config)
    else:
        dataset = load_dataset(hf_dataset_name)
    
    # Create train/val split
    dataset = dataset["train"].train_test_split(test_size=0.01, seed=489, shuffle=True)
    dataset["val"] = dataset.pop("test")
    
    print(f"Dataset: {dataset}")
    print(f"Dataset example: {dataset['train'][0]}\n")
    print(f"Tokenizing dataset...")
    
    # Initialize tokenizer and prepare tokenization function
    tokenizer = BaseTokenizer()
    tokenize_fn = partial(tokenize, tokenizer=tokenizer)
    
    # Determine number of processes to use
    max_procs = max(os.cpu_count() // 2, 12)
    print(f"Using {max_procs} processors (out of {os.cpu_count()} available).")
    
    # Tokenize the dataset
    dataset_tokenized = dataset.map(
        tokenize_fn,
        desc="Tokenizing dataset",
        num_proc=max_procs
    )
    
    return dataset_tokenized


def save_tokenized_to_memmap(tokenized_dataset, memmap_folder):
    """Save tokenized dataset as memmap bin files."""
    try:
        write_tokenized_data_as_memmap(
            tokenized=tokenized_dataset, 
            tokenized_data_folder=memmap_folder,
        )
        print(f"Successfully saved tokenized dataset as memmap to {memmap_folder}")
        return True
    except Exception as exc:
        print(f"Error saving memmap: {exc}")
        # Clean up partial files
        for file in os.listdir(memmap_folder):
            os.remove(os.path.join(memmap_folder, file))
        raise RuntimeError("Failed to process and write data") from exc


def save_tokenized_to_hf(dataset_tokenized, hf_username, dataset_name, data_dir):
    """Save tokenized dataset to HuggingFace or local directory as fallback."""
    hf_repo_id = f"{hf_username}/{dataset_name}"
    
    try:
        print(f"Attempting to push dataset to: {hf_repo_id}")
        dataset_tokenized.push_to_hub(hf_repo_id)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{hf_repo_id}")
        return True
    except Exception as e:
        print(f"Pushing to HuggingFace failed: {e}")
        
        # Fallback to local save
        try:
            print(f"Attempting to save to local directory")
            local_folder = os.path.join(data_dir, "data_as_datasets", dataset_name)
            ensure_directory_exists(local_folder)
            dataset_tokenized.save_to_disk(local_folder)
            print(f"Successfully saved dataset to: {local_folder}")
            return False
        except Exception as e:
            print(f"Saving to local directory failed: {e}")
            return False


def prepare_data_tokenize(
        hf_dataset_name,
        data_dir,
        get_pretokenized_from_hf=False,
        save_to_hf=True,
        hf_username=None,
        config=None
        ):
    """
    Main function to prepare and tokenize dataset for training.
    
    Args:
        hf_dataset_name: HuggingFace dataset identifier
        data_dir: Directory to store processed data
        get_pretokenized_from_hf: Whether to load pre-tokenized data
        save_to_hf: Whether to save tokenized data to HuggingFace
        hf_username: HuggingFace username for uploading
        config: Optional dataset configuration
    """
    # Setup paths and check if already processed
    dataset_name = get_dataset_name(hf_dataset_name, config)
    memmap_folder = get_memmap_folder_path(data_dir, dataset_name)
    
    if check_memmap_exists(memmap_folder):
        print(f"Tokenized memmap data already exists (path={memmap_folder})")
        return
    
    ensure_directory_exists(memmap_folder)
    
    # Load or tokenize dataset
    if get_pretokenized_from_hf:
        tokenized_dataset = load_pretokenized_dataset(hf_dataset_name)
    else:
        tokenized_dataset = load_and_tokenize_dataset(hf_dataset_name, config)
    
    # Save as memmap (required for training)
    print(f"Saving to memmap...")
    successfully_saved_memmap = save_tokenized_to_memmap(tokenized_dataset, memmap_folder)
    
    # Optionally save to HuggingFace
    if save_to_hf and hf_username:
        save_tokenized_to_hf(tokenized_dataset, hf_username, dataset_name, data_dir)
    
    print(f"\n\nSuccessfully prepared dataset for training: {successfully_saved_memmap}")
    print(f"Dataset path: {memmap_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name", default="Skylion007/openwebtext", type=str)
    parser.add_argument("--data_dir", default="./data", type=str, help="Path to the data directory")
    parser.add_argument("--get_pretokenized_from_hf", default=False, type=bool, help="Load a dataset that is already tokenized from HuggingFace")
    parser.add_argument("--save_to_hf", default=False, type=bool, help="Save to HuggingFace after tokenization.")
    parser.add_argument("--hf_username", default=None, type=str, help="HuggingFace username used if save-to-hf==True.")
    parser.add_argument("--config", default=None, type=str, help="HuggingFace config used.")
    args = parser.parse_args()
    print(f"\nArgs: {args}\n")
    if args.save_to_hf:
        assert args.hf_username is not None, "hf_username must be provided if args.save_to_hf==True"

    prepare_data_tokenize(
        hf_dataset_name=args.hf_dataset_name,
        data_dir=args.data_dir,
        get_pretokenized_from_hf=args.get_pretokenized_from_hf,
        save_to_hf=args.save_to_hf,
        hf_username=args.hf_username,
        config=args.config,
    )

    # run with:
    # python dataset_preprocessing/tokenize_dataset.py --hf_dataset_name Skylion007/openwebtext --data_dir ./data --get_pretokenized_from_hf True --save_to_hf True --hf_username XXX
    # python dataset_preprocessing/tokenize_dataset.py --get_pretokenized_from_hf True --hf_username anyasims
    # python dataset_preprocessing/tokenize_dataset.py --save_to_hf True --hf_username anyasims
    # python dataset_preprocessing/tokenize_dataset.py --get_pretokenized_from_hf True
            

    



