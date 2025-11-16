import os
import json
from tqdm import tqdm
import random
import numpy as np


# Helper functions for file operations
def get_file_path(filename):
    """Get absolute file path relative to this module."""
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    return os.path.join(current_directory, filename)


def load_json_dict(filepath, convert_tuple_keys=False):
    """Load a JSON dictionary from file with optional tuple key conversion."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if convert_tuple_keys:
            # Convert string keys "id1,id2" back to tuple (id1, id2)
            return {tuple(map(int, k.split(','))): v for k, v in data.items()}
        return data


def save_json_dict(filepath, data, convert_tuple_keys=False):
    """Save a dictionary to JSON file with optional tuple key conversion."""
    if convert_tuple_keys:
        # Convert tuple keys (id1, id2) to string "id1,id2"
        data = {f"{k[0]},{k[1]}": v for k, v in data.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def replace_tokens_at_index(token_ids, idx, replacement, num_to_remove=1):
    """Replace tokens at a given index with new tokens."""
    if isinstance(replacement, int):
        replacement = [replacement]
    return token_ids[:idx] + list(replacement) + token_ids[idx + num_to_remove:]


class PatokProcessor:
    """
    A processor that applies Patok expansion to the tokenized data.
    """
    def __init__(self, tokenizer, expand_prop=0.3, contract_prop=0.3, affix_preference=0.7, affixes_file=None):
        self.tokenizer = tokenizer
        self.expand_prop = expand_prop
        self.contract_prop = contract_prop
        self.affix_preference = affix_preference
        self.set_expansions()
        self.set_contractions()
        self.load_affixes(affixes_file)

    def expand(self, token_ids, expand_prop=0.3, max_num_to_expand=None, disable_tqdm=True):
        """
        Expand the sequence of tokens by splitting tokens.
        Args:
            token_ids (list): list of token_ids to expand.
            expand_prop (float): proportion of tokens to (try) expanding.
            max_num_to_expand (int): maximum number of tokens to expand.
            disable_tqdm (bool): whether to disable tqdm progress bar.
        Returns:
            list: list of token_ids after expanding.
        """
        expand_prop = self._get_prop_value(expand_prop, self.expand_prop)
        num_to_expand = int(len(token_ids) * expand_prop)
        num_expanded = 0
        
        for _ in tqdm(range(num_to_expand), disable=disable_tqdm):
            if max_num_to_expand is not None and num_expanded >= max_num_to_expand:
                break
            
            idx = np.random.randint(len(token_ids))
            token_id = token_ids[idx]
            
            if token_id in self.expansions:
                chosen_expansion = random.choice(self.expansions[token_id])
                token_ids = replace_tokens_at_index(token_ids, idx, chosen_expansion, num_to_remove=1)
                num_expanded += 1
        
        return token_ids

    def contract(self, token_ids, contract_prop=0.3, max_num_to_contract=None, disable_tqdm=True):
        """
        Contract the sequence of tokens by merging adjacent token pairs.
        Args:
            token_ids (list): list of token_ids to contract.
            contract_prop (float): proportion of tokens to (try) contracting.
            max_num_to_contract (int): maximum number of token pairs to contract.
            disable_tqdm (bool): whether to disable tqdm progress bar.
        Returns:
            list: list of token_ids after contracting.
        """
        contract_prop = self._get_prop_value(contract_prop, self.contract_prop)
        num_to_contract = int(len(token_ids) * contract_prop)
        num_contracted = 0
        
        for _ in tqdm(range(num_to_contract), disable=disable_tqdm):
            if max_num_to_contract is not None and num_contracted >= max_num_to_contract:
                break
            if len(token_ids) < 2:
                break
            
            idx = np.random.randint(len(token_ids) - 1)
            token_pair = (token_ids[idx], token_ids[idx + 1])
            
            if token_pair in self.contractions:
                contracted_token_id = self.contractions[token_pair]
                token_ids = replace_tokens_at_index(token_ids, idx, contracted_token_id, num_to_remove=2)
                num_contracted += 1
        
        return token_ids

    def set_expansions(self):
        """Loads expansions dict from file if it exists, otherwise builds and saves it."""
        filename = "data/tokenizer_expansions.json"
        json_file_path = get_file_path(filename)

        if os.path.exists(json_file_path):
            print(f"Found '{filename}' at: {json_file_path}")
            self.expansions = load_json_dict(json_file_path)
            assert isinstance(self.expansions, dict), f"{filename} must be a dictionary."
            print(f"Successfully loaded {filename}.")
        else:
            contractions, self.expansions = self.build_contractions_and_expansions()
            save_json_dict(json_file_path, self.expansions)
            print(f"Successfully saved {filename}.")
        
        print(f"Successfully set self.expansions.")

    def set_contractions(self):
        """Loads contractions dict from file if it exists, otherwise builds and saves it."""
        filename = "data/tokenizer_contractions.json"
        json_file_path = get_file_path(filename)

        if os.path.exists(json_file_path):
            print(f"Found '{filename}' at: {json_file_path}")
            self.contractions = load_json_dict(json_file_path, convert_tuple_keys=True)
            assert isinstance(self.contractions, dict), f"{filename} must be a dictionary."
            print(f"Successfully loaded {filename}.")
        else:
            self.contractions, expansions = self.build_contractions_and_expansions()
            save_json_dict(json_file_path, self.contractions, convert_tuple_keys=True)
            print(f"Successfully saved {filename}.")
        
        print(f"Successfully set self.contractions.")

    def build_contractions_and_expansions(self):
        """
        Build contractions and expansions dictionaries from tokenizer vocabulary.
        
        Returns:
            contractions (dict): {(token_id1, token_id2): merged_token_id}
            expansions (dict): {token_id: [(split_id1, split_id2), ...]}
        """
        contractions = self._build_contractions_dict()
        expansions = self._invert_contractions_to_expansions(contractions)
        return contractions, expansions

    def _build_contractions_dict(self):
        """Build the contractions dictionary from tokenizer's mergeable ranks."""
        ttokenizer_byt2int = self.tokenizer._mergeable_ranks
        ttokenizer_tokens_as_tuples = [tuple(token) for token in ttokenizer_byt2int.keys()]
        contractions = {}
        
        for i, (token_as_bytes, token_id) in tqdm(
            enumerate(ttokenizer_byt2int.items()),
            total=len(ttokenizer_byt2int),
            desc="Building tokenizer's contractions"
        ):
            if i < 256:
                assert len(token_as_bytes) == 1
                continue
            
            # Find all valid splits for this token
            splits = self._find_valid_splits(
                token_as_bytes, 
                ttokenizer_tokens_as_tuples, 
                ttokenizer_byt2int
            )
            
            for first_id, second_id in splits:
                contractions[(first_id, second_id)] = token_id
            
            assert len(splits) >= 1, f"No contraction found for {token_as_bytes=}"
        
        return contractions

    def _find_valid_splits(self, token_as_bytes, valid_tokens, token_mapping):
        """Find all valid ways to split a token into two sub-tokens."""
        splits = []
        for j in range(1, len(token_as_bytes)):
            first_part = token_as_bytes[:j]
            second_part = token_as_bytes[j:]
            
            if tuple(first_part) in valid_tokens and tuple(second_part) in valid_tokens:
                first_part_id = token_mapping[first_part]
                second_part_id = token_mapping[second_part]
                splits.append((first_part_id, second_part_id))
        
        return splits

    def _invert_contractions_to_expansions(self, contractions):
        """Convert contractions dict to expansions dict (inverse mapping)."""
        expansions = {}
        for token_pair, merged_token_id in contractions.items():
            if merged_token_id in expansions:
                expansions[merged_token_id].append(token_pair)
            else:
                expansions[merged_token_id] = [token_pair]
        return expansions

    def load_affixes(self, affixes_file=None):
        """Load Filipino affixes from file and convert them to token IDs."""
        if affixes_file is None:
            affixes_file = "data_other/filipino_affixes.txt"
        
        affixes_file_path = get_file_path(affixes_file)
        self.affix_strings = []
        self.affix_token_ids = set()
        
        if os.path.exists(affixes_file_path):
            print(f"Loading affixes from: {affixes_file_path}")
            self.affix_strings = self._load_affixes_from_file(affixes_file_path)
            self.affix_token_ids = self._convert_affixes_to_token_ids(self.affix_strings)
            print(f"Loaded {len(self.affix_strings)} affixes, {len(self.affix_token_ids)} are single tokens")
        else:
            print(f"Warning: Affixes file not found at {affixes_file_path}")

    def _load_affixes_from_file(self, filepath):
        """Read affixes from file, one per line."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _convert_affixes_to_token_ids(self, affix_strings):
        """Convert affix strings to single-token IDs."""
        affix_token_ids = set()
        for affix in affix_strings:
            try:
                token_ids = self.tokenizer.encode(affix)
                if len(token_ids) == 1:
                    affix_token_ids.add(token_ids[0])
            except Exception as e:
                print(f"Warning: Could not encode affix '{affix}': {e}")
        return affix_token_ids

    def affix_aware_expand_contract(self, token_ids, num_iterations=3, expand_prop=0.3, 
                                     contract_prop=0.3, affix_preference=0.7, 
                                     disable_tqdm=True):
        """
        Stochastically expand and contract tokens multiple times, with preference for
        forming Filipino affixes.
        
        Args:
            token_ids (list): list of token_ids to process.
            num_iterations (int): number of expand-contract cycles to perform.
            expand_prop (float): proportion of tokens to attempt expanding per iteration.
            contract_prop (float): proportion of tokens to attempt contracting per iteration.
            affix_preference (float): probability [0-1] of choosing affix-forming contractions when available.
            disable_tqdm (bool): whether to disable tqdm progress bar.
        
        Returns:
            list: list of token_ids after processing.
        """
        expand_prop = self._get_prop_value(expand_prop, self.expand_prop)
        contract_prop = self._get_prop_value(contract_prop, self.contract_prop)
        affix_preference = self._get_prop_value(affix_preference, self.affix_preference)
        
        for iteration in tqdm(range(num_iterations), desc="Affix-aware processing", disable=disable_tqdm):
            token_ids = self._selective_expand(token_ids, expand_prop, disable_tqdm=True)
            token_ids = self._affix_preferring_contract(token_ids, contract_prop, affix_preference, disable_tqdm=True)
        
        return token_ids

    def _get_prop_value(self, provided_value, default_value):
        """Return provided value if not None, otherwise return default."""
        return provided_value if provided_value is not None else default_value

    def _selective_expand(self, token_ids, expand_prop, disable_tqdm=True):
        """Expand tokens, avoiding expansion of tokens that are already affixes."""
        num_to_expand = int(len(token_ids) * expand_prop)
        num_expanded = 0
        
        for _ in range(num_to_expand):
            if len(token_ids) == 0:
                break
            
            expandable_indices, non_affix_expandable = self._find_expandable_indices(token_ids)
            
            if len(expandable_indices) == 0:
                break
            
            # Prefer expanding non-affix tokens
            idx = self._choose_index_preferring_non_affixes(expandable_indices, non_affix_expandable)
            
            # Perform expansion
            token_id = token_ids[idx]
            chosen_expansion = random.choice(self.expansions[token_id])
            token_ids = replace_tokens_at_index(token_ids, idx, chosen_expansion, num_to_remove=1)
            num_expanded += 1
        
        return token_ids

    def _find_expandable_indices(self, token_ids):
        """Find indices of expandable tokens, separating affix and non-affix tokens."""
        expandable_indices = []
        non_affix_expandable = []
        
        for idx, token_id in enumerate(token_ids):
            if token_id in self.expansions:
                expandable_indices.append(idx)
                if token_id not in self.affix_token_ids:
                    non_affix_expandable.append(idx)
        
        return expandable_indices, non_affix_expandable

    def _choose_index_preferring_non_affixes(self, all_indices, non_affix_indices):
        """Choose an index, preferring non-affix indices if available."""
        if len(non_affix_indices) > 0:
            return random.choice(non_affix_indices)
        return random.choice(all_indices)

    def _affix_preferring_contract(self, token_ids, contract_prop, affix_preference, disable_tqdm=True):
        """Contract tokens with preference for forming affixes."""
        num_to_contract = int(len(token_ids) * contract_prop)
        num_contracted = 0
        
        for _ in range(num_to_contract):
            if len(token_ids) < 2:
                break
            
            contractible_pairs, affix_forming_pairs = self._find_contractible_pairs(token_ids)
            
            if len(contractible_pairs) == 0:
                break
            
            # Choose pair based on affix preference
            idx, contracted_token_id = self._choose_contraction_with_affix_preference(
                contractible_pairs, affix_forming_pairs, affix_preference
            )
            
            # Perform the contraction
            token_ids = replace_tokens_at_index(token_ids, idx, contracted_token_id, num_to_remove=2)
            num_contracted += 1
        
        return token_ids

    def _find_contractible_pairs(self, token_ids):
        """Find all contractible adjacent token pairs, separating affix-forming ones."""
        contractible_pairs = []
        affix_forming_pairs = []
        
        for idx in range(len(token_ids) - 1):
            token_pair = (token_ids[idx], token_ids[idx + 1])
            if token_pair in self.contractions:
                contracted_token_id = self.contractions[token_pair]
                contractible_pairs.append((idx, contracted_token_id))
                
                if contracted_token_id in self.affix_token_ids:
                    affix_forming_pairs.append((idx, contracted_token_id))
        
        return contractible_pairs, affix_forming_pairs

    def _choose_contraction_with_affix_preference(self, all_pairs, affix_pairs, preference):
        """Choose a contraction pair, preferring affix-forming ones based on preference."""
        if len(affix_pairs) > 0 and random.random() < preference:
            return random.choice(affix_pairs)
        return random.choice(all_pairs)