# Affix-Aware Token Processing Implementation

## Overview

The `PatokProcessor` now includes an affix-aware expand-contract mechanism that stochastically manipulates tokens to preferentially form Filipino affixes while maintaining vocabulary validity.

## Key Features

### 1. **Affix Loading (`load_affixes`)**
- Loads Filipino affixes from `data_other/filipino_affixes.txt`
- Converts affixes to token IDs
- Only single-token affixes are used for preference (multi-token affixes are loaded but not prioritized)
- Creates `self.affix_token_ids` set for fast lookup

### 2. **Main Function (`affix_aware_expand_contract`)**
```python
affix_aware_expand_contract(
    token_ids, 
    num_iterations=3,
    expand_prop=0.3,
    contract_prop=0.3,
    affix_preference=0.7,
    disable_tqdm=True
)
```

**Parameters:**
- `token_ids`: Input token sequence
- `num_iterations`: Number of expand-contract cycles (default: 3)
- `expand_prop`: Proportion of tokens to expand per iteration (default: 0.3)
- `contract_prop`: Proportion of tokens to contract per iteration (default: 0.3)
- `affix_preference`: Probability [0-1] of choosing affix-forming contractions (default: 0.7)
- `disable_tqdm`: Whether to show progress bar

**Note:** These defaults are also set in the `PatokProcessor.__init__()` constructor, so they apply when creating a processor instance.

**Process:**
Each iteration performs two phases:
1. **Expand Phase**: Splits tokens, avoiding affix tokens when possible
2. **Contract Phase**: contractions token pairs, preferring contractions that form affixes

### 3. **Selective Expansion (`_selective_expand`)**
- Identifies expandable tokens (those with valid splits in vocabulary)
- Preferentially avoids expanding tokens that are already affixes
- If only affix tokens are expandable, expands them as fallback
- Ensures all splits are valid according to tokenizer vocabulary

### 4. **Affix-Preferring Contraction (`_affix_preferring_contract`)**
- Scans all adjacent token pairs for possible contractions
- Categorizes contractions into:
  - **Affix-forming**: contractions that result in an affix token
  - **Regular**: Other valid contractions
- Uses `affix_preference` probability to choose:
  - High probability → prefer affix-forming contractions
  - Low probability → random contractions
- All contractions are guaranteed valid (exist in tokenizer vocabulary)

## Validity Guarantees

✅ **All tokens remain valid** throughout the process:
- Expansions use `self.expansions` dict (built from tokenizer's contraction operations)
- Contractions use `self.contractions` dict (tokenizer's actual contraction rules)
- No invalid tokens can be created

✅ **Text is decodable** at every step:
- Each token ID maps to a valid vocabulary entry
- The sequence can always be decoded back to text

## Example Usage

```python
# Initialize
tokenizer = tiktoken.get_encoding("gpt2")
processor = PatokProcessor(
    tokenizer=tokenizer,
    affixes_file="data_other/filipino_affixes.txt"
)

# Process tokens
token_ids = tokenizer.encode("Kumakain ako ng pagkain")
processed_ids = processor.affix_aware_expand_contract(
    token_ids,
    num_iterations=3,
    expand_prop=0.2,
    contract_prop=0.2,
    affix_preference=0.8  # 80% chance to prefer affix-forming contractions
)
```

## Parameters Tuning Guide

- **`num_iterations`**: More iterations = more opportunities to form affixes
  - Recommended: 2-5 for balanced processing
  - Higher values increase processing time

- **`expand_prop` / `contract_prop`**: Control how much change occurs
  - 0.1 = 10% of tokens affected per iteration (conservative)
  - 0.3 = 30% of tokens affected per iteration (aggressive)
  - Balance these to control final sequence length

- **`affix_preference`**: Controls affix formation likelihood
  - 0.5 = no preference (random)
  - 0.7-0.8 = moderate preference (recommended)
  - 0.9-1.0 = strong preference (may be too restrictive)

## Implementation Benefits

1. **Linguistic Awareness**: Prefers linguistically meaningful tokens (affixes)
2. **Vocabulary Safety**: All operations guaranteed valid by tokenizer
3. **Stochastic Variety**: Multiple runs produce different but valid results
4. **Configurable**: Tunable parameters for different use cases
5. **Reversible**: Can decode tokens at any point in processing
