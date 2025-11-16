"""
Test script for affix-aware token processing
"""
import tiktoken
from patok_processor import PatokProcessor

def test_affix_processing():
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Initialize processor
    processor = PatokProcessor(
        tokenizer=tokenizer,
        expand_prop=0.5,
        contract_prop=0.5,
        affixes_file="data_other/filipino_affixes.txt"
    )
    
    # Test text with Filipino affixes
    test_texts = [
        "Maganda ang panahon ngayon",
        "Kumakain ako ng pagkain",
        "Naglalakad siya sa kalye",
        "Makigpagsayawan naman kayo sa isa't isa",
        "Nasa huli ang pagsisisi ng sambayanang Pilipino"
    ]
    
    print("\n" + "="*80)
    print("Testing Affix-Aware Token Processing")
    print("="*80)
    
    for text in test_texts:
        print(f"\nOriginal text: '{text}'")
        
        # Tokenize
        token_ids = tokenizer.encode(text)
        print(f"Original tokens ({len(token_ids)}): {token_ids}")
        print(f"Decoded: {[tokenizer.decode([tid]) for tid in token_ids]}")
        
        # Process with affix awareness
        processed_ids = processor.affix_aware_expand_contract(token_ids)
        
        print(f"\nProcessed tokens ({len(processed_ids)}): {processed_ids}")
        print(f"Decoded: {[tokenizer.decode([tid]) for tid in processed_ids]}")
        
        # Check which tokens are affixes
        affix_tokens = [tid for tid in processed_ids if tid in processor.affix_token_ids]
        if affix_tokens:
            print(f"Affix tokens found: {[tokenizer.decode([tid]) for tid in affix_tokens]}")
        
        # Verify the sequence is still valid
        try:
            reconstructed = tokenizer.decode(processed_ids)
            print(f"Reconstructed text: '{reconstructed}'")
        except Exception as e:
            print(f"ERROR: Could not decode processed tokens: {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_affix_processing()
