# utils.py
import re
from typing import List, Dict, Tuple, Any

# Define PII entity types (matches brackets in problem description)
PII_TYPES = {
    "full_name": r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)", # Simple pattern for capitalized names
    "email": r"[\w\.-]+@[\w\.-]+\.\w+",
    "phone_number": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "dob": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b",
    "aadhar_num": r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b",
    "credit_debit_no": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b|\b\d{16}\b", # Basic patterns for 16 digits or 4x4 blocks
    "cvv_no": r"\b\d{3}\b", # Basic 3-digit CVV assumption
    "expiry_no": r"\b(0[1-9]|1[0-2])\/\d{2,4}\b" # MM/YY or MM/YYYY
}

# Reverse mapping for brackets
ENTITY_TYPE_MAP = {v: k for k, v in {
    "full_name": "full_name",
    "email": "email",
    "phone_number": "phone_number",
    "dob": "dob",
    "aadhar_num": "aadhar_num",
    "credit_debit_no": "credit_debit_no",
    "cvv_no": "cvv_no",
    "expiry_no": "expiry_no"
}.items()}

def mask_pii(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Detects and masks PII in the input text using Regex.

    Args:
        text: The original email text.

    Returns:
        A tuple containing:
        - masked_text: The text with PII replaced by placeholders like [entity_type].
        - entities_list: A list of dictionaries detailing found PII.
    """
    masked_text = text
    entities_list = []
    found_spans = [] # Keep track of spans already masked to avoid overlap issues

    # Iterate through PII types and their regex patterns
    for entity_type, pattern in PII_TYPES.items():
        for match in re.finditer(pattern, text):
            start, end = match.span()
            original_value = match.group(0)

            # Basic overlap check: Skip if this span overlaps significantly with an already found span
            is_overlapping = False
            for found_start, found_end in found_spans:
                # Simple overlap check: if start or end falls within another span
                if (found_start <= start < found_end) or \
                   (found_start < end <= found_end) or \
                   (start <= found_start and end >= found_end):
                   # More sophisticated check might be needed for nested entities
                   # For now, we prevent masking substrings of already masked entities
                   is_overlapping = True
                   break
            if is_overlapping:
                continue

             # Add to found spans
            found_spans.append((start, end))

            # Prepare entity detail dictionary
            entity_info = {
                "position": [start, end],
                "classification": entity_type, # Use the internal name
                "entity": original_value
            }
            entities_list.append(entity_info)

            # IMPORTANT: Masking happens *after* finding all entities to ensure indices are correct
            # We'll do the actual replacement in a separate step or carefully manage index shifts

    # Sort entities by start position to mask correctly from left to right (or end to start)
    entities_list.sort(key=lambda x: x['position'][0])

    # Perform masking using placeholders based on the sorted list
    # Iterate backwards to avoid index shifting issues during replacement
    offset = 0
    masked_text_builder = list(text) # Work with a list for easier replacement
    for entity_info in sorted(entities_list, key=lambda x: x['position'][0], reverse=True):
        start, end = entity_info['position']
        entity_type = entity_info['classification']
        placeholder = f"[{entity_type}]"

        # Replace the original slice with the placeholder
        masked_text_builder[start:end] = list(placeholder)

    masked_text = "".join(masked_text_builder)


    return masked_text, entities_list

# Example Usage (for testing)
if __name__ == '__main__':
    example_email = "Hello John Doe, your email is john.doe@example.com and phone (123) 456-7890. Born 1/1/90. Card 1234-5678-9012-3456 expires 12/25, CVV 123. Aadhar 9876 5432 1098."
    masked, entities = mask_pii(example_email)
    print("Original:", example_email)
    print("Masked:", masked)
    print("Entities:", entities)

    example_email_2 = "Call me at 555.123.4567 or reach out to test.user+alias@company.co.uk. My card is 1111222233334444."
    masked_2, entities_2 = mask_pii(example_email_2)
    print("\nOriginal 2:", example_email_2)
    print("Masked 2:", masked_2)
    print("Entities 2:", entities_2)