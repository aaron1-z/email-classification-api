# Email Classification and PII Masking API

This project implements an API that classifies incoming support emails into predefined categories while masking Personal Identifiable Information (PII) *before* classification, using non-LLM methods (Regex).

## Features

*   **PII Masking:** Detects and masks Full Name, Email Address, Phone Number, Date of Birth, Aadhar Number, Credit/Debit Card Number, CVV, and Card Expiry using Regular Expressions.
*   **Email Classification:** Classifies the masked email content using a TF-IDF + Logistic Regression model trained on the provided dataset.
*   **API:** Exposes the functionality via a FastAPI endpoint (`/classify/`).
*   **Strict Output Format:** Adheres to the specified JSON output structure for automated evaluation.
