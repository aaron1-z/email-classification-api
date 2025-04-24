# Email Classification and PII Masking API

This project implements an API that classifies incoming support emails into predefined categories while masking Personal Identifiable Information (PII) *before* classification, using non-LLM methods (Regex).

## Features

*   **PII Masking:** Detects and masks Full Name, Email Address, Phone Number, Date of Birth, Aadhar Number, Credit/Debit Card Number, CVV, and Card Expiry using Regular Expressions.
*   **Email Classification:** Classifies the masked email content using a TF-IDF + Logistic Regression model trained on the provided dataset.
*   **API:** Exposes the functionality via a FastAPI endpoint (`/classify/`).
*   **Strict Output Format:** Adheres to the specified JSON output structure for automated evaluation.
![WhatsApp Image 2025-04-24 at 16 48 21_4f78bc15](https://github.com/user-attachments/assets/6c0979ae-8d08-458f-8f16-83f6076d2168)

![WhatsApp Image 2025-04-24 at 16 49 29_f9021bb6](https://github.com/user-attachments/assets/507d4d32-e515-4b34-bd51-508987df6e0b)

![WhatsApp Image 2025-04-24 at 16 50 04_933da1f1](https://github.com/user-attachments/assets/48894e75-1a11-4a6f-af26-0270c88473fa)

![WhatsApp Image 2025-04-24 at 16 50 33_c88f5068](https://github.com/user-attachments/assets/4c103854-9618-45c3-b34c-22fbf74b6e4f)

![WhatsApp Image 2025-04-24 at 16 51 52_897d92fa](https://github.com/user-attachments/assets/57ba5e56-78da-40ef-ad16-ef4e748c0363)

![WhatsApp Image 2025-04-24 at 16 52 24_1258217b](https://github.com/user-attachments/assets/5609e5df-0c0d-4b3d-b319-4c785dc24dd0)





