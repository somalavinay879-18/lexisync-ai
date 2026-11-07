# OCR benchmark

Document: gdpr.pdf, Tesseract, 200 dpi.

| page | characters | character accuracy |
|---|---|---|
| 1 | 2703 | 94.9% |
| 18 | 5099 | 97.7% |
| 35 | 3252 | 96.1% |
| 52 | 3464 | 96.5% |
| 69 | 2901 | 95.8% |

**Mean character accuracy: 96.2%**

Fallback: if Tesseract is not installed, scanned pages are flagged in the structure view and skipped instead of producing wrong text.
