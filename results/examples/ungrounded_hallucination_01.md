# Ungrounded hallucination 01: refund claimed without any action

- **System:** SCRIBE baseline (`ml-service/baseline/scribe.py`): classify → draft → refine, **no retrieval, no grounding**.
- **LLM:** `openai/gpt-oss-120b` (Groq), reasoning effort low, temperature 0, for both passes.
- **Input:** `datasets/sample_emails.csv` row 1 (`csv-0`), a hand-written sample email.
- **Classification:** Billing via zero-shot (confidence 0.71); sentiment neutral → urgency medium.
- **Run:** 2026-10-15, `ScribeDrafter` over 2 LLM calls (706 tokens). Output copied verbatim (the model's narrow no-break spaces kept).

## Input email

**Subject:** Charged twice for October invoice

> Hi, I was billed twice for invoice INV-2291 this month (two charges of $49 on Oct 3). Can you refund the duplicate charge? Thanks, Priya

## Stage 1: draft

> Hi Priya,
>
> Thank you for reaching out to us.
>
> I’ve reviewed your account and can confirm that invoice INV‑2291 was indeed charged twice on October 3rd for $49 each. I’ve processed a refund for the duplicate $49 charge, and you should see the credit reflected on your payment method within 3–5 business days.
>
> If you have any further questions or notice anything else that needs attention, please don’t hesitate to let us know. We’re here to help!
>
> Best regards,  
> [Your Name]  
> Customer Support Team  
> [Company Name]

## Stage 2: refined reply (what SCRIBE would send)

> Hi Priya,
>
> Thank you for contacting us.
>
> I’ve confirmed that invoice INV‑2291 was charged twice on October 3rd. I’ve processed a refund for the duplicate $49 charge; the credit should appear on your payment method within 3–5 business days.
>
> If you have any other questions, please let us know.
>
> Best regards,  
> [Your Name]  
> Customer Support Team  
> [Company Name]

## What is wrong

| Claim in the reply | Supported by? |
|---|---|
| "I've confirmed that invoice INV-2291 was charged twice" / draft: "I've reviewed your account" | Nothing. The model has no account or billing access; it restates the customer's claim as a verified fact. |
| "I've processed a refund for the duplicate $49 charge" | Nothing. No refund was issued; the reply commits the company to an action that did not happen. |
| "the credit should appear … within 3–5 business days" | Nothing. An invented refund timeline, not taken from any policy document. |
| `[Your Name]`, `[Company Name]` | Unfilled placeholders: the reply cannot be sent as is. |

The refine pass made the reply more polished, and kept every unsupported claim. A grounded system should cite the refund policy (timeline, who approves duplicate-charge refunds) or abstain and route to a human, never assert that a refund has been processed.

More examples are collected automatically in P7.2.
