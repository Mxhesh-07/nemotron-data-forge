# Dataset Quality Report

- **Topic:** Indian tax law for freelancers
- **Format:** Q&A
- **Generated:** 2026-10-12 23:14
- **Generator model:** `nvidia/nemotron-3-super-120b-a12b`
- **Grader model:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Embeddings:** `nvidia/nemotron-3-embed-1b`

## Summary

| Metric | Value |
| --- | --- |
| Sources used | 4 |
| Examples generated | 12 |
| Examples graded | 12 |
| Examples kept | 9 |
| Rejected (not kept) | 3 |
| Duplicates removed | 3 |
| Rewritten (on reject) | 0 |
| Recovered by rewrite | 0 |
| Avg accuracy (kept) | 9.78 |
| Avg helpfulness (kept) | 8.89 |
| Avg clarity (kept) | 8.89 |
| Estimated cost (USD) | 0.006011 |
| Total tokens | 6697 |
| Duration (s) | 14.9 |

## Token usage

| Model | Calls | Prompt | Completion | Total |
| --- | --- | --- | --- | --- |
| `nvidia/nemotron-3-super-120b-a12b` | 4 | 1969 | 787 | 2756 |
| `nvidia/nemotron-3-ultra-550b-a55b` | 4 | 2594 | 704 | 3298 |
| `nvidia/nemotron-3-embed-1b` | 1 | 643 | 0 | 643 |

## Sources

- [Income Tax Filing For the Freelancers | Policybazaar](https://www.policybazaar.com/income-tax/income-tax-filing-for-freelancers)
- [Freelance and taxes: India | Mellow](https://help.mellow.io/en/articles/9134949-freelance-and-taxes-india)
- [Freelancer Income Tax in India: A Practical Guide](https://www.xflowpay.com/blog/freelancer-income-tax-india)
- [How Gig Workers & Freelancers Are Taxed in India ?](https://www.kotaklife.com/insurance-guide/savingstax/tax-obligations-for-freelancers-and-gig-economy-professionals)

## Kept examples

### 1. What is the tax rate under Section 44AD for small businesses with annual income of INR 20 million or below if all transactions are made digitally?

Under Section 44AD, if all transactions are made digitally, the tax amounts to 6% of annual turnover.

_accuracy 10.0 · helpfulness 9.0 · clarity 9.0 · [source](https://help.mellow.io/en/articles/9134949-freelance-and-taxes-india)_

### 2. What does Section 44ADA allow for professional income not exceeding INR 5 million per year?

Section 44ADA allows the taxpayer to pay tax on 50% of the actual income, with the state deeming the other 50% as expenses, using a standard progressive rate.

_accuracy 10.0 · helpfulness 9.0 · clarity 9.0 · [source](https://help.mellow.io/en/articles/9134949-freelance-and-taxes-india)_

### 3. What is the monthly tax amount per vehicle under Section 44AE for goods carriage entities operating no more than 10 vehicles?

Section 44AE provides for a monthly tax of INR 7,500 per vehicle.

_accuracy 10.0 · helpfulness 9.0 · clarity 9.0 · [source](https://help.mellow.io/en/articles/9134949-freelance-and-taxes-india)_

### 4. What is the minimum annual income threshold for Indian freelancers to be liable for income tax?

Every Indian earning more than ₹3,00,000 is liable to pay income tax annually, and this rule applies to freelancers and gig workers.

_accuracy 10.0 · helpfulness 9.0 · clarity 9.0 · [source](https://www.kotaklife.com/insurance-guide/savingstax/tax-obligations-for-freelancers-and-gig-economy-professionals)_

### 5. What is the TDS rate applicable to freelancers in India, and from which types of payments is it deducted?

TDS is deducted at the rate of 10% from payments such as salary, commission, rent, interest, or professional fees.

_accuracy 8.0 · helpfulness 8.0 · clarity 8.0 · [source](https://www.kotaklife.com/insurance-guide/savingstax/tax-obligations-for-freelancers-and-gig-economy-professionals)_

### 6. Under what condition can freelancers in India be subject to a tax audit, and how does this affect TDS applicability?

Freelancers can only be audited if their annual gross receipt exceeds ₹50 lakhs, and in such a scenario, TDS will not be applicable.

_accuracy 10.0 · helpfulness 9.0 · clarity 9.0 · [source](https://www.kotaklife.com/insurance-guide/savingstax/tax-obligations-for-freelancers-and-gig-economy-professionals)_

### 7. Do foreign clients deduct TDS on payments to Indian freelancers?

No. Only Indian payers who are liable to deduct do so.

_accuracy 10.0 · helpfulness 9.0 · clarity 9.0 · [source](https://www.xflowpay.com/blog/freelancer-income-tax-india)_

### 8. What is the effective zero-tax ceiling for resident individuals under the new tax regime for FY2025-26?

The effective zero-tax ceiling is ₹12 lakh of taxable income for resident individuals under the new tax regime for FY2025-26, due to the Section 87A rebate.

_accuracy 10.0 · helpfulness 9.0 · clarity 9.0 · [source](https://www.xflowpay.com/blog/freelancer-income-tax-india)_

### 9. Under which category is freelancing income classified for income tax purposes in India?

Freelancing income falls under the category of 'Profits and Gains from Business or Profession' according to income tax laws in India.

_accuracy 10.0 · helpfulness 9.0 · clarity 9.0 · [source](https://www.policybazaar.com/income-tax/income-tax-filing-for-freelancers)_

## Rejected examples

| # | Instruction | Avg score | Reason |
| --- | --- | --- | --- |
| 1 | When must a freelancer pay advance tax during the year? | 9.33 | near-duplicate of kept example #4 (sim=0.84) |
| 2 | What is TDS and how can freelancers benefit from it when filing their income tax | 9.33 | near-duplicate of kept example #5 (sim=0.83) |
| 3 | Which section allows freelancers to claim a deduction for health insurance premi | 9.33 | near-duplicate of kept example #4 (sim=0.84) |
