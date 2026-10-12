# Nebius organizer request — Token Factory access without a bank card

**Why this is needed.** The submission runs on Nebius Token Factory. Token
Factory onboarding requires a billing account with a bank card, and the $1 trial
plus the AI Builder Program's $25 credit do **not** bypass that (both still
require a card on file). As an individual participant without a card, we cannot
create a key self-serve. We need either a sponsored/promo key or explicit
guidance.

## Where to post

1. **Nebius Discord** — the hackathon channel (fastest; organizers are active).
2. **Devpost** — message the hackathon organizers / post in the discussion forum.
3. **Email** — the hackathon contact address if one is listed.

## Version A — Discord (short)

> Hi Nebius team / hackathon organizers 👋
>
> I'm participating in the Nebius × NVIDIA Global AI Hackathon (Best Apps &
> Agents track) as an **individual participant**. My project uses **Tavily +
> NVIDIA Nemotron on Nebius Token Factory**, so the whole submission depends on
> being able to run on Token Factory.
>
> I tried to sign up for Token Factory and join the AI Builder Program to claim
> the $25 credit, but onboarding requires a billing account with a **bank card**,
> which I don't have as a solo/student participant. The $1 trial and the Builder
> credit both still require a card on file.
>
> **Is there a card-free way for hackathon participants to get a Token Factory
> API key** — e.g. a sponsored/promo-code account, a hackathon credit, or a
> temporary key? Any guidance would be hugely appreciated.
>
> Thanks!
> Repo: https://github.com/Mxhesh-07/nemotron-data-forge

## Version B — Devpost / email (formal)

> **Subject:** Token Factory API key for hackathon participant without a bank card
>
> Hi Nebius × NVIDIA hackathon organizers,
>
> I'm an individual participant in the **Best Apps & Agents** track. My
> submission, *Nemotron Data Forge*, uses **Tavily + NVIDIA Nemotron models on
> Nebius Token Factory**, so running on Token Factory is central to the project.
>
> I attempted to create Token Factory access and to join the AI Builder Program
> to claim the $25 credit, but onboarding requires a billing account with a bank
> card. As a solo participant without a card, I can't complete that step, and the
> $1 trial / Builder credit don't remove the card requirement.
>
> Could you advise on a card-free path for participants — for example:
> - a sponsored or promo-code Token Factory account for hackathon participants, or
> - a temporary/evaluation API key for the submission period?
>
> I'm happy to provide any details you need. Thank you very much — the whole
> submission hinges on this.
>
> Best regards,
> Mahesh
> GitHub: https://github.com/Mxhesh-07/nemotron-data-forge

## If a direct answer is slow — alternatives to raise in the same message

- **Teammate/colleague card.** If a teammate can add a card to a shared Nebius
  account, that unblocks Token Factory immediately. (Note: the hackathon allows
  teams; if you can pair with someone who has a card, do so.)
- **Organizer-provisioned account.** Ask whether they can provision a key
  directly for the track.
- **Explicit written allowance.** Ask whether, if truly no card is possible,
  NVIDIA NIM is acceptable as a documented fallback — get this **in writing**
  before relying on it in the submission.

## What to include so they can act fast

- Track: Best Apps & Agents
- Project: Nemotron Data Forge (Streamlit app; Tavily + Nemotron on Token Factory)
- Repo: https://github.com/Mxhesh-07/nemotron-data-forge
- Constraint: individual participant, no bank card, cannot complete billing onboarding
- Deadline pressure: submission closes Oct 30, 2026, 10:00 AM PT

## While waiting

Keep developing against `LLM_PROVIDER=nvidia` (the free NIM dev fallback, same
Nemotron models, no card) so every other part of the project — tests, UI, the
LoRA demo, the video's non-live shots — is finished. Flip `.env` to
`LLM_PROVIDER=nebius`, run `--check`, and re-record the live segments as soon as
a Token Factory key arrives.

