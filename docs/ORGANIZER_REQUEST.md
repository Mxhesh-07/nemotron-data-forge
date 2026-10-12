# Nebius organizer request — Token Factory access without a bank card

**Why this is needed.** The submission must run on **Nebius Token Factory or
Nebius AI Cloud** (Official Rules) and use at least one NVIDIA open-source model.
As a solo participant without a bank card we cannot finish the default Token
Factory signup — billing onboarding asks for a card, and the plain **$1 trial** /
**AI Builder Program** page do not obviously bypass it. Before asking organizers
for a key, exhaust the **official hackathon credit routes** below.

## Try these first — official hackathon credit routes

Straight from the hackathon **Resources** page
(<https://nebiusglobalaihackathon.devpost.com/resources>):

1. **$25 Token Factory credits — promo form.** Fill out
   <https://nebius.com/promo-code?utm_promo_event_code=2026-devpost-global-ai-hack&utm_promo_code_type=Token_Factory&utm_promo_activation_code=NEBIUS-DEVPOST-GLOBAL26>
   using activation code **`NEBIUS-DEVPOST-GLOBAL26`**, then redeem the code in the
   Token Factory console. Watch the inbox and spam folder.
2. **Another $25 Token Factory credits — Nebius Builders Program.**
   <https://dev.nebius.com/builders> (free). Community guides report **Gmail
   addresses are blocked** on this form, so use a work/school/other address.
   Sign-up can emit two emails — only the **"Here's your promo code"** (Token
   Factory) one matters; ignore a **"promo code exhausted"** (AI Cloud) message.
3. **Nebius Discord (current invite).** <https://discord.com/invite/eXYTGhgnhK> —
   where Devpost staff and the Nebius team answer access questions, and where
   organizers can hand out a code directly if the forms dead-end.
4. **Builders & Brews (in person).** <https://luma.com/builderandbrews> —
   attendees unlock extra AI Cloud / Token Factory / Tavily credits. (Most 2026
   cities have passed; LA is/was Oct 13.)

**Card reality check.** The current Token Factory billing doc still says billing
onboarding is mandatory and requires a bank card, while also documenting a
**promo-code** path that grants free credits. Reports are mixed on whether a code
fully bypasses the card prompt — hence the fallback ask below.

## Where to post (only if the routes above dead-end)

1. **Nebius Discord** — <https://discord.com/invite/eXYTGhgnhK> (fastest; the
   Nebius team is active).
2. **Devpost** — post in the hackathon discussion forum (Devpost staff reply there).
3. **Email** — the hackathon contact address if one is listed.

## Version 0 — Discord, send this first (shortest)

> Hi Nebius + hackathon organizers 👋
>
> Quick ask: is there a **participant promo / sponsored Token Factory API key** for
> the Nebius × NVIDIA hackathon? I'm a solo participant building on **Tavily +
> Nemotron on Token Factory**. I filled the `NEBIUS-DEVPOST-GLOBAL26` promo form and
> joined the Builders Program, but redeeming still lands on a **bank-card** billing
> step I can't complete.
>
> If a card-free key exists, I can run immediately. If not, is **NVIDIA NIM**
> acceptable as a documented fallback? Thanks!
>
> Repo: https://github.com/Mxhesh-07/nemotron-data-forge

## Version A — Discord (short)

> Hi Nebius team / hackathon organizers 👋
>
> I'm participating in the Nebius × NVIDIA Global AI Hackathon (Best Apps &
> Agents track) as an **individual participant**. My project uses **Tavily +
> NVIDIA Nemotron on Nebius Token Factory**, so the whole submission depends on
> being able to run on Token Factory.
>
> I filled the hackathon promo form (`NEBIUS-DEVPOST-GLOBAL26`) and joined the
> Builders Program to claim the $25 credit, but redeeming still lands on a billing
> account with a **bank card**, which I don't have as a solo/student participant.
> The $1 trial and the Builder credit both still require a card on file.
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
> I filled the hackathon promo form (`NEBIUS-DEVPOST-GLOBAL26`) and joined the
> Builders Program to claim the $25 credit, but redeeming still requires a billing
> account with a bank card. As a solo participant without a card, I can't complete
> that step, and the $1 trial / Builder credit don't remove the card requirement.
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
- **Non-Nebius Nemotron stand-in.** OpenRouter exposes free Nemotron routes
  (`nvidia/nemotron-3-super-120b-a12b:free`, `nvidia/nemotron-3-ultra-550b-a55b`)
  over an OpenAI-compatible API with no card. Dev-only: the rules still require
  Token Factory / AI Cloud and free routes log prompts.

## What to include so they can act fast

- Track: Best Apps & Agents
- Project: Nemotron Data Forge (Streamlit app; Tavily + Nemotron on Token Factory)
- Repo: https://github.com/Mxhesh-07/nemotron-data-forge
- Constraint: individual participant, no bank card, cannot complete billing onboarding
- Deadline pressure: submission closes Oct 30, 2026, 10:00 AM PT

## While waiting

Keep developing against `LLM_PROVIDER=nvidia` (the free NIM dev fallback, same
Nemotron models, no card) so every other part of the project — tests, UI, the
LoRA demo, the video's non-live shots — is finished. If NIM throttles, the free
OpenRouter Nemotron routes are an OpenAI-compatible stand-in. Flip `.env` to
`LLM_PROVIDER=nebius`, run `--check`, and re-record the live segments as soon as
a Token Factory key arrives.

