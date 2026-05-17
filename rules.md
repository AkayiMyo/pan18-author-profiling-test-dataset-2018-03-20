# Myanmar Author Profiling Translation Rules

Use these rules when translating English tweets into Myanmar language for NLP author profiling thesis work using the PAN18 Twitter dataset.

## Translate to Myanmar

- Regular English sentences and words
- Well-known country and city names using Myanmar pronunciation
  - France -> ပြင်သစ်
  - London -> လန်ဒန်
  - Japan -> ဂျပန်

## Keep in English

- Unknown or rare country and city names
- Person names and artist names
- Movie, song, and game titles
- Brand, platform, technology, and medicine names
- Political and technical terms without a Myanmar equivalent
- `@mentions`
- `#hashtags`

## Remove Completely

- All URLs
- All `t.co` links

## Translation Style

- Write like a real Myanmar Twitter user, not a robotic word-for-word translator
- Keep the original tone
  - Short stays short
  - Emotional stays emotional
  - Sarcastic stays sarcastic

## Gender and Person Reference

- `I` should usually be translated as `ငါ` when natural
- `we` and `us` can be translated as `ငါတို့` when natural
- Use `မင်း` instead of `သင်` when addressing a person naturally
- Do not use gender-revealing first-person forms such as:
  - `ကျွန်တော်`
  - `ကျွန်မ`
  - `ကျွန်ုပ်`

- Prefer neutral Myanmar wording when it fits the tone better
- Use `ငါ` / `ငါတို့` only when it sounds natural in a real Myanmar Twitter voice

## Output Format for Each Author

1. `👤 Gender: ကျား / မ / ⚪ ဆုံးဖြတ်၍မရ`
2. `🧠 What kind of person + why` in Myanmar
3. `🏷️ Interests — primary, secondary, other + why` in Myanmar
4. `📝 Translated tweets` with numbering
5. `🔍 Notes` explaining why each word was kept or translated, in Myanmar
6. Do not include `<![CDATA[]]>`

## Display Requirement

- Do not provide both English and Myanmar versions separately
- Translate directly into Myanmar in one pass
- Keep only the final Myanmar text in the output

## Important Notes

- Gender should be treated as neutral in the translation stage unless there is direct evidence in the text
- Do not infer male or female from language style alone
- Keep the dataset useful for classification and author profiling without adding unnecessary gender cues
