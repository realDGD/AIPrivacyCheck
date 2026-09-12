# Third-Party Licenses and Attributions

This project interfaces with or downloads the following third-party models and components upon user request:

## 1. Upstream Model Weights (ModelScope)

The model weights listed below are distributed under their respective upstream licenses. AIPrivacyCheck does not sub-license or redistribute the upstream model weights:

- **MemPrivacy Models**
  - Identifiers: `memprivacy-1.7b-rl`, `memprivacy-4b-rl`
  - Upstream Repository: `MemTensor/MemPrivacy-1.7B-RL`, `MemTensor/MemPrivacy-4B-RL`
  - Upstream Authors: MemTensor
  - License: **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**
  - Note: Downloaded on-demand or imported by the user into the local instance.

- **GLiNER Models**
  - Identifiers: `gliner-pii-edge`, `gliner-pii-base`
  - Upstream Repository: `knowledgator/gliner-pii-edge-v1.0`, `knowledgator/gliner-pii-base-v1.0`
  - Upstream Authors: Knowledgator
  - License: **Apache-2.0 (Apache License 2.0)**

- **SiameseUIE Model**
  - Identifier: `siamese-uie`
  - Upstream Repository: `iic/nlp_structbert_siamese-uie_chinese-base`
  - Upstream Authors: Alibaba DAMO Academy / ModelScope
  - License: **Apache-2.0 (Apache License 2.0)**

## 2. AIPrivacyCheck Semantic Privacy Extraction Prompt

- Component: `AIPRIVACY_SEMANTIC_EXTRACTION_PROMPT`
- Author: AI Privacy Check Contributors
- License: **Apache-2.0 (Apache License 2.0)**
- Note: Independently authored for functional JSON extraction of privacy-sensitive spans within AIPrivacyCheck. It does not incorporate upstream MemPrivacy prompt taxonomy, examples, or wording. Consequently, upstream published benchmark scores are not directly transferable.
