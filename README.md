
# White Paper on Advanced Transformer Architectures for Real-Time Translation

## Abstract

This  white paper explores advanced techniques in transformer-based machine translation architectures, emphasizing simultaneous machine translation (SiMT). We introduce the `SiMTTransformerDecoderLayer` which extends traditional transformer models by integrating language model outputs, and the `SiMT` strategy for dynamic input processing during translation. These innovations aim to optimize the balance between translation accuracy and latency, crucial for real-time applications.

## Introduction


Transformer models are pivotal in natural language processing, particularly for translation and summarization. The `SiMTTransformerDecoderLayer` and the `SiMT` strategy enhance these models to address the unique demands of real-time translation, such as in live subtitling or during multilingual meetings.

## SiMT Strategy for Dynamic Input Processing

### Methodology

#### Context Extraction and Dynamic Input Handling

`SiMT` dynamically controls how much source input the model processes at any step, critical for real-time translation:
   
- **Token-level Context**: The number of source tokens considered before generating a target token.
- **Word-level Context**: Useful for languages without clear token delimiters, considering the number of source words.

**![](https://lh7-us.googleusercontent.com/UAhgIPUOxPBNQSoj8y6-qNpCFnl51v6S39frEnzrOBT-Ug9Osgw3dUzzYILkDmwMSVW1r5xPWGoe9_55LHGr6wjDk9qyMmhVw8w1FomvKwEvwSo54Zcmp1oFi31wthgPRRVGdZSnOIjnRwDcnjogcv5b5w=s2048)** 

Real-time requires information processing within the sentences and mapping tokens/words to other tokens. Above is an example of these policies. 

## Proposed Methods

#### Preliminaries

Given a source sentence $x = (x_1, x_2, ..., x_n)$, the aim of a SiMT model is to generate a target sentence $y = (y_1, y_2, ..., y_m)$ while minimizing latency metrics. The SiMT model’s policy, denoted as $g_i$, dictates the number of source tokens processed before predicting each target token $y_i$. The probability of generating $y$ given $x$ is formulated as:
$$p(y|x) = \prod_{i=1}^{|y|} p(y_i | x \leq g_i, y < i; \theta)$$
where $\theta$ represents the model’s parameters, typically optimized using a cross-entropy loss.

The Transformer encoder-decoder model is the prevalent architecture for SiMT. To enhance efficiency, the encoder is often modified to encode the source tokens unidirectionally. More advanced encoding techniques, such as the Recurrent Linear Transformer or Partial Bidirectional Encoding, further augment encoding capabilities.

It is documented that the evaluation of SiMT systems typically employs latency metrics applied to encoded source tokens, which are subject to varying tokenization and encoding protocols. This variation results in disparate token sequences being compared across different systems when analyzing the same dataset, thus complicating the task of performing consistent cross-system evaluations.

To resolve this inconsistency, we introduce a methodology for computing word-level latency scores that account for the boundaries inherent in token-level sequences. We define the initiation of a READ operation at the onset of the first token of a source word as the commencement of reading the entire word. Correspondingly, the completion of writing a target word is marked by a WRITE operation on the last token of that word. This method ensures uniformity in latency calculations across different tokenization and encoding strategies, thereby standardizing comparisons across systems.

  

### Implementation of Word-Level SiMT Policies

  

Our proposed word-level policy mandates that transitions between READ and WRITE operations within a SiMT policy occur strictly at word boundaries. This policy can be derived from  any token-level policy by employing the following transformation process.

  

Specifically, we implement a word-level READ by delaying the index $g_i$ until it coincides with the nearest boundary of a source word. The refined index $r_i$, representing a modification of $g_i$, is defined mathematically by:

$$ r_i := \min \{ j \mid j \geq g_i \wedge j \in BS \} $$

where $BS$ indicates the indices at which source words terminate. By substituting $r_i$  for  $g_i$, the policy adheres to the original decision-making criteria while ensuring that the reading of an entire word is uninterrupted upon the start of a token read.

  

Similarly, for word-level WRITE operations, the adjustment ensures that writing actions conclude at the end of a word. We adjust $r_i$ to continue until a token indicating the end of a word is produced. This adjustment is defined as:

$$ w_i := ( r_i, \text{ if } i = 1 \lor b_{i-1} \neq b_i; w_{i-1}, \text{ otherwise} ) $$

where $b_i$  is defined as:

$$ b_i := \min \{ j \mid j \geq i \wedge j \in BT \} $$

Here, $BT$ denotes the indices where target words conclude, and  $b_i$ marks the index of the last token within the word containing $y_i$. Employing $w_i$ instead of $r_i$ (or  $g_i$) ensures that the policy enables comprehensive word formation without interruptions from READ actions. This approach not only maintains linguistic integrity but also potentially reduces latency by allowing faster completion of certain tokens compared to the base policy, compensating for any increase in latency due to the word-level READ.


## Enhanced Transformer Decoder Layer: `SiMTTransformerDecoderLayer`

### Utilizing Pre-trained Language Models for Machine Translation

The advent of Transformer-based language models (LMs), pre-trained on extensive text corpora, has catalyzed their application to a variety of downstream NLP tasks. These models have been particularly impactful in the domain of machine translation (MT), where their integration into neural machine translation (NMT) systems has proved to be effective. Encoder-only LMs have enhanced encoder representations through gating mechanisms, while more comprehensive integrations have included attention mechanisms that link both encoder and decoder components with pre-trained models like BERT.

Further research has been directed towards developing LMs that are specifically tailored for NMT applications. These models have shown significant improvements, especially in scenarios involving low-resource language pairs. Optimization strategies for these LMs in MT contexts have focused on fine-tuning specific components of the LMs, reducing domain mismatch, and mitigating copying behaviors.

### Integration of Pre-trained LMs into SiMT

The exploration of pre-trained LMs within simultaneous machine translation (`SiMT`) systems is still in its nascent stages. Previous attempts have enhanced mixed-model architectures (MMA) by leveraging the LM's capability to predict the next target token, albeit at the cost of semantic coherence due to token fragmentation. This approach has predominantly focused on integrating target-side LMs, with potential benefits of source-side LM integration remaining largely untapped.

This paper proposes an innovative approach by integrating a source-side LM into `SiMT` systems. The approach builds on established methods in offline MT, introducing key modifications with a focus on word-level policies. The effective management of vocabulary mismatches between the LM and the `SiMT` model is critical, and this is addressed through the implementation of word-level `SiMT` policies. This integration aims to leverage the predictive power of LMs while maintaining the semantic integrity of the translation output, thus enhancing the overall efficacy of the `SiMT` systems.

Because some of these models are big, we used 16 A100 GPUs with 80GB of VRAM and trained for ~30,000 epochs. 

#### Layer Architecture

The `SiMTTransformerDecoderLayer` builds upon the standard transformer decoder layer by incorporating three key components:

- **Self-Attention Mechanism**: Focuses on different positions within the input sequence to aggregate contextual information.
- **Encoder-Decoder Attention**: Integrates source sequence information by attending to encoder outputs.
- **Language Model Attention**: Utilizes outputs from an external language model to enrich the translation with broader contextual details.

#### Attention Mechanisms Explained

1. **Self-Attention**:
   $$\text{SelfAttn}(X) = \text{Attention}(Q=X, K=X, V=X)$$
   Here, $X$ represents the input embedding matrix, highlighting the decoder's ability to reference its prior outputs.

2. **Encoder-Decoder Attention**:
   $$\text{EncDecAttn}(X, E) = \text{Attention}(Q=X, K=E, V=E)$$
   $E$ denotes the encoder outputs, facilitating the integration of the source language context.

3. **Language Model Attention**:
   $$\text{LMAttn}(X, L) = \text{Attention}(Q=X, K=L, V=L)$$
   $L$ is the output from the external language model, providing supplementary context that enhances translation quality.

#### Attention Combination Strategy

The integration of outputs from both encoder-decoder and language model attention mechanisms can be adjusted dynamically using dropout techniques:
   
- **Static Mixing**: Proportions are fixed by`encoder_ratio` and `lm_ratio`.
- **Dynamic Dropout Mixing**: Adjusts the mix based on training conditions, enhancing model robustness and adaptability.
- other various efficient attention mechanisms, critical for managing computational resources in real-time translation tasks (e.g., sparse, low-rank, or block-sparse attention), optimizing computational efficiency.

$$ Attention(Q, K, V) = softmax(\frac{AttOp(Q, K)}{sqrt(dk)}) \times V$$

Where:
- $AttOp(Q, K)$ could be any above operation
- $M$ is an optional mask applied to attention scores, accommodating sequence-specific constraints like padding or causal dependencies.
- $dk$ (dimensionality of keys) is used as a scaling factor to stabilize gradients during training.

### Quantization in Multihead Attention

Quantization enhances model efficiency and robustness by reducing the precision of the weights:
   
$$ Q(w)=round(w/Δ)×Δ+ϵ $$

Where $w$ is the original weight, $Δ$ is the quantization scale, and $ϵ$ represents the noise added to improve robustness and potentially avoid overfitting by introducing stochasticity into the weight representation.

## Evaluation Metrics

During `SiMT` system evaluations, translation quality is assessed alongside the latency required for generating translations. Various metrics have been proposed for latency, with Average Lagging (AL) being the most commonly used metric.

This section enriches our understanding of the current landscape in leveraging pre-trained LMs for machine translation, specifically within the realm of `SiMT`, providing a strong foundation for future advancements in real-time translation systems.

#### Performance Metrics

##### Average Lagging Calculation

The `get_al` function calculates average lagging, a critical metric for optimizing `SiMT` parameters:
   
$$\text{Average Lagging} = \frac{\sum_{t=1}^{T} (C_t - \frac{t}{\gamma})}{T}$$

Here, $C_t$ is the context at time $t$, $\gamma$ is a normalization factor, and $T$ is the total tokens or words translated.

##### Delay and Quality Metrics

Performance is evaluated on:
   
- **Token Lagging (TAL)**
- **Word Lagging (WAL)**
- **BLEU Score**: Assessing translation quality against reference translations.

---
| Model            | Language Pair | Token AL     | Word AL      | BLEU        |
|------------------|---------------|--------------|--------------|-------------|
| SiMT Only        | EN-FR         | 4.189568556  | 3.330422804  | 35.91683003 |
| SiMT Only        | FR-EN         | 3.911838345  | 2.851635377  | 34.95073872 |
| XGLM+SiMT        | EN-FR         | 4.197559222  | 3.31459377   | 37.74441305 |
| XGLM+SiMT        | FR-EN         | 3.883420777  | 2.822404448  | 35.36649502 |
| Llama3+SiMT      | EN-FR         | 4.088925497  | 3.265113931  | 34.73498433 |
| Llama3+SiMT      | FR-EN         | 3.912389712  | 2.902345892  | 35.12093819 |
| SiMT Only        | EN-KO         | 3.4627043966 | 3.3543437852 | 23.13078565  |
| SiMT Only        | KO-EN         | 7.596031253  | 4.869157186  | 24.965637574 |
| XGLM+SiMT        | EN-KO         | 4.49435285   | 4.123016377  | 23.056010005 |
| XGLM+SiMT        | KO-EN         | 6.817742537  | 4.33638808   | 22.71229535 |
| Llama3+SiMT      | EN-KO         | 4.107180372  | 4.8444595116 | 22.986324411 |
| Llama3+SiMT      | KO-EN         | 7.367142994  | 4.695905045  | 25.100826219 |
| State of the Art - [CMU](https://aclanthology.org/2023.iwslt-1.20.pdf)| -        | - | 5.67 | 26.7 |
| State of the Art - [Baidu](https://aclanthology.org/2021.autosimtrans-1.2.pdf) | -         | - | 7.467 | 19.45 |
| State of the Art - [Xiaomi](https://aclanthology.org/2022.iwslt-1.17.pdf) | -         | - | 11.2 | 19.8 |

**![](https://lh7-us.googleusercontent.com/dx8mNylvVmvY7Z9uPsLoY3weHh1pAFEPLu36vPdrW-tKQf8TzxrYgMArozM0mvM7Jm0EcBBBWyWbBMNWF6Q6yrgWS90hm0rcFesfp_K3HyN3j4HL3rYc_TxDYbXi_zwwOu139nOKc20Q6446Fpv9w3zxmw=s2048)**
## Conclusion

The integration of the `SiMTTransformerDecoderLayer` and the `SiMT` strategy represents a significant advancement in transformer architecture, tailored for the demands of real-time translation. These methods improve both the contextual richness and the responsiveness of translation systems, essential for accurate and timely communication across languages.


## Appendix
### Examples:

 ID    | Source (S)                                   | Target Reference (T)                | Hypothesis (H)                         |
|-------|---------------------------------------------|-------------------------------------|----------------------------------------|
| S-897  | This is really cool."                       | 정말 멋진 걸." 이라고요.            | 이것은 정말 멋져요."                  |
| S-2351 | These changes are coming.                   | 이런 변화들이 다가오고 있습니다.       | 이런 변화들이 오고 있습니다.       |
| S-2383 | You could see Manhattan.                    | 맨하탄이 보였죠.                    | 여러분은 맨하탄도 볼 수 있습니다.      |
|S-5602 | And because the engineers that I've worked with have taught me to become really in touch with my inner nerd, I want to summarize with an equation. Take your science, subtract your bullet points and your jargon, divide by relevance, meaning share what's relevant to the audience, and multiply it by the passion that you have for this incredible work that you're doing, and that is going to equal incredible interactions|Et parce que les ingénieurs avec qui j'ai travaillé m'ont appris à être en contact avec le "nerd" qui est en moi. j'aimerais tout résumer par une équation. Prenez votre science, soustrayez vos listes à puces et votre jargon, divisez par la pertinence, c'est-à-dire, partagez ce qui est pertinent pour l'auditoire, et multipliez par la passion que vous avez pour le travail incroyable que vous faites, et vous obtiendrez des interactions incroyables |Et parce que les ingénieurs que j'ai travaillé avec mont appris à devenir vraiment en contact avec mon "inner nerd", je veux résumer avec une équation: Prenez votre science, soustrayez vos listes à puces et vos points de jargon, divisez par la pertinence, partager ce qui est pertinent pour le public, et multipliez par la passion que vous avez pour ce travail incroyable que vous faites, et tout cela équivaut à des interactions incroyables 
| S-2537 | That's right, some lucky individual, corporation, for-profit or non-profit, was going to get the once-in-a-lifetime opportunity -- because I'm sure Chris Anderson will never let it happen again -- to buy the naming rights to the talk you're watching right now, that at the time didn't have a title, didn't really have a lot of content and didn't really give much hint | 
C'est vrai, un individu chanceux, une corporation, à but lucratif ou à but non lucratif, allait avoir la chance de toute une vie - parce que je suis sûr que Chris Anderson ne laissera pas cela se reproduire - pour acheter les droits d'appellation de la conversation que vous regardez en ce moment, qui à l'époque, n'avait pas de titre, n'avait pas vraiment beaucoup de contenu et n'avait pas vraiment donné beaucoup d'indices | 
C'est vrai, certains d'entre eux ont la chance pour un but profit ou non lucratif, allaient recevoir l'opportunité d'une vie -- parce que je suis sûr que Chris Anderson ne l'arrivera jamais -- pour acheter les droits de la conférence vous regardez maintenant, que à l'époque où vous l'avez vu n'avait pas un titre, n'avez pas vraiment beaucoup de contenu et ne leur donnez pas vraiment d'indices.
---


**![](https://lh7-us.googleusercontent.com/7O2sbvn6IGB5gOAGYUiCeOD-zagCUyzEx9cVQcBNFX_rms22QqmBvPmz0WgcuCO9aK0YTVfY6tPP3lHP5b1xKe7ufM_qEGpcNnWpGVxUOqHvyuEcQz4ynKO6mYDpTyJgE_yu6YOep0ch1MZbVZrNRFT7Iw=s2048)**
