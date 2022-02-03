# Awesome BERT & Transfer Learning in NLP [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository contains a hand-curated of great machine (deep) learning resources for Natural Language Processing (NLP) with a focus on Bidirectional Encoder Representations from Transformers (BERT), attention mechanism, Transformer architectures/networks, and transfer learning in NLP.

<br />
<p align="center">
  <img src="https://user-images.githubusercontent.com/145605/79639176-9ca33d80-81bc-11ea-8cde-f7ff68ee2042.png" width="600" />
<h4 align="center">Transformer (BERT) (<a href="https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/bert-encoder">Source</a>)</h4>
</p>
<br />

# Table of Contents

<details>

<summary><b>Expand Table of Contents</b></summary>

- [Papers](#papers)
- [Articles](#articles)
  - [BERT and Transformer](#bert-and-transformer)
  - [Attention Concept](#attention-concept)
  - [Transformer Architecture](#transformer-architecture)
  - [Generative Pre-Training Transformer (GPT)](#generative-pre-training-transformer-gpt)
  - [Additional Reading](#additional-reading)
- [Tutorials](#tutorials)
- [Videos](#videos)
  - [BERTology](#bertology)
  - [Attention and Transformer Networks](#attention-and-transformer-networks)
- [Official Implementations](#official-implementations)
- [Other Implementations](#other-implementations)
  - [PyTorch and TensorFlow](#pytorch-and-tensorflow)
  - [PyTorch](#pytorch)
  - [Keras](#keras)
  - [TensorFlow](#tensorflow)
  - [Chainer](#chainer)
- [Transfer Learning in NLP](#transfer-learning-in-nlp)
- [Books](#books)
- [Other Resources](#other-resources)
- [Tools](#tools)
- [Tasks](#tasks)
  - [Named-Entity Recognition (NER)](#named-entity-recognition-ner)
  - [Classification](#classification)
  - [Text Generation](#text-generation)
  - [Question Answering (QA)](#question-answering-qa)
  - [Knowledge Graph](#knowledge-graph)
</details>

---

## Papers

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai, Zhilin Yang, Yiming Yang, William W. Cohen, Jaime Carbonell, Quoc V. Le and Ruslan Salakhutdinov.
  - Uses smart caching to improve the learning of long-term dependency in Transformer. Key results: state-of-art on 5 language modeling benchmarks, including ppl of 21.8 on One Billion Word (LM1B) and 0.99 on enwiki8. The authors claim that the method is more flexible, faster during evaluation (1874 times speedup), generalizes well on small datasets, and is effective at modeling short and long sequences.
2. [Conditional BERT Contextual Augmentation](https://arxiv.org/abs/1812.06705) by Xing Wu, Shangwen Lv, Liangjun Zang, Jizhong Han and Songlin Hu.
3. [SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering](https://arxiv.org/pdf/1812.03593) by Chenguang Zhu, Michael Zeng and Xuedong Huang.
4. [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever.
5. [The Evolved Transformer](https://arxiv.org/abs/1901.11117) by David R. So, Chen Liang and Quoc V. Le.
  - They used architecture search to improve Transformer architecture. Key is to use evolution and seed initial population with Transformer itself. The architecture is better and more efficient, especially for small size models.
6. [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
  - A new pretraining method for NLP that significantly improves upon BERT on 20 tasks (e.g., SQuAD, GLUE, RACE).
  - "Transformer-XL is a shifted model (each hyper-column ends with next token) while XLNet is a direct model (each hyper-column ends with contextual representation of same token)." — [Thomas Wolf](https://twitter.com/Thom_Wolf/status/1141803437719506944?s=20).
  - [Comments from HN](https://news.ycombinator.com/item?id=20229145):
    <details>
  
    <summary>A clever dual masking-and-caching algorithm.</summary>

    - This is NOT "just throwing more compute" at the problem.
    - The authors have devised a clever dual-masking-plus-caching mechanism to induce an attention-based model to learn to predict tokens from all possible permutations of the factorization order of all other tokens in the same input sequence.
    - In expectation, the model learns to gather information from all positions on both sides of each token in order to predict the token.
      - For example, if the input sequence has four tokens, ["The", "cat", "is", "furry"], in one training step the model will try to predict "is" after seeing "The", then "cat", then "furry".
      - In another training step, the model might see "furry" first, then "The", then "cat".
      - Note that the original sequence order is always retained, e.g., the model always knows that "furry" is the fourth token.
    - The masking-and-caching algorithm that accomplishes this does not seem trivial to me.
    - The improvements to SOTA performance in a range of tasks are significant -- see tables 2, 3, 4, 5, and 6 in the paper.
    </details>
7. [CTRL: Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) by Nitish Shirish Keskar, Richard Socher et al. [[Code](https://github.com/salesforce/ctrl)].
8. [PLMpapers](https://github.com/thunlp/PLMpapers) - BERT (Transformer, transfer learning) has catalyzed research in pretrained language models (PLMs) and has sparked many extensions. This repo contains a list of papers on PLMs.
9. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Google Brain.
- The group perform a systematic study of transfer learning for NLP using a unified Text-to-Text Transfer Transformer (T5) model and push the limits to achieve SoTA on SuperGLUE (approaching human baseline), SQuAD, and CNN/DM benchmark. [[Code](https://git.io/Je0cZ)].
10. [Reformer: The Efficient Transformer](https://openreview.net/forum?id=rkgNKkHtvB) by Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya.
- "They present techniques to reduce the time and memory complexity of Transformer, allowing batches of very long sequences (64K) to fit on one GPU. Should pave way for Transformer to be really impactful beyond NLP domain." — @hardmaru
11. [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/abs/1909.02950) (MMBT) by Facebook AI.
11. [A Primer in BERTology: What we know about how BERT works](https://arxiv.org/abs/2002.12327) by Anna Rogers et al.
- "Have you been drowning in BERT papers?". The group survey over 40 papers on BERT's linguistic knowledge, architecture tweaks, compression, multilinguality, and so on.
12. [tomohideshibata/BERT-related papers](https://github.com/tomohideshibata/BERT-related-papers)
13. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) by Google Brain. [[Code]](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py) | [[Blog post (unofficial)]](https://syncedreview.com/2021/01/14/google-brains-switch-transformer-language-model-packs-1-6-trillion-parameters/)
- Key idea: the architecture use a subset of parameters on every training step and on each example. Upside: model train much faster. Downside: super large model that won't fit in a lot of environments.
14. [An Attention Free Transformer](https://arxiv.org/abs/2105.14103) by Apple.
15. [A Survey of Transformers](https://arxiv.org/abs/2106.04554) by Tianyang Lin et al.
16. [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) by OpenAI.
- Codex, a GPT language model that powers GitHub Copilot.
- They investigate their model limitations (and strengths).
- They discuss the potential broader impacts of deploying powerful code generation techs, covering safety, security, and economics.

## Articles

### BERT and Transformer

1. [Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) from Google AI.
2. [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/).
3. [Dissecting BERT](https://medium.com/dissecting-bert) by Miguel Romero and Francisco Ingham - Understand BERT in depth with an intuitive, straightforward explanation of the relevant concepts.
3. [A Light Introduction to Transformer-XL](https://medium.com/dair-ai/a-light-introduction-to-transformer-xl-be5737feb13).
4. [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html) by Lilian Weng, Research Scientist at OpenAI.
5. [What is XLNet and why it outperforms BERT](https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335)
  - Permutation Language Modeling objective is the core of XLNet.
6. [DistilBERT](https://github.com/huggingface/pytorch-transformers/tree/master/examples/distillation) (from HuggingFace), released together with the blog post [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5).
7. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations paper](https://arxiv.org/abs/1909.11942v3) from Google Research and Toyota Technological Institute. — Improvements for more efficient parameter usage: factorized embedding parameterization, cross-layer parameter sharing, and Sentence Order Prediction (SOP) loss to model inter-sentence coherence. [[Blog post](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html) | [Code](https://github.com/google-research/ALBERT)]
8. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB) by Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning - A BERT variant like ALBERT and cost less to train. They trained a model that outperforms GPT by using only one GPU; match the performance of RoBERTa by using 1/4 computation. It uses a new pre-training approach, called replaced token detection (RTD), that trains a bidirectional model while learning from all input positions. [[Blog post](https://ai.googleblog.com/2020/03/more-efficient-nlp-model-pre-training.html) | [Code](https://github.com/google-research/electra)]
9. [Visual Paper Summary: ALBERT (A Lite BERT)](https://amitness.com/2020/02/albert-visual-summary/)

### Attention Concept

1. [The Annotated Transformer by Harvard NLP Group](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Further reading to understand the "Attention is all you need" paper.
2. [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) - Attention guide by Lilian Weng from OpenAI.
3. [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) by Jay Alammar, an Instructor from Udacity ML Engineer Nanodegree.
4. [Making Transformer networks simpler and more efficient](https://ai.facebook.com/blog/making-transformer-networks-simpler-and-more-efficient/) - FAIR released an all-attention layer to simplify the Transformer model and an adaptive attention span method to make it more efficient (reduce computation time and memory footprint).
5. [What Does BERT Look At? An Analysis of BERT’s Attention paper](https://arxiv.org/abs/1906.04341) by Stanford NLP Group.

### Transformer Architecture

1. [The Transformer blog post](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html).
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar, an Instructor from Udacity ML Engineer Nanodegree.
3. Watch [Łukasz Kaiser’s talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) walking through the model and its details.
4. [Transformer-XL: Unleashing the Potential of Attention Models](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html) by Google Brain.
5. [Generative Modeling with Sparse Transformers](https://openai.com/blog/sparse-transformer/) by OpenAI - an algorithmic improvement of the attention mechanism to extract patterns from sequences 30x longer than possible previously.
6. [Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764) paper by DeepMind and CMU - they propose architectural modifications to the original Transformer and XL variant by moving layer-norm and adding gating creates Gated Transformer-XL (GTrXL). It substantially improve the stability and learning speed (integrating experience through time) in RL.
7. [The Transformer Family](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html) by Lilian Weng - since the paper "Attention Is All You Need", many new things have happened to improve the Transformer model. This post is about that.
8. [DETR (**DE**tection **TR**ansformer): End-to-End Object Detection with Transformers](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers/) by FAIR - :fire: Computer vision has not yet been swept up by the Transformer revolution. DETR completely changes the architecture compared with previous object detection systems. ([PyTorch Code and pretrained models](https://github.com/facebookresearch/detr)). "A solid swing at (non-autoregressive) end-to-end detection. Anchor boxes + Non-Max Suppression (NMS) is a mess. I was hoping detection would go end-to-end back in ~2013)" — Andrej Karpathy

### Generative Pre-Training Transformer (GPT)

1. [Better Language Models and Their Implications](https://openai.com/blog/better-language-models/).
2. [Improving Language Understanding with Unsupervised Learning](https://blog.openai.com/language-unsupervised/) - this is an overview of the original OpenAI GPT model.
3. [🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning](https://convai.huggingface.co/) by Hugging Face.
4. [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar.
5. [MegatronLM: Training Billion+ Parameter Language Models Using GPU Model Parallelism](https://nv-adlr.github.io/MegatronLM) by NVIDIA ADLR.
6. [OpenGPT-2: We Replicated GPT-2 Because You Can Too](https://medium.com/@vanya_cohen/opengpt-2-we-replicated-gpt-2-because-you-can-too-45e34e6d36dc) - the authors trained a 1.5 billion parameter GPT-2 model on a similar sized text dataset and they reported results that can be compared with the original model.
7. [MSBuild demo of an OpenAI generative text model generating Python code](https://www.youtube.com/watch?v=fZSFNUT6iY8) [video] - The model that was trained on GitHub OSS repos. The model uses English-language code comments or simply function signatures to generate entire Python functions. Cool!
8. [GPT-3: Language Models are Few-Shot Learners (paper)](https://arxiv.org/abs/2005.14165) by Tom B. Brown (OpenAI) et al. - "We train GPT-3, an autoregressive language model with 175 billion parameters :scream:, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting."
9. [elyase/awesome-gpt3](https://github.com/elyase/awesome-gpt3) - A collection of demos and articles about the OpenAI GPT-3 API.
10. [How GPT3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/) by Jay Alammar.
11. [GPT-Neo](https://www.eleuther.ai/gpt-neo) - Replicate a GPT-3 sized model and open source it for free. GPT-Neo is "an implementation of model parallel GPT2 & GPT3-like models, with the ability to scale up to full GPT3 sizes (and possibly more!), using the mesh-tensorflow library." [[Code](https://github.com/EleutherAI/gpt-neo)].
12. [GitHub Copilot](https://copilot.github.com/), powered by OpenAI Codex - Codex is a descendant of GPT-3. Codex translates natural language into code.
13. [GPT-J-6B](https://towardsdatascience.com/cant-access-gpt-3-here-s-gpt-j-its-open-source-cousin-8af86a638b11) - Can't access GPT-3? Here's GPT-J — its open-source cousin.
14. [Fun and Dystopia With AI-Based Code Generation Using GPT-J-6B](https://minimaxir.com/2021/06/gpt-j-6b/) - Prior to GitHub Copilot tech preview launch, Max Woolf, a data scientist tested GPT-J-6B's code "writing" abilities.
15. [GPT-Code-Clippy (GPT-CC)](https://github.com/CodedotAl/gpt-code-clippy) - An open source version of GitHub Copilot. The GPT-CC models are fine-tuned versions of GPT-2 and GPT-Neo.
16. [GPT-NeoX-20B](https://blog.eleuther.ai/announcing-20b/) - A 20 billion parameter model trained using EleutherAI’s [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) framework. They expect it to perform well on many tasks. You cam try out the model on [GooseAI](https://goose.ai/) playground.

### Additional Reading

1. [How to Build OpenAI's GPT-2: "The AI That's Too Dangerous to Release"](https://www.reddit.com/r/MachineLearning/comments/bj0dsa/d_how_to_build_openais_gpt2_the_ai_thats_too/).
2. [OpenAI’s GPT2 - Food to Media hype or Wake Up Call?](https://www.skynettoday.com/briefs/gpt2)
3. [How the Transformers broke NLP leaderboards](https://hackingsemantics.xyz/2019/leaderboards/) by Anna Rogers. :fire::fire::fire:
- A well put summary post on problems with large models that dominate NLP these days.
- Larger models + more data = progress in Machine Learning research :question:
4. [Transformers From Scratch](http://www.peterbloem.nl/blog/transformers) tutorial by Peter Bloem.
5. [Real-time Natural Language Understanding with BERT using NVIDIA TensorRT](https://devblogs.nvidia.com/nlu-with-tensorrt-bert/) on Google Cloud T4 GPUs achieves 2.2 ms latency for inference. Optimizations are open source on GitHub.
6. [NLP's Clever Hans Moment has Arrived](https://thegradient.pub/nlps-clever-hans-moment-has-arrived/) by The Gradient.
7. [Language, trees, and geometry in neural networks](https://pair-code.github.io/interpretability/bert-tree/) - a series of expository notes accompanying the paper, "Visualizing and Measuring the Geometry of BERT" by Google's People + AI Research (PAIR) team.
8. [Benchmarking Transformers: PyTorch and TensorFlow](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2) by Hugging Face - a comparison of inference time (on CPU and GPU) and memory usage for a wide range of transformer architectures.
9. [Evolution of representations in the Transformer](https://lena-voita.github.io/posts/emnlp19_evolution.html) - An accessible article that presents the insights of their EMNLP 2019 paper. They look at how the representations of individual tokens in Transformers trained with different objectives change.
10. [The dark secrets of BERT](https://text-machine-lab.github.io/blog/2020/bert-secrets/) - This post probes fine-tuned BERT models for linguistic knowledge. In particular, the authors analyse how many self-attention patterns with some linguistic interpretation are actually used to solve downstream tasks. TL;DR: They are unable to find evidence that linguistically interpretable self-attention maps are crucial for downstream performance.
11. [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) - Tutorial on using BERT in practice, such as for sentiment analysis on movie reviews by Jay Alammar.
12. [Turing-NLG: A 17-billion-parameter language model](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/) by Microsoft that outperforms the state of the art on many downstream NLP tasks. This work would not be possible without breakthroughs produced by the [DeepSpeed library](https://github.com/microsoft/DeepSpeed) (compatible with PyTorch) and [ZeRO optimizer](https://arxiv.org/abs/1910.02054), which can be explored more in this accompanying [blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters).
13. [MUM (Multitask Unified Model): A new AI milestone for understanding information](https://blog.google/products/search/introducing-mum/) by Google.
- Based on transformer architecture but more powerful.
- Multitask means: supports text and images, knowledge transfer between 75 languages, understand context and go deeper in a topic, and generate content.
14. [GPT-3 is No Longer the Only Game in Town](https://lastweekin.ai/p/gpt-3-is-no-longer-the-only-game) - GPT-3 was by far the largest AI model of its kind last year (2020). Now? Not so much.
15. [OpenAI's API Now Available with No Waitlist](https://openai.com/blog/api-no-waitlist/) - GPT-3 access without the wait. However, apps must be approved before [going live](https://beta.openai.com/docs/going-live). This release also allow them to review applications, monitor for misuse, and better understand the effects of this tech.
16. [The Inherent Limitations of GPT-3](https://lastweekin.ai/p/the-inherent-limitations-of-gpt-3) - One thing missing from the article if you've read [Gwern's GPT-3 Creative Fiction article](https://www.gwern.net/GPT-3#repetitiondivergence-sampling) before is the mystery known as "Repetition/Divergence Sampling":
    > when you generate free-form completions, they have a tendency to eventually fall into repetitive loops of gibberish.

    For those using Copilot, you should have experienced this wierdness where it generates the same line or block of code over and over again.
17. [Language Modelling at Scale: Gopher, Ethical considerations, and Retrieval](https://deepmind.com/blog/article/language-modelling-at-scale) by DeepMind - The paper present an analysis of Transformer-based language model performance across a wide range of model scales — from models with tens of millions of parameters up to a 280 billion parameter model called Gopher.

## Tutorials

1. [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train) tutorial by Hugging Face. :fire:

## Videos

### [BERTology](https://huggingface.co/transformers/bertology.html)

1. [XLNet Explained](https://www.youtube.com/watch?v=naOuE9gLbZo) by NLP Breakfasts.
  - Clear explanation. Also covers the two-stream self-attention idea.
2. [The Future of NLP](https://youtu.be/G5lmya6eKtc) by 🤗
  - Dense overview of what is going on in transfer learning in NLP currently, limits, and future directions.
3. [The Transformer neural network architecture explained](https://youtu.be/FWFA4DGuzSc) by AI Coffee Break with Letitia Parcalabescu.
  - High-level explanation, best suited when unfamiliar with Transformers.

### Attention and Transformer Networks

1. [Sequence to Sequence Learning Animated (Inside Transformer Neural Networks and Attention Mechanisms)](https://youtu.be/GTVgJhSlHEk) by learningcurve.

## Official Implementations

1. [google-research/bert](https://github.com/google-research/bert) - TensorFlow code and pre-trained models for BERT.

## Other Implementations

### PyTorch and TensorFlow

1. [🤗 Hugging Face Transformers](https://github.com/huggingface/transformers) (formerly known as [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) and [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT)) provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, CTRL...) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch. [[Paper](https://arxiv.org/abs/1910.03771)]
2. [spacy-transformers](https://github.com/explosion/spacy-transformers) - a library that wrap Hugging Face's Transformers, in order to extract features to power NLP pipelines. It also calculates an alignment so the Transformer features can be related back to actual words instead of just wordpieces.

### PyTorch

1. [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch) - Google AI 2018 BERT pytorch implementation.
2. [innodatalabs/tbert](https://github.com/innodatalabs/tbert) - PyTorch port of BERT ML model.
3. [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl) - Code repository associated with the Transformer-XL paper.
4. [dreamgonfly/BERT-pytorch](https://github.com/dreamgonfly/BERT-pytorch) - A PyTorch implementation of BERT in "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding".
5. [dhlee347/pytorchic-bert](https://github.com/dhlee347/pytorchic-bert) - A Pytorch implementation of Google BERT.
6. [pingpong-ai/xlnet-pytorch](https://github.com/pingpong-ai/xlnet-pytorch) - A Pytorch implementation of Google Brain XLNet.
7. [facebook/fairseq](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md) - RoBERTa: A Robustly Optimized BERT Pretraining Approach by Facebook AI Research. SoTA results on GLUE, SQuAD and RACE.
8. [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Ongoing research training transformer language models at scale, including: BERT.
9. [deepset-ai/FARM](https://github.com/deepset-ai/FARM) - Simple & flexible transfer learning for the industry.
10. [NervanaSystems/nlp-architect](https://www.intel.ai/nlp-transformer-models/) - NLP Architect by Intel AI. Among other libraries, it provides a quantized version of Transformer models and efficient training method.
11. [kaushaltrivedi/fast-bert](https://github.com/kaushaltrivedi/fast-bert) - Super easy library for BERT based NLP models. Built based on 🤗 Transformers and is inspired by fast.ai.
12. [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo) - Neural Modules is a toolkit for conversational AI by NVIDIA. They are trying to [improve speech recognition with BERT post-processing](https://nvidia.github.io/NeMo/nlp/intro.html#improving-speech-recognition-with-bertx2-post-processing-model).
13. [facebook/MMBT](https://github.com/facebookresearch/mmbt/) from Facebook AI - Multimodal transformers model that can accept a transformer model and a computer vision model for classifying image and text.
14. [dbiir/UER-py](https://github.com/dbiir/UER-py) from Tencent and RUC - Open Source Pre-training Model Framework in PyTorch & Pre-trained Model Zoo (with more focus on Chinese).

### Keras

1. [Separius/BERT-keras](https://github.com/Separius/BERT-keras) - Keras implementation of BERT with pre-trained weights.
2. [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert) - Implementation of BERT that could load official pre-trained models for feature extraction and prediction.
3. [bojone/bert4keras](https://github.com/bojone/bert4keras) - Light reimplement of BERT for Keras.

### TensorFlow

1. [guotong1988/BERT-tensorflow](https://github.com/guotong1988/BERT-tensorflow) - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl) - Code repository associated with the Transformer-XL paper.
3. [zihangdai/xlnet](https://github.com/zihangdai/xlnet) - Code repository associated with the XLNet paper.

### Chainer

1. [soskek/bert-chainer](https://github.com/soskek/bert-chainer) - Chainer implementation of "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding".

## Transfer Learning in NLP

As Jay Alammar put it:

> The year 2018 has been an inflection point for machine learning models handling text (or more accurately, Natural Language Processing or NLP for short). Our conceptual understanding of how best to represent words and sentences in a way that best captures underlying meanings and relationships is rapidly evolving. Moreover, the NLP community has been putting forward incredibly powerful components that you can freely download and use in your own models and pipelines (It's been referred to as [NLP's ImageNet moment](http://ruder.io/nlp-imagenet/), referencing how years ago similar developments accelerated the development of machine learning in Computer Vision tasks).
>
> One of the latest milestones in this development is the [release](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) of [BERT](https://github.com/google-research/bert), an event [described](https://twitter.com/lmthang/status/1050543868041555969) as marking the beginning of a new era in NLP. BERT is a model that broke several records for how well models can handle language-based tasks. Soon after the release of the paper describing the model, the team also open-sourced the code of the model, and made available for download versions of the model that were already pre-trained on massive datasets. This is a momentous development since it enables anyone building a machine learning model involving language processing to use this powerhouse as a readily-available component – saving the time, energy, knowledge, and resources that would have gone to training a language-processing model from scratch.
>
> BERT builds on top of a number of clever ideas that have been bubbling up in the NLP community recently – including but not limited to [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) (by [Andrew Dai](https://twitter.com/iamandrewdai) and [Quoc Le](https://twitter.com/quocleix)), [ELMo](https://arxiv.org/abs/1802.05365) (by Matthew Peters and researchers from [AI2](https://allenai.org/) and [UW CSE](https://www.engr.washington.edu/about/bldgs/cse)), [ULMFiT](https://arxiv.org/abs/1801.06146) (by [fast.ai](https://fast.ai) founder [Jeremy Howard](https://twitter.com/jeremyphoward) and [Sebastian Ruder](https://twitter.com/seb_ruder)), the [OpenAI transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (by OpenAI researchers [Radford](https://twitter.com/alecrad), [Narasimhan](https://twitter.com/karthik_r_n), [Salimans](https://twitter.com/timsalimans), and [Sutskever](https://twitter.com/ilyasut)), and the Transformer ([Vaswani et al](https://arxiv.org/abs/1706.03762)).
>
> **ULMFiT: Nailing down Transfer Learning in NLP**
>
> [ULMFiT introduced methods to effectively utilize a lot of what the model learns during pre-training](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html) – more than just embeddings, and more than contextualized embeddings. ULMFiT introduced a language model and a process to effectively fine-tune that language model for various tasks.
>
> NLP finally had a way to do transfer learning probably as well as Computer Vision could.

[MultiFiT: Efficient Multi-lingual Language Model Fine-tuning](http://nlp.fast.ai/classification/2019/09/10/multifit.html) by Sebastian Ruder et al. MultiFiT extends ULMFiT to make it more efficient and more suitable for language modelling beyond English. ([EMNLP 2019 paper](https://arxiv.org/abs/1909.04761))

## Books

1. [Transfer Learning for Natural Language Processing](https://www.manning.com/books/transfer-learning-for-natural-language-processing) - A book that is a practical primer to transfer learning techniques capable of delivering huge improvements to your NLP models.

## Other Resources

<details>

<summary><b>Expand Other Resources</b></summary>

1. [hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service) - Mapping a variable-length sentence to a fixed-length vector using pretrained BERT model.
2. [brightmart/bert_language_understanding](https://github.com/brightmart/bert_language_understanding) - Pre-training of Deep Bidirectional Transformers for Language Understanding: pre-train TextCNN.
3. [algteam/bert-examples](https://github.com/algteam/bert-examples) - BERT examples.
4. [JayYip/bert-multiple-gpu](https://github.com/JayYip/bert-multiple-gpu) - A multiple GPU support version of BERT.
5. [HighCWu/keras-bert-tpu](https://github.com/HighCWu/keras-bert-tpu) - Implementation of BERT that could load official pre-trained models for feature extraction and prediction on TPU.
6. [whqwill/seq2seq-keyphrase-bert](https://github.com/whqwill/seq2seq-keyphrase-bert) - Add BERT to encoder part for https://github.com/memray/seq2seq-keyphrase-pytorch
7. [xu-song/bert_as_language_model](https://github.com/xu-song/bert_as_language_model) - BERT as language model, a fork from Google official BERT implementation.
8. [Y1ran/NLP-BERT--Chinese version](https://github.com/Y1ran/NLP-BERT--ChineseVersion)
9. [yuanxiaosc/Deep_dynamic_word_representation](https://github.com/yuanxiaosc/Deep_dynamic_word_representation) - TensorFlow code and pre-trained models for deep dynamic word representation (DDWR). It combines the BERT model and ELMo's deep context word representation.
10. [yangbisheng2009/cn-bert](https://github.com/yangbisheng2009/cn-bert)
11. [Willyoung2017/Bert_Attempt](https://github.com/Willyoung2017/Bert_Attempt)
12. [Pydataman/bert_examples](https://github.com/Pydataman/bert_examples) - Some examples of BERT. `run_classifier.py` based on Google BERT for Kaggle Quora Insincere Questions Classification challenge. `run_ner.py` is based on the first season of the Ruijin Hospital AI contest and a NER written by BERT.
13. [guotong1988/BERT-chinese](https://github.com/guotong1988/BERT-chinese) - Pre-training of deep bidirectional transformers for Chinese language understanding.
14. [zhongyunuestc/bert_multitask](https://github.com/zhongyunuestc/bert_multitask) - Multi-task.
15. [Microsoft/AzureML-BERT](https://github.com/Microsoft/AzureML-BERT) - End-to-end walk through for fine-tuning BERT using Azure Machine Learning.
16. [bigboNed3/bert_serving](https://github.com/bigboNed3/bert_serving) - Export BERT model for serving.
17. [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese) - BERT with SentencePiece for Japanese text.
18. [nickwalton/AIDungeon](https://github.com/nickwalton/AIDungeon) - AI Dungeon 2 is a completely AI generated text adventure built with OpenAI's largest 1.5B param GPT-2 model. It's a first of it's kind game that allows you to enter and will react to any action you can imagine.
19. [turtlesoupy/this-word-does-not-exist](https://github.com/turtlesoupy/this-word-does-not-exist) - "This Word Does Not Exist" is a project that allows people to train a variant of GPT-2 that makes up words, definitions and examples from scratch. We've never seen fake text so real.
</details>

## Tools 

1. [jessevig/bertviz](https://github.com/jessevig/bertviz) - Tool for visualizing attention in the Transformer model.
2. [FastBert](https://github.com/kaushaltrivedi/fast-bert) - A simple deep learning library that allows developers and data scientists to train and deploy BERT based models for NLP tasks beginning with text classification. The work on FastBert is inspired by fast.ai.

## Tasks

### Named-Entity Recognition (NER)

<details>

<summary><b>Expand NER</b></summary>

1. [kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER) - Use google BERT to do CoNLL-2003 NER.
2. [zhpmatrix/bert-sequence-tagging](https://github.com/zhpmatrix/bert-sequence-tagging) - Chinese sequence labeling.
3. [JamesGu14/BERT-NER-CLI](https://github.com/JamesGu14/BERT-NER-CLI) - Bert NER command line tester with step by step setup guide.
4. [sberbank-ai/ner-bert](https://github.com/sberbank-ai/ner-bert)
5. [mhcao916/NER_Based_on_BERT](https://github.com/mhcao916/NER_Based_on_BERT) - This project is based on Google BERT model, which is a Chinese NER.
6. [macanv/BERT-BiLSMT-CRF-NER](https://github.com/macanv/BERT-BiLSMT-CRF-NER) - TensorFlow solution of NER task using Bi-LSTM-CRF model with Google BERT fine-tuning.
7. [ProHiryu/bert-chinese-ner](https://github.com/ProHiryu/bert-chinese-ner) - Use the pre-trained language model BERT to do Chinese NER.
8. [FuYanzhe2/Name-Entity-Recognition](https://github.com/FuYanzhe2/Name-Entity-Recognition) - Lstm-CRF, Lattice-CRF, recent NER related papers.
9. [king-menin/ner-bert](https://github.com/king-menin/ner-bert) - NER task solution (BERT-Bi-LSTM-CRF) with Google BERT https://github.com/google-research.
</details>

### Classification

<details>

<summary><b>Expand Classification</b></summary>

1. [brightmart/sentiment_analysis_fine_grain](https://github.com/brightmart/sentiment_analysis_fine_grain) - Multi-label classification with BERT; Fine Grained Sentiment Analysis from AI challenger.
2. [zhpmatrix/Kaggle-Quora-Insincere-Questions-Classification](https://github.com/zhpmatrix/Kaggle-Quora-Insincere-Questions-Classification) - Kaggle baseline—fine-tuning BERT and tensor2tensor based Transformer encoder solution.
3. [maksna/bert-fine-tuning-for-chinese-multiclass-classification](https://github.com/maksna/bert-fine-tuning-for-chinese-multiclass-classification) - Use Google pre-training model BERT to fine-tune for the Chinese multiclass classification.
4. [NLPScott/bert-Chinese-classification-task](https://github.com/NLPScott/bert-Chinese-classification-task) - BERT Chinese classification practice.
5. [fooSynaptic/BERT_classifer_trial](https://github.com/fooSynaptic/BERT_classifer_trial) - BERT trial for Chinese corpus classfication.
6. [xiaopingzhong/bert-finetune-for-classfier](https://github.com/xiaopingzhong/bert-finetune-for-classfier) - Fine-tuning the BERT model while building your own dataset for classification.
7. [Socialbird-AILab/BERT-Classification-Tutorial](https://github.com/Socialbird-AILab/BERT-Classification-Tutorial) - Tutorial.
8. [malteos/pytorch-bert-document-classification](https://github.com/malteos/pytorch-bert-document-classification/) - Enriching BERT with Knowledge Graph Embedding for Document Classification (PyTorch)
</details>

### Text Generation

<details>

<summary><b>Expand Text Generation</b></summary>

1. [asyml/texar](https://github.com/asyml/texar) - Toolkit for Text Generation and Beyond. [Texar](https://texar.io) is a general-purpose text generation toolkit, has also implemented BERT here for classification, and text generation applications by combining with Texar's other modules.
2. [Plug and Play Language Models: a Simple Approach to Controlled Text Generation](https://arxiv.org/abs/1912.02164) (PPLM) paper by Uber AI.
</details>

### Question Answering (QA)

<details>

<summary><b>Expand QA</b></summary>

1. [matthew-z/R-net](https://github.com/matthew-z/R-net) - R-net in PyTorch, with BERT and ELMo.
2. [vliu15/BERT](https://github.com/vliu15/BERT) - TensorFlow implementation of BERT for QA.
3. [benywon/ChineseBert](https://github.com/benywon/ChineseBert) - This is a Chinese BERT model specific for question answering.
4. [xzp27/BERT-for-Chinese-Question-Answering](https://github.com/xzp27/BERT-for-Chinese-Question-Answering)
5. [facebookresearch/SpanBERT](https://github.com/facebookresearch/SpanBERT) - Question Answering on SQuAD; improving pre-training by representing and predicting spans.
</details>

### Knowledge Graph

<details>

<summary><b>Expand Knowledge Graph</b></summary>

1. [sakuranew/BERT-AttributeExtraction](https://github.com/sakuranew/BERT-AttributeExtraction) - Using BERT for attribute extraction in knowledge graph. Fine-tuning and feature extraction. The BERT-based fine-tuning and feature extraction methods are used to extract knowledge attributes of Baidu Encyclopedia characters.
2. [lvjianxin/Knowledge-extraction](https://github.com/lvjianxin/Knowledge-extraction) - Chinese knowledge-based extraction. Baseline: bi-LSTM+CRF upgrade: BERT pre-training.
</details>

## License

<details>

<summary><b>Expand License</b></summary>

This repository contains a variety of content; some developed by Cedric Chee, and some from third-parties. The third-party content is distributed under the license provided by those parties.

*I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer.*

The content developed by Cedric Chee is distributed under the following license:

### Code

The code in this repository, including all code samples in the notebooks listed above, is released under the [MIT license](LICENSE). Read more at the [Open Source Initiative](https://opensource.org/licenses/MIT).

### Text

The text content of the book is released under the CC-BY-NC-ND license. Read more at [Creative Commons](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode).
</details>
