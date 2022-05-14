# 组会论文阅读

## 1. Audio Based Prosody Estimation 

### 1.1 Abstract 

Prosodic boundary plays an important role in text-to-speech synthesis (TTS) in terms of naturalness and readability. However, the acquisition of prosodic boundary labels relies on manual annotation, which is costly and time-consuming. In this paper, we propose to automatically extract prosodic boundary labels from text-audio data via a neural text-speech model with pre-trained audio encoders. This model is pre-trained on text and speech data separately and jointly fine-tuned on TTS data in a triplet format: {speech, text, prosody}. The experimental results on both automatic evaluation and human evaluation demonstrate that: 1) the proposed text-speech prosody annotation framework significantly outperforms text-only baselines; 2) the quality of automatic prosodic boundary annotations is comparable to human annotations; 3) TTS systems trained with model-annotated boundaries are slightly better than systems that use manual ones. 
.
<!-- 韵律边界在语篇合成的自然度和可读性方面起着至关重要的作用。然而，韵律边界标签的获取依赖于人工标注，成本高、耗时长。在本文中，我们提出通过一个预先训练音频编码器的文本-语音神经模型，从文本-音频数据中自动提取韵律边界标签。该模型分别对文本和语音数据进行预训练，并对TTS数据进行联合微调，采用三组格式:{speech, text，韵律}。自动评价和人工评价的实验结果表明:1)本文提出的文本-语音韵律标注框架显著优于文本基准;2)韵律边界自动标注的质量与人工标注的质量相当;3)使用模型注释边界训练的TTS系统比使用人工边界训练的系统略好。 -->

### 1.2 Introduction 

In text-to-speech synthesis (TTS), prosody modeling plays an important role in synthesizing high naturalness and intelligibility speech. Due to the scarcity and high cost of prosody annotation in the current TTS dataset, there have been a lot of works attempting to model the prosody in a latent space without explicit prosody annotation [1,2]. However, recent research efforts have shown that using explicit hierarchical prosodic boundary annotation [3] in training and inference can still improve the fidelity and expressiveness of Mandarin speech synthesis [4], which indicates that prosody annotation is still useful for TTS system construction. As shown in Figure 1, the hierarchical prosody annotation adopted in this work categorizes the prosodic boundaries of Mandarin speech into five levels, including Character (CC), Lexicon Word (LW), Prosodic Word (PW), Prosodic Phrase (PPH) and Intonational Phrase (IPH) [5].

<!-- 在文本到语音合成(TTS)中，韵律建模在合成自然度高、可理解度高的语音中起着重要作用。由于当前TTS数据集中韵律注释的稀缺和成本较高，已有大量的研究试图在没有明确韵律注释的潜在空间中建模韵律[1,2]。然而，最近的研究表明，在训练和推理中使用显式层次韵律边界标注[3]仍然可以提高汉语语音合成[4]的保真度和表达性，这表明韵律边界标注在TTS系统构建中仍然是有用的。如图1所示，本文采用的韵律分级标注将普通话语音的韵律边界划分为五个层次，分别是字符(CC)、词汇(LW)、韵律词(PW)、韵律短语(PPH)和语调短语(IPH)[5]。 -->

While the importance of prosodic boundary has been demonstrated in previous studies, one most critical challenge is to obtain the prosodic boundaries. The pipeline for collecting TTS training data with prosody annotations is shown in Figure 2, the acquisition of prosodic boundaries relies on manual annotation with text and speech, which is expensive and timeconsuming. In addition, through preliminary experiments, we also find the inter-annotator agreement between different human annotators is low, indicating that prosody annotation can be ambiguous and the inconsistency may lead to difficulties in training models.

<!-- 虽然韵律边界的重要性已经在以往的研究中得到证实，但最关键的挑战是获得韵律边界。基于韵律标注的TTS训练数据采集管道如图2所示，韵律边界的获取依赖于文本和语音的人工标注，成本高且耗时。此外，通过初步实验，我们还发现不同的人类注释者之间的注释者之间的一致性较低，这表明韵律注释可能存在歧义，这种不一致性可能会导致模型的训练困难。 -->

In this paper, we propose to reduce the cost of prosody annotation and improve the label consistency via an automatic annotator. Our key idea is to automatically annotate the prosodic boundaries through a pre-trained text-speech model that takes a text-speech pair as input. Specifically, the proposed model consists of three components: a text encoder, an audio encoder and a multi-modal fusion decoder. The former two components are pre-trained on text and audio data respectively, and the multimodal fusion decoder is optimized using triplet format TTS data: {speech, text, prosody}.

<!-- 本文提出利用自动标注器来降低韵律标注的成本，提高标注的一致性。我们的关键思想是通过一个预先训练的文本-语音模型自动标注韵律边界，该模型以文本-语音对作为输入。具体地说，该模型由三个部分组成:文本编码器、音频编码器和多模态融合解码器。前两个分量分别对文本数据和音频数据进行预训练，利用三联格式TTS数据:{speech, text，韵律}对多模态融合解码器进行优化。 -->

Three experiments are conducted to evaluate the effectiveness of this automatic annotator. The first one directly calculates the precision, recall and f1 scores of the automatic annotations, the second one compares the accuracy of human- and modelannotated boundaries using A/B test, while the third evaluates the naturalness of TTS systems trained with different prosodic boundaries. Surprisingly, the experimental results show that the model-annotated boundaries do not result in worse, but instead slightly better, TTS performance. We attribute the results to our previous finding that human-annotated prosodic boundaries are inconsistent across individuals.

<!-- 通过三个实验来评价该自动标注器的有效性。第一种方法直接计算自动标注的precision、recall和f1得分，第二种方法使用A/B测试比较human- annotated boundary和model - annotated boundary的准确率，第三种方法评估不同韵律边界训练的TTS系统的自然度。令人惊讶的是，实验结果表明，模型注释的边界并没有导致更差的TTS性能，而是略有提高。我们将这一结果归因于我们之前的发现，人类注释的韵律边界在个体之间是不一致的。 -->

The contributions of this paper are listed as follows: 1) We propose a text-speech framework to automatically annotate prosodic boundaries. 2) Different kinds of audio encoders pretrained on character-level targets and phonetic posteriorgram (PPG) [6] are systematically investigated to improve the annotation quality. 3) The experimental result suggests that automatic annotations can translate to comparable TTS performance with manual annotations

<!-- 本文的主要贡献如下:1)提出了一种文本-语音自动标注韵律边界的框架。2)系统研究了不同类型的字符级目标和语音后置图(PPG)[6]预训练的音频编码器，以提高标注质量。3)实验结果表明，自动标注的TTS性能可与人工标注的TTS性能相媲美 -->


### 1.3 Automatic Prosody Annotator

In this section, we introduce our proposed automatic prosody annotator framework. Similar to human annotation, the model is requested to annotate prosodic boundaries according to the prosody information inherently contained in audio, so it takes both audio waveform and text as input. This is also the main difference between our model and those text-based prosodic boundary prediction models [3, 7–19]. As shown in Fig 3, the proposed framework consists of three main components: a text encoder, an audio encoder, and a multi-modal fusion decoder. The text and audio encoders are used to extract high-level hidden representations from text and audio respectively, and the multi-modal fusion decoder is used to fuse the text and audio information to estimate prosodic boundary. Since this model requires paired audio-prosody training data that are sparse and costly, we initialize our encoders with pre-trained models

<!-- 在本节中，我们将介绍我们提出的自动韵律注释器框架。与人类注释相似，模型需要根据音频中固有的韵律信息对韵律边界进行注释，因此它既以音频波形为输入，也以文本为输入。这也是我们的模型与基于文本的韵律边界预测模型的主要区别[3,7 - 19]。如图3所示，该框架由三个主要组件组成:文本编码器、音频编码器和多模态融合解码器。利用文本编码器和音频编码器分别从文本和音频中提取高级隐藏表示，利用多模态融合解码器融合文本和音频信息来估计韵律边界。由于这个模型需要成对的音频韵律训练数据，这些数据是稀疏和昂贵的，所以我们使用预先训练的模型初始化编码器 -->

#### 1.3.1 Text Encoder 

Chinese Bert 

#### 1.3.2 Audio Encoder 

As prosody information is inherently contained in the audio modality, a powerful audio encoder that can extract representative prosody-related information from the audio waveform is important to boost estimation accuracy

<!-- 由于韵律信息固有地包含在音频模态中，一种强大的音频编码器可以从音频波形中提取具有代表性的韵律相关信息，这对提高估计精度非常重要 -->

PPG extractor: Motivated by the great success of Phonetic posteriorgram (PPG) [6] in voice conversion, we firstly adopt a pre-trained conformer-based PPG extractor as the audio encoder. Specifically, the PPG extractor used in this work is a speaker-independent frame-level classifier that maps each input time frame to posterior probabilities of phonetic classes. In automatic prosody annotator task, we believe that PPG can represent the duration and transition information of each phoneme in the audio, which is important to prosodic boundary estimation

<!-- PPG提取器:基于语音后验图(Phonetic posteriorgram, PPG)[6]在语音转换方面的巨大成功，我们首先采用一个预先训练的基于一致性的PPG提取器作为音频编码器。具体来说，本研究中使用的PPG提取器是一个与说话者无关的帧级分类器，它将每个输入时间帧映射到语音类的后端概率。在自动韵律标注任务中，我们认为PPG可以表示音频中每个音素的持续时间和过渡信息，这对韵律边界估计很重要 -->

Character-based encoder: However, as prosodic boundaries are usually related to the word and character language information, the phoneme-based PPG model misses the characterlevel language context information, which can lead to suboptimal performance. For example, these two sequences ”大学生物，必修课” and ”大学生, 务必修课” have the same phone sequence, however, their prosodic boundaries are different. To this end, two character-based ASR models based on CNN and conformer [20] architectures are investigated. Different from the PPG extractor, the character-based model transforms each input time frame to posterior probabilities of char acter classes, which can better keep the character-level information. The difference between the two model architectures is that the CNN-based model focuses on the local information, while the conformer-based model considers the whole utterance information. Note that, since there are more than 5000 characters in the vocabulary, the 512 dimensional hidden representations of the second last layer are adopted as the audio encoder hidden outputs for character-based pre-trained audio encoders.

<!-- 基于字符的编码器:然而，韵律边界通常与单词和字符的语言信息有关，基于音素的PPG模型忽略了字符级的语言上下文信息，导致性能欠佳。例如,这两个序列”大学生物,必修课”和“大学生,务必修课”有相同的电话序列,然而,它们的韵律边界是不同的。为此，本文研究了两种基于CNN和构象[20]架构的基于字符的ASR模型。与PPG提取器不同的是，基于字符的模型将每个输入帧转换为字符类的后验概率，可以更好地保留字符级信息。两种模型架构的不同之处在于，基于cnn的模型关注的是局部信息，而基于从众者的模型考虑的是整个话语信息。需要注意的是，由于词汇表中有超过5000个字符，对于基于字符的预训练音频编码器，我们采用最后一层的512维隐藏表示作为音频编码器的隐藏输出。 -->

#### 1.2.3 Multi-Modal Fusion Decoder

With the audio and text hidden representations, a multi-modal fusion decoder is needed to fuse these two representations and then estimate the prosodic boundary. One challenge of the fusion of these two hidden representations is that the frame-level audio hidden representation is usually much longer than the token-level text representation. To address this issue, a crossattention based multi-modal fusion decoder is proposed in this work.

<!-- 对于音频和文本隐藏表示，需要一个多模态融合解码器来融合这两种表示，然后估计韵律边界。融合这两种隐藏表示的一个挑战是，帧级音频隐藏表示通常比标记级文本表示长得多。为了解决这一问题，本文提出了一种基于交叉注意的多模态融合解码器。 -->

As shown in Fig.3, the audio hidden representations are fed into a stack of 6 identical layers, each layer is composed of a multi-head self-attention layer followed by a feed-forward layer. Then, the transformed representations go through a linear layer, which changes their dimension to be the same with the text hidden representations. Lastly, the hidden representations of the text and audio modalities are fed into another stack of 6 identical layers, each composing a multi-head cross-attention layer followed by a feed-forward layer.

<!-- 如图3所示，音频隐藏表示被送入6个相同层的堆栈中，每一层由一个多头自我注意层和一个前馈层组成。然后，转换后的表示通过一个线性层，该层将其维数更改为与文本隐藏表示相同。最后，文本和音频模式的隐藏表示被送入另一个由6个相同层组成的堆栈，每个层构成一个多头部交叉注意层，然后是一个前馈层。 -->

Since the length of output prosodic boundary estimation is the same as the length of the input text sequence, in crossattention, the text modality is used as query, the audio modality is used as key and value. Let O = [o1, ..., oT] ∈ R T×D and X = [x1, ..., xN] ∈ R N×D denote the audio and text hidden input of the multi-head cross-attention layer respectively, where D is the dimension of the hidden representation. The fusion of the audio and text modality can be expressed as follows:


