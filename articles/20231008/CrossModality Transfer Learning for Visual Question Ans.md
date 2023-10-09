
作者：禅与计算机程序设计艺术                    

# 1.背景介绍



Visual Question Answering (VQA) is a challenging task in computer vision that requires an AI agent to read and comprehend textual questions related to images or videos. The key challenge lies in how to transfer knowledge across modalities between the image and the language input during training and inference of VQA systems. In this paper, we propose a novel cross-modality transfer learning framework called CMTL (Cross Modality Transfer Learning), which can effectively leverage information from both the visual and linguistic inputs to improve the performance of VQA models on various tasks such as open-ended question answering (OEQA), multiple choice (MCQA), and cloze question answering (CQA). We conduct extensive experiments on two public datasets - OEQA-VQA and MC-VQA, demonstrating that our proposed method achieves state-of-the-art results on several benchmark metrics while significantly outperforming alternative approaches by significant margins. Moreover, we showcase the potential of CMTL in improving VQA model generalization beyond the dataset domain using unseen combinations of image and language inputs.


# 2.核心概念与联系
## 2.1 Cross-modality Transfer Learning（CMTL）

In order to address the limitations of traditional VQA models, we propose a new approach called Cross-modality Transfer Learning (CMTL) for transferring knowledge across modalities. 

The basic idea behind CMTL is straightforward: given two modality pairs (X,Y) and their corresponding feature vectors x ∈ X and y ∈ Y respectively, learn a shared representation r = f(x,y) that captures the underlying semantics shared between them. This allows us to bridge the gap between disparate features captured by different modalities, enabling us to perform better multimodal reasoning at test time. Specifically, during training, we use pairwise ranking losses that encourage the learned representation to capture similarities between samples from each modality pair. At test time, we predict an output label based only on the shared representation alone, ignoring any specific modality features. 

This approach involves three main components: 

1. Pairwise Ranking Losses: These are used to train the model to map representations from one modality into another space where they share the same semantic meaning. For example, if we have an image-text modality pair with data points labeled as pos/neg, we may want to assign higher scores to pairs where the corresponding image and text embeddings are close together and lower scores otherwise. There exist many types of loss functions that can be used here, but a popular one is triplet ranking loss which measures the distance between anchor sample (e.g., the image) and positive sample (e.g., the correct caption) relative to negative sample (e.g., incorrect captions). However, other variants of ranking loss such as softmax regression loss also work well in practice.

2. Shared Representation Function: During training, we first compute the joint embedding of each datapoint across all modality pairs. Then, we apply some non-linear transformation function f to these joint embeddings to obtain a shared representation vector r. One common choice for f is a neural network with hidden layers.

3. Multimodal Reasoning Module: Finally, we use the trained shared representation r to make predictions about the output label, without relying on any individual modality features. This improves the robustness and generality of the model since it learns a single representation that captures both visual and linguistic aspects of the problem.



## 2.2 Modalities and Feature Vectors

For each modality, we represent its content as a feature vector x ∈ Rd, where d is the dimensionality of the feature space. Examples of commonly used modality pairs include:

- Image and Text: An image represented as an RGB pixel matrix and a sentence or paragraph representing natural language description.
- Video and Text: A video represented as a sequence of frames, where each frame contains RGB pixels and a timestamp indicating when it was captured. Also, there could be a sentence or paragraph describing what should happen in the video.
- Speech and Audio: A speech recording represented as a sequence of audio samples and metadata like speaker ID, gender, age etc. We assume that the speaker’s voice belongs to a particular class C and hence their utterances share the same semantic meaning. Similarly, audio data belonging to the same class will typically be closer in distance than those from different classes. Therefore, we can utilize the class labels to create classification-based pairwise rankings. Additionally, we can directly compare speech and audio representations using cosine similarity rather than converting them into word embeddings and then performing inference over the resulting dense vectors.

## 2.3 Pairwise Ranking Loss Functions

We use the following set of loss functions to enforce the consistency of the representations across modality pairs:

1. Contrastive Loss: This measures the minimum difference between pairs of vectors, encouraging them to be more similar than different. The goal of contrastive loss is to minimize the average distance between pairs from the same class versus pairs from different classes. A typical way to implement this loss function is to use binary logistic regression on concatenated pairs of image and text features.

2. Triplet Ranking Loss: This variant of contrastive loss rewards pairs whose distances from anchors are smaller than the sum of distances from negatives. It computes the Euclidean distance between anchor and positive samples, as well as the maximum distance from the negative examples. The objective is to push the anchor towards the positive direction and pull away the negatives. The hyperparameter α controls the balance between hard and easy positives.

3. Softmax Regression Loss: Another approach is to use softmax regression on the predicted probability distribution of the correct answers for each modality pair. Each sample has a weight associated with each possible answer, computed as a dot product between the final layer of the network and the precomputed feature vectors. The weights are adjusted based on the error made in prediction. Again, we choose the best answer among these probabilities as the final decision.

4. Maximum Mean Discrepancy (MMD)-based Loss: MMD is a measure of the difference between distributions, specifically designed for comparing probability distributions. Here, we use kernel density estimation (KDE) to estimate the distribution of each modality independently. We define two KDE estimates p_i(x) and q_j(x), where i represents the image modality and j represents the text modality. We then compute the MMD between the two distributions as follows:

   $MMD^2(P,Q)=\frac{1}{m^2}\sum_{i=1}^{m}\sum_{j=1}^{n} k(\mu_i,\mu_j)^2 + \frac{2}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n} k(\mu_i,\mu_j)$

    where m is the number of data points in P, n is the number of data points in Q, $\mu_i$ is the mean of the ith distribution, and $k(a,b)$ is a kernel function that takes two scalar values and returns a scalar value. Common choices for the kernel function include Gaussian, Laplacian, and Tukey biweight.

All four loss functions mentioned above have their advantages and drawbacks depending on the type of modality and the size of the dataset. Nevertheless, experimentation shows that the choice of loss function plays a crucial role in achieving good performance and generalizing well to unseen combinations of modality pairs.