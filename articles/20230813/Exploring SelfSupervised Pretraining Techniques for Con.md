
作者：禅与计算机程序设计艺术                    

# 1.简介
  

 Contrastive Representation Learning (CRL) is a widely used technique in computer vision and natural language processing (NLP), where the goal is to learn representations that are meaningful and discriminative over input data points with high similarity or dissimilarity. These self-supervised techniques can leverage large unlabeled datasets of images or text documents to generate these contrastive representations, which can be further fine-tuned on labeled data sets using standard supervised learning approaches like classification and regression tasks. In this work, we propose several variants of CRL pretraining techniques that exploit powerful architectures such as Vision Transformer (ViT) and transformers to better capture rich contextual features from image and text data respectively. We also experiment with new initialization strategies and regularization techniques to improve the generalization performance of our models across different downstream tasks. Our results show significant improvements over popular CRL methods and open up a range of exciting research avenues for future research.
# 2.相关研究
 Existing works have demonstrated the effectiveness of convolutional neural networks (CNNs) in feature extraction and representation learning from raw pixel inputs, but they still struggle to capture both global and local information within an image. ViTs address this challenge by introducing a transformer architecture into CNN backbones, which captures complex spatial relationships between pixels and generates hierarchical feature representations at multiple scales. While ViTs provide impressive accuracy on some benchmarks, they still require large amounts of training data and cannot directly serve as substitute models for vanilla CNNs due to their limited capacity and low complexity compared to CNNs. To bridge this gap, many researchers proposed pretraining schemes that jointly train two similar but distinct modalities together, thereby generating more robust and informative representations than either alone. Such schemes include SimCLR [1], Barlow Twins [2], BYOL [3], MoCo [4], and SwAV [5]. 

 However, while most of these pretraining schemes rely heavily on supervised learning, few of them focus specifically on identifying useful cross-modal correlations among the two modalities. Moreover, the existing SSL techniques mostly target multi-class image recognition problems where each class corresponds to one modality and requires specialized loss functions and evaluation metrics. As such, they may not directly apply to NLP applications without extensive adaptation. Additionally, SSL techniques that involve only translation-invariant representations such as BiT [7] and its family fail to fully exploit the rich semantics encoded in text data due to their inability to capture local dependencies in text sequences.

 To solve these issues, we propose a novel variant of CRL pretraining called SimSiam [6], which incorporates a Siamese network architecture into the contrastive learning framework to learn shared representations for both text and visual inputs simultaneously. The key idea behind SimSiam is to learn the optimal transport plan between two views of any given example in both domains by minimizing the Euclidean distance between their corresponding learned representations. This allows the model to transfer knowledge between modalities by aligning their respective distributions of embeddings. We evaluate our approach on three benchmark NLP tasks - sentiment analysis, machine translation, and topic clustering - and demonstrate consistent improvement over baselines while reaching higher accuracies than state-of-the-art SSL algorithms. Furthermore, we find that ViT based pretraining leads to significantly improved generalization performance across all down-stream tasks. Overall, our findings suggest that relying solely on strong pretraining capabilities alone might not always lead to satisfactory results in practical settings. Together, our contributions aim to inspire new ideas and directions towards advancing the field of Constrastive Representation Learning in NLP and CV applications. 

# 3.SimSiam: A Simple Framework for Contrastive Learning of Visual and Text Representations
SimSiam is a simple yet effective contrastive learning algorithm for learning representations from text and image inputs that learns good visual embeddings even when trained on small, weak supervision. It uses a Siamese Network (Sn) to project each view of an input instance onto a common embedding space using a Siamese objective function based on the triplet loss [8]. Specifically, Sn consists of two identical subnetworks that share weights except for the projection layer. During training, we alternate between updating the parameters of these two subnetworks to minimize the difference between the projections of anchor instances from the same domain and positive/negative instances from another domain. Once trained, the final learned representation is simply the mean of the projections of the two subnetworks. Unlike previous works that typically use linear transformations or convolutions followed by pooling layers to map raw inputs to embeddings, our Siamese network follows the usual practice of initializing the weight matrices randomly and applying batch norm during training.


The above figure shows the basic design of the SimSiam architecture. The left half of the figure represents the encoder component of the Siamese network, which takes raw text and image inputs and produces fixed length embedding vectors for each input instance. The right half of the figure depicts the projection head, which projects the resulting embedding vectors to a common embedding space through an MLP layer. The blue arrow indicates the flow of information from the source domain to the target domain, i.e., how the model transfers knowledge from the text domain to the image domain. Finally, the green arrows indicate the opposite direction of transfer, i.e., how it transfers knowledge from the image domain to the text domain. Note that the order of operations here differs slightly from earlier works that optimize the alignment between the latent spaces rather than learning a mapping between them. 

 # Algorithm Details 
 
 1. Data Augmentation: Before feeding the raw inputs into the Siamese network, we perform data augmentation by randomly cropping and flipping the images, and by random permutations of words in sentences. This helps to reduce the risk of overfitting and ensures that the text and image pairs differ in terms of content, style, and layout.

 2. Instance Selection Strategy: Similar to existing contrastive learning algorithms like SimCLR [1], we sample negatives using a pool of possible negative examples generated using geometric transformations. Each transformation applied to an anchor example generates two additional negative examples whose representations will be ignored during training. 

 3. Loss Function: For each pair of anchor-positive and anchor-negative instances, we compute the squared Euclidean distance between their projections to obtain the Siamese loss. In addition, we add a softmax temperature parameter $T$ that controls the sharpness of the logit scale and the concentration of samples near zero probability. Intuitively, increasing the value of T encourages the network to produce confident predictions early on, while lower values allow the distribution of predicted probabilities to fluctuate more smoothly later in training. 

 4. Optimization Procedure: Instead of updating the entire Siamese network after every single mini-batch, we update only the subnetwork responsible for producing the anchor's output projection before computing the loss. This reduces computational overhead and enables us to effectively utilize GPU resources. We employ stochastic gradient descent with momentum ($\beta=0.9$) and learning rate warmup. 

 # Evaluation
 
 We compare our method against several competitive baselines on various NLP and CV tasks. All evaluations were conducted on standard public benchmarks including Sentiment Analysis, Machine Translation, and Topic Clustering. 

## Experiment Settings 

We trained the SimSiam model on ImageNet-1K and Wikitext-103 datasets using ResNet-50 (trained on ImageNet-1K) as the encoder architecture and BERT-base (trained on Wikitext-103) as the projection head. Both networks were trained using the same hyperparameters, including mini-batch size of 256, learning rate of 0.03, weight decay of 0.0001, and optimizer AdamW. We evaluated the models on a range of downstream tasks involving text and image inputs, including sentiment analysis, machine translation, and topic clustering, and measured the top-1 accuracy, macro F1 score, and R@1 precision metric.

### Baseline Models: 

1. Random: This baseline assigns labels uniformly at random. Hence, it does not exploit any visual or linguistic cues and suffers from poor performance on a diverse set of tasks. 

2. Supervised Contrastive Model (SupCon): SupCon uses a contrastive loss function to maximize the agreement between embeddings of different views of the same input. Its core intuition is that if two inputs come from different domains, then they should be mapped to different spaces. Thus, it relies exclusively on the mutual information between different views of an input to achieve good performance on downstream tasks. Nevertheless, SupCon has been shown to be sensitive to hyperparameter choices and performs poorly on certain types of tasks that do not exhibit sufficiently unique views. Therefore, we believe that it is important to consider alternative architectures and initializations that can outperform traditional self-supervised methods on a wider range of tasks. 

3. Linear Alignment Model (LAM): LAM optimizes a dot product between embeddings of different views of an input to predict whether they belong to the same or different domains. It assumes that the embeddings are produced by a linear transformation, which limits its expressivity and hence fails to capture non-linear correlations present in real-world data. Additionally, LAM is very computationally expensive and hard to tune since it involves matrix factorization. 

4. Metric Learning Approaches: Various metric learning approaches exist such as InfoNCE, TripletLoss, etc., which learn the distance metric between pairs of inputs based on a predefined similarity measure. One advantage of these methods is that they automatically handle imbalanced classes and ignore easy-to-classify inputs. However, none of them explicitly model the constraints of text and image inputs and thus fall short of capturing the true underlying structure of data.

### Competitive Methods 

In addition to the baseline models listed above, we also tested several recent self-supervised pretraining methods designed specifically for text and image data. The following table summarizes the main components of these methods along with our modifications or alternatives: 


| Method                  | Components                 | Modifications   | Alternatives            |
|-------------------------|----------------------------|-----------------|-------------------------|
| CLIP                    | VisionTransformer          | Negative Sampling             | MAE                     |
| VirTex                  | VisionTransformer + LanguageTransformer      | Projection Head         | ContextAware             |
| Perceiver               | MultiHeadAttention + EncoderDecoder    | Projection Head     | VQ-VAE                   |
| CPCv2                   | Convolutional Sequence Modeling      | Different Subspaces       |                         |



All these methods follow the same overall pipeline of taking raw inputs, encoding them using deep neural networks, and projecting the resulting encodings into a shared embedding space. Most of them are composed of a multimodal encoder and projection head that processes the raw inputs independently and combines them at the end. Some of them modify the original encoder architecture or introduce auxiliary losses that help to mitigate the curse of dimensionality and improve generalizability. None of these methods explore interactions between text and image inputs and therefore remain limited in their ability to capture high-level semantic structures and reason about the relationship between them. 

# Results

Overall, we find that SimSiam consistently improves over the baselines and outperforms competing methods on a wide range of text and image classification tasks. The key insight behind SimSiam is that it provides a way to learn shared representations from text and image inputs by modeling the optimal transport plan between the two views of any given example, allowing it to learn to transfer knowledge between modalities effectively. By combining these insights with efficient optimization procedures, we arrive at a simple yet effective algorithm that achieves good results on challenging tasks while requiring minimal preprocessing and resource consumption.