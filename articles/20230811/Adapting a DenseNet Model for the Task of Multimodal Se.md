
作者：禅与计算机程序设计艺术                    

# 1.简介
         

Multimodal sentiment analysis (MSA) refers to analyzing multiple modalities such as text and image in order to predict the overall sentiment or emotion conveyed by an input text along with its visual content. MSA is becoming more popular due to the advancements in various NLP techniques that enable models to incorporate different features from both textual and visual domains into their predictions. 

One commonly used deep learning architecture for MSA tasks is called DenseNet [1]. The main idea behind this model is to build densely connected networks that share weights between layers, leading to improved performance on semantic segmentation tasks like pixel-level classification [2]. However, while traditional convolutional neural network (CNN) architectures have been shown to be effective at processing spatial information, they are not well suited for processing sequential data like language, making it challenging to adapt these CNNs for multimodal sentiment analysis.

In this article, we will discuss how we adapted DenseNet for the task of MSA using TensorFlow 2.x library. We will also compare our results against other state-of-the-art approaches for benchmarking purposes.


# 2.Multimodal Sentiment Analysis (MSA)
## Definition 
Multimodal sentiment analysis (MSA) is the task of analyzing multiple sources of evidence or signals such as text and images in order to classify the overall sentiment of the text alongside any relevant emotions expressed within the image context. While there are many variations of this task, common approaches include text classification followed by fine-tuning of the classifier for each modality's specific attributes using image information.

Therefore, let us define some key terms:

- Input Text: This refers to the sentence or phrase which contains a sentiment that needs to be classified. 
- Visual Context: This consists of one or more images associated with the given text.
- Label/Classification: This refers to the predicted sentiment of the input text along with any applicable emotion(s). For example, positive, negative, or neutral.

## Dataset
For evaluating the performance of our approach, we use a publicly available dataset called "Sentiment Analysis on Twitter" [3] consisting of tweets annotated with sentiment labels and emotion tags. It includes tweets containing positive, negative, and neutral sentiments labeled according to five categories including happiness, sadness, love, surprise, disgust. These annotations are obtained through crowdsourcing and involving volunteers who label each tweet based on a pre-defined set of criteria. Each tweet can also be tagged with up to three additional emotions or traits corresponding to the descriptive expression displayed in the post, including anger, fear, joy, or sadness.

We split the dataset into training and testing sets with 70% and 30%, respectively. Additionally, we randomly sampled another 10% of the dataset to create a validation set for hyperparameter tuning.

The preprocessed version of the dataset includes two subsets:

- Tweets: A list of texts representing individual tweets with their respective sentiment labels and emotion tags.

- Images: A directory of directories containing separate folders for each tweet ID, wherein each folder contains images related to that particular tweet.

Both subsets were converted into TFRecord files for efficient reading during training and evaluation.

## Pretrained Embeddings
To improve the accuracy of our approach, we leveraged pre-trained word embeddings trained on large corpora of text to initialize the embedding layer of our model. Specifically, we used GloVe vectors, which represent words as dense vectors of fixed size, to embed each tokenized word into a continuous vector space. Using pre-trained embeddings helps the model capture meaningful semantics and relationships between words and hence improves its performance. 


# 3.Approach

Our approach involves adapting the existing DenseNet model for the task of MSA by adding a new branch that takes in the concatenated output of the last few convolutional layers of the base model along with the embedded representation of the input text. In contrast to traditional feedforward neural networks (FNN), DenseNet uses residual connections, allowing gradients to propagate back through the entire network without vanishing. By concatenating the outputs of all the layers together before passing them through a final linear layer, our model captures high-level representations of both textual and visual features and fuses them into a single representation for prediction.

## Architecture
DenseNet is a convolutional neural network architecture developed by Huang et al., which employs a modified bottleneck design with regular strides instead of pooling operations, resulting in a smaller computational footprint compared to traditional CNNs. In addition to several downsampling steps, DenseNet has a series of dense blocks, where each block consists of a set of bottleneck layers connected via short-cut connections, enabling it to learn rich hierarchical features over increasing receptive fields. Finally, the network feeds the combined feature maps into a final fully connected layer for binary classification of the input.


Our proposed modification to DenseNet adds a new branch at the end of the network that combines the output of the last few convolutional layers along with the embedded representation of the input text. Let $\{h_{i}\}_{i=l}^{m}$ denote the $m$ intermediate feature maps generated after the $l$-th layer of the base model, where $l\leq m$. Then, we concatenate the following tensors:

$$C = [\overline{h_m}, \overline{h_m-1},..., \overline{h_l}] \bigoplus h_{text}$$

where $\bigoplus$ represents concatenation, $\overline{h_k} = BN(\sigma(ReLU(h_k)))$, and $\sigma$ denotes the sigmoid activation function. Here, $BN$ denotes batch normalization, $\sigma$ denotes the sigmoid activation function, and $ReLU$ denotes the rectified linear unit activation function. Note that since the output dimensions of the base model depend on the number of filters applied during training, we need to ensure that the dimensionality of the concatenated tensor matches the desired output size.  

Next, we apply a fully connected layer with softmax activation to produce the probability distribution over the classes (positive, negative, or neutral).

Finally, we train the model using mini-batch gradient descent with Adam optimizer with weight decay and early stopping. During training, we evaluate the performance of the model on the validation set every epoch and stop training when no improvement is observed for a certain number of epochs. 

## Hyperparameters

We experimented with several hyperparameters for our approach, including the learning rate, batch size, number of filters, depth of the network, and dropout rate. Initially, we used a learning rate of 0.001, a batch size of 64, 16 filters per convolutional layer, a total depth of four blocks, and a dropout rate of 0.5. After comparing the results across different settings, we found that these values worked well for achieving good performance on the test set.

# 4.Results

Let us now evaluate the performance of our model on the test set. Below are the metrics used to measure the quality of our model:

- Accuracy: This measures the proportion of correctly classified examples among the total number of examples.

- F1 Score: This calculates the harmonic mean of precision and recall, providing a balance between precision and recall.

- Precision: This measures the proportion of true positives among the predicted positives.

- Recall: This measures the proportion of true positives among the actual positives.

Additionally, we calculate the confusion matrix and visualize some sample predictions.