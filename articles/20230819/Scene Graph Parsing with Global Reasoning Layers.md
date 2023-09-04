
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Scene graph parsing refers to the process of extracting visual information from an image and organizing it in a structured format that can be used for downstream tasks such as object detection or instance segmentation. In this work, we propose a new framework called “Scene Graph Parsing with Global Reasoning Layers” (SG-GRL) which takes advantage of global reasoning layers to represent objects and relationships between them. We use a transformer network architecture similar to GPT-2 to encode both textual and visual information into a fixed-size representation that captures context and spatial dependencies. The model is able to effectively extract relevant visual and semantic features through the fusion of global reasoning layers over all possible combinations of object and relationship embeddings. Experiments show that our approach outperforms prior state-of-the-art approaches on several scene graph benchmark datasets and achieves competitive results on two challenging real-world applications: visual question answering and action recognition in videos.
# 2.关键术语、概念及定义
1）scene graph（场景图）：A scene graph describes the structure of an image by relating individual objects together based on their position, size, color, shape, location, appearance, and behavior. It contains information about each object’s attributes and relations with other objects in the image.

2）object embedding（对象嵌入）：Object embeddings are vectors that capture high-level visual and semantic properties of each object in the scene. They are learned using convolutional neural networks trained end-to-end on large image datasets. For example, given an input image, we would train a CNN to produce output feature maps that correspond to various attributes of the image. These outputs could include bounding boxes, class probabilities, and other attributes such as whether the object has a person, hat, etc. Then, we would feed these feature maps into a fully connected layer at the end to obtain the final vector encoding of the object. Object embeddings are commonly generated from images using deep learning techniques, but they may also be obtained by humans labeling images or machine vision algorithms analyzing scenes.

3）relationship embedding（关系嵌入）：Relationship embeddings describe how one object is related to another. They contain information about the nature of the relationship, such as spatial position, directionality, temporal continuity, or whether there is a connection between the objects. Relationship embeddings can be inferred automatically using techniques like clustering or tag descriptions, but they can also be manually created or derived from domain knowledge.

4）global reasoning layer（全局推理层）：Global reasoning layers combine multiple inputs, including object embeddings and relationship embeddings, to generate more complex representations of the scene. Each layer consists of multiple fully connected layers with attention mechanisms, which allow different parts of the input to interact with each other during training and inference. Global reasoning layers enable the model to learn rich representations of the scene that are invariant to variations in viewpoint, scale, or pose.

The overall goal of SG-GRL is to jointly model both the visual and textual aspects of the scene while taking into account their mutual interactions. This allows the model to exploit the complementary strengths of each modality, leading to better performance compared to previous methods that solely focus on one modality.
# 3.算法流程及实现细节
Let's take a look at how SG-GRL works step by step. 

First, let's define some key terms:

Input Image : The input image contains visual information about objects in the scene along with textual annotations that specify relationships between those objects. Here's what a sample input might look like:


Attention mask matrix A: This is a binary matrix where Aij indicates if object i mentions object j in the caption annotation. If there isn't any mention of object j in the captions, then Aij = 0.

Caption Embedding : This is the output of a pre-trained language model that encodes the textual content of the image captions.

Object Features : These are extracted from an object detector network or other computer vision algorithm that identifies and localizes the objects present in the scene. The object features are usually produced by applying traditional convolutional neural networks to the raw pixel data.

One way to formulate the problem of relation extraction is to consider three types of relationships between pairs of objects: direct, indirect, and compositional. The most straightforward method to identify direct relationships between objects is to compare their geometric locations. However, detecting indirect and compositional relationships requires additional cues that cannot always be easily determined from just their positions. One common technique for inferring indirect and compositional relationships is to leverage multimodal contexts provided by natural language, especially when combined with visual information. To do so, we need to introduce some notion of object hierarchy, which represents the interconnectedness among objects in the scene. In practice, the object hierarchy can be constructed using heuristics or external knowledge bases, such as WordNet or Visual Genome.

Once we have defined the basic components of the problem, we move on to discuss the details of the proposed approach. Let's start with the Transformer Network Architecture:


In the above figure, X represents the concatenated sequence of objects and their corresponding features, Y represents the target prediction, M represents the attention mask matrix, K represents the number of objects in the current sentence, L represents the maximum length of the sequence, and H represents the dimensionality of the hidden states.

Each block is a self-attention mechanism applied to its own input, followed by a position-wise fully connected layer and a residual connection. The first self-attention layer generates Q, K, V matrices from the concatenation of the object features and their positional encodings, respectively. The second layer computes the output values O by multiplying QK^T, adding the position encodings, and applying a softmax function to each row of the resulting scores. Finally, the output of the second layer is fed into the third layer, which applies a linear transformation and adds back the original input. Since the model only uses relative distances between elements within each sequence, it doesn’t rely on absolute coordinates or orientation of elements.

Now that we have discussed the transformer net architecture, let's dive deeper into the specific implementation of SG-GRL using global reasoning layers. The idea behind global reasoning layers is to create a unified space for representing objects and their relationships in the scene. Specifically, we construct a hierarchical graph that connects objects by shared spatial relationships rather than directly linking them with edges. This hierarchical graph enables us to capture both the visual and semantic relationships between objects, making it easier for the model to predict the correct relationships without relying too heavily on local information alone.

To implement the global reasoning layers, we first compute the mean of all object embeddings. This creates a central node in the hierarchical graph that serves as a hub for all interactions. Next, we apply a set of recursive layers to expand the hierarchical graph around this center node. During training time, we alternate between updating the center node and recursively propagating updates throughout the entire graph. At test time, we simply traverse the tree upwards to obtain the predicted relationships. The visualization below shows the internal organization of the hierarchical graph:


As shown in the diagram, each object is represented by a node in the graph, and nodes are connected via directed edges indicating spatial relationships. At each recursion level, the edge weights indicate the degree of similarity between the associated objects. The depth of each node in the graph corresponds to its distance from the root node, and the width of each node reflects the number of children it has. By constraining the propagation of updates throughout the graph, we ensure that the model learns robust and consistent representations of the scene.

Finally, we concatenate the outputs of the global reasoning layers with the object embeddings and pass them through a classifier layer to obtain the final predictions. Overall, the objective of the SG-GRL model is to accurately infer both visual and semantic relationships between objects in the scene, taking into account the mutual influences of both modalities.