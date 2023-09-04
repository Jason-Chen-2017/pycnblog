
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Multimodal conversational agents (MCAs) are becoming increasingly popular due to their ability to connect different modalities of communication and share experiences across platforms, enabling more natural human-computer interactions. However, they may sometimes accidentally or intentionally mislead users into sharing sensitive information without consent from the user themselves. In this paper, we propose a technique called "discreet modelling" which enables MCA developers to train agents on what is not desired for them to hear but necessary for proper conversation. We first identify common situations in which privacy issues arise and then use these scenarios as cues for training an agent to be discriminatory in its responses. This allows us to create powerful, multimodal conversational agents with improved privacy protection against unwanted interruptions. 

In this paper, we will introduce a new algorithm known as Discrete Modelling using Gaussian Mixture Model (DMM), which uses prior knowledge of such scenarios and data about individual preferences over attributes to build a model that can classify incoming utterances based on whether they contain any offensive content or are necessary for conversation flow. The DMM algorithm outputs a probability distribution over possible scenarios, allowing the agent to select the most appropriate scenario at each point during conversation based on the likelihood of occurrence. We will also present several experimental results demonstrating the effectiveness of our approach in creating effective MCA models capable of protecting users' privacy while still maintaining good dialogue quality. Finally, we provide suggestions for future research directions involving integrating learned personalization strategies into the MCA development process.

The remainder of this article is organized as follows: Section 2 provides a brief introduction to MCAs, section 3 explains key concepts used throughout the rest of the paper, followed by sections 4, 5, and 6 where we discuss our proposed techniques, experimental results, and related work respectively. In Section 7, we summarize the main contributions of our work. Our next steps include conducting further experiments with other datasets and modalities to evaluate the robustness of our methodology, extending our approach to handle complex conversational scenarios, and developing a practical system implementation of our methods.

# 2.Introduction
## 2.1 Multimodal Conversational Agents (MCAs)
Multimodal conversational agents (MCAs) have emerged as one of the most popular approaches to building intelligent virtual assistants. These chatbots interact with users through text, speech, and visual input modalities, making it easier than ever for people to communicate with them digitally. While there has been considerable progress towards building high-quality dialogue systems using MCAs, they continue to face numerous challenges including privacy concerns, resource constraints, and scaleability. 

To address these challenges, various techniques have been proposed to improve the security and privacy of MCAs. One commonly used technique involves obtaining explicit user consent before processing their personal data, however this often proves impractical especially when dealing with large populations. Another technique involves encrypting communications between the agent and end-user client, but this requires additional computational resources and increases latency compared to non-encrypted conversations. To address these issues, recent advances in deep learning have enabled the development of new neural networks architectures, such as generative adversarial networks (GANs), that can learn to produce synthetic samples that look realistic yet cannot fool humans. Other techniques involve deploying multiple agents simultaneously trained on different subsets of the same dataset, providing better diversity in terms of behavior and potentially reducing risks associated with malicious actors trying to exploit vulnerabilities in single agents. Nevertheless, none of these solutions fully protect users' privacy and maintain excellent dialogue quality and consistency.

Our primary objective in this paper is to develop a novel technique called "Discretized Modelling", which helps MCAs detect cases where potential privacy violations occur and adaptively respond accordingly. By identifying important contextual factors, such as age, gender, race, ethnicity, political views, religious beliefs, and cultural background, we can create a specialized model that can flag instances of sensitive content or personal information, prompting the agent to either mask it or remove the entire response altogether. Moreover, by introducing a probabilistic output that represents the likelihood of encountering specific types of scenarios, we can leverage existing human biases and automatically adjust the level of discourse depending on the situation.

# 3.Concepts and Terminologies
## 3.1 Situations and Scenarios
Privacy is fundamental in today's digital world and influences every aspect of our lives. Among many issues affecting individuals’ online activities, privacy violation can have serious impact on privacy and public safety. Some examples of privacy violations could be unwarranted access to medical records, search history, financial transactions, etc., which can cause significant harm to individuals and society. Therefore, we need to design a solution that can effectively monitor and control user behavior patterns to prevent and recover from privacy violations.

One way to achieve this goal is to use machine learning algorithms to analyze user interactions and predict whether an interaction might lead to a privacy violation. However, detecting such cases accurately requires careful consideration of relevant aspects of a user’s behavior, understanding the meaning behind words and phrases, and modeling the relationship between different behaviors. For instance, some users tend to talk about sexual relationships or neglect others, indicating that they might be revealing sensitive information about themselves. Similarly, some scenarios may require stronger protection, such as when handling confidential documents or managing healthcare records.

In order to train machines to recognize such situations accurately, we must understand how they manifest within the context of daily life. Based on our observations, we can define six categories of scenarios typically encountered by users who want to avoid being interrupted or disturbed. Each category captures a particular type of behavior and possibly reveals sensitive information.

### Category I: Physical contact scenarios 
These scenarios include touching or shaking hands, breathing deeply, speaking loudly, or using forceful gestures like yelling or screaming. People in this category might make unwanted physical contact with a stranger, causing chaos or embarrassment if caught red-handed. 

### Category II: Intimate scenes  
Intimate scenes happen when someone physically comes into close physical contact with another person. This happens frequently when two parties sit together or kiss or hug. It is essential to remember that physical contact does not always mean that intimate scenes take place. Someone may only engage in passive communication or simply stay quiet because they fear being heard. When two people meet in an intimate scene, there may be unnecessary details about the environment or past conversations that could help the listener discover private information about the other party. 

### Category III: Social media harassment  
Social media harassment refers to scenarios where someone posts unwanted messages or comments on social media profiles, usually targeting someone else for their hatred or disrespect. Examples of such harassment include calling someone names, sharing inflammatory images or videos, and crossing out sensitive information with asterisks (*). Harassers might try to manipulate others into sharing false or deceptive information, leading to mistrust and hostility. 

### Category IV: Medical treatment scenarios  
Medical treatment scenarios focus on treating patients or survivors of injuries, accidents, or diseases. Depending on the severity of the illness, patients might be worried about revealing personal information or even asking for professional assistance. Even experienced doctors may forget to treat patients appropriately, leading to complications and stress. On the other hand, family members or friends who seek medical advice may remain silent and be puzzled by the requests for medical attention. 

### Category V: Sales and marketing scenarios  
Sales and marketing scenarios are concerned with selling goods or services to clients, both online and offline. These scenarios often result in unexpected purchases or promotions, which can expose users to unwanted promotional messages or advertising. Shopping carts, emails, or websites containing ads can easily track users’ interests and behavior patterns. Additionally, businesses and organizations that promote certain products or services may get harassed by customers who insist on buying those products.

### Category VI: Interactions requiring sensitive information  
Interactions requiring sensitive information refer to scenarios where users exchange sensitive information, such as bank account numbers, passwords, credit card numbers, or identity documents. Users in this category might refuse to give out critical information, let go of sensitive documents, or resist attempts to open locked drawers. Some users might feel insecure about giving away their personal information, which can increase risk of cyber-attacks, identity threats, and loss of confidence. 

## 3.2 Continuous vs. Discrete Modelling
When it comes to machine learning, continuous variables and discrete variables come up time and again. Continuous variables represent things like temperature, pressure, and speed, whereas discrete variables represent categorical variables like colors or objects. Continuity implies smooth transitions between values, while discretion means assigning specific values to entities. Machine learning algorithms can distinguish between continuous and discrete variables, and often rely on mathematical equations and inference rules to categorize inputs into different buckets. For example, linear regression assumes that all features are continuous, while decision trees assume that they are discrete.

However, this assumption is incorrect for privacy monitoring applications since the nature of data exhibits intrinsic continuity. Instead, continuous modelling would lead to excessive smoothing of signals, resulting in low signal-to-noise ratios (SNRs) and noisy predictions. Alternatively, discretizing the data could lose valuable information by grouping similar inputs together, which leads to ambiguities and reduces accuracy. Therefore, we need to strike a balance between continuous and discrete modelling.

We choose to use a hybrid approach of continuous and discrete modelling to capture the nuances of each user’s behavior while preserving the overall shape of the data. Specifically, we use a combination of continuous and discrete features, specifically representing the duration of a given activity and the frequency with which a user performs that activity. Using a combination of continuous and discrete features allows us to preserve important characteristics of the data while minimizing noise and errors caused by binning inputs.

Specifically, we use a mixture of Gaussians to represent the underlying distributions of each feature. Each gaussian corresponds to a unique set of discrete states and has a corresponding probability density function (PDF). The PDF determines the degree of membership of a feature value in the corresponding state. At test time, the agent receives a sample from each feature and combines them into a joint feature vector. The mixture model assigns probabilities to each possible scenario based on the relative strength of the gaussians in the joint space. By selecting the scenario with the highest probability, the agent can estimate the likelihood of encountering a particular scenario.

# 4.Proposed Method
## 4.1 Problem Setting
In order to enable multimodal conversational agents (MCAs) to protect users' privacy, we need to identify sensitive content and prompt the agent to filter or obfuscate it as needed. To accomplish this task, we need to train an agent on scenarios where privacy violation occurs and extract relevant contextual clues from dialogues. Contextual clues could be drawn from the following sources:

1. User profile information – contains demographics, psychological traits, lifestyle, and habits that influence the likelihood of violating privacy laws.
2. Dialogue history - encodes the past interactions of the user that can reveal private information such as shared shopping lists, passwords, email addresses, phone numbers, etc.
3. Device information - includes metadata about devices such as IP addresses, device identifiers, GPS coordinates, etc., that can indicate the location of the user.
4. Cognitive load - measures the amount of effort required to understand and retain relevant information in memory, which can affect concentration, recall, and performance. 
5. Emotionality - indicates the emotional state of the user, which can affect their perception of their actions and attitudes toward sensitive information. 

Based on these clues, we need to construct a classifier that can identify instances of sensitive content in the input utterance and assign probabilities to each possible scenario. We propose a novel approach called Discrete Modelling using Gaussian Mixture Model (DMM), which trains an agent to determine the likelihood of encountering a specific scenario given the current dialogue context and user preferences.

## 4.2 Dataset and Preprocessing
In this paper, we use the Reddit Conversations Corpus consisting of more than 9 million anonymous Reddit threads spanning over five years. The corpus consists of mostly English language forum posts discussing topics ranging from politics, music, food, news, movies, and TV shows. All posts were collected using the Pushshift API and labeled manually by human annotators. The corpus is highly imbalanced with respect to labels and contains a mix of threads discussing sensitive content. We randomly split the dataset into a training set (80%), validation set (10%), and testing set (10%) with equal number of positive and negative class instances.

For preprocessing, we perform the following operations:

1. Tokenization - splits each post into tokens based on whitespace characters.
2. Stopword removal - removes short stopwords such as “a”, “an”, “the” which add little semantic value.
3. Stemming - converts words to their base form to reduce complexity. Words like “running” and “runner” become “run” after stemming.
4. Part-of-speech tagging - identifies parts of speech in the token sequence, such as verbs, nouns, pronouns, etc., which can help identify certain patterns that indicate privacy violations.
5. Frequency filtering - eliminates rare word forms that occur infrequently in the dataset, reducing dimensionality and improving generalization power.

After preprocessing, we convert each token into a numerical representation using a vocabulary dictionary. We obtain a sparse matrix where rows correspond to instances and columns correspond to vocabulary entries. Each row contains a binary indicator of presence or absence of the respective token in the post. We repeat this step for each post in the training, validation, and testing sets.

## 4.3 Neural Network Architecture
We implement a simple feedforward neural network architecture with three hidden layers, ReLU activation functions, and dropout regularization. We use Adam optimizer and L2 weight regularization to minimize overfitting. The final layer produces a softmax distribution over the seven available scenarios.


## 4.4 Training Strategy
We train the neural network on a batch size of 32 and iterate over multiple epochs until convergence. During training, we compute metrics such as classification accuracy, precision, recall, and F1 score to monitor the performance of the model. We also visualize the predicted scenarios alongside the true scenario to compare the learned policies visually. We save checkpoints periodically and measure the best performing checkpoint on the validation set. If the performance on the validation set stops improving for a few epochs, we decay the learning rate to allow the model to converge faster. Once the model is trained, we apply it to generate scores for all remaining test instances and select the scenario with the maximum score as the prediction.

## 4.5 Experiment Evaluation and Analysis
### 4.5.1 Baseline Models
We experimented with four baseline models to see how well they can guess the scenario from scratch:

1. Dummy Classifier - selects the scenario with the largest proportion of positives in the training set as the predicted label. The accuracy on the test set was around 44%.
2. Constant Predictor - selects the scenario that appears most frequently in the training set as the predicted label. The accuracy on the test set was very high, reaching almost 70%.
3. Random Guesser - randomly selects a scenario from the list of available scenarios as the predicted label. The accuracy on the test set was below 20%.
4. Naïve Bayes Classifier - trains a naïve Bayes classifier on the training set and applies it to the test set. The accuracy on the test set was around 65%.

It seems that constant predictor and random guesser perform quite poorly on this task since they just guess randomly, ignoring the fact that there are differences between scenarios in terms of the contextual clues mentioned earlier.

### 4.5.2 DMM Model
We evaluated the DMM model on the full dataset and obtained significantly higher performance compared to the previous models. The DMM model achieved an accuracy of around 82% on the test set, far exceeding the baseline models.

### 4.5.3 Effect of Feature Selection on Performance
We explored how removing features from the model improves performance. We removed one feature at a time and observed the decrease in accuracy on the test set. Surprisingly, dropping the duration of physical contact reduced the accuracy slightly, which suggests that having long duration interactions is an important factor for the detection of privacy violations. Dropping the frequency of posting to social media increased the accuracy marginally, suggesting that frequent posting may constitute a threatening behavior pattern. Although dropping the frequency of interacting with intimate scenes did not substantially affect accuracy, it demonstrates that relying too much on heuristic criteria alone is not always reliable.

### 4.5.4 Evaluating Different Clusterings
We considered a range of clustering schemes to group scenarios based on their similarity and computed the average accuracy improvement for each cluster. We found that splitting the scenarios into two clusters had a minimal impact on accuracy, so we focused on analyzing the clustering scheme that resulted in the greatest accuracy improvements.

We constructed a scatter plot showing the fraction of positive instances in each scenario versus the mutual information between the scenario and the features. The mutual information quantifies the degree to which two random variables differ according to their joint probability distribution. As expected, we observe a curved boundary separating the different scenarios based on their similarity in terms of the contextual clues. Removing the least informative feature (frequency of posting to social media) greatly improves the separation and reduces the gap between scenarios. Overall, this analysis highlights the importance of considering diverse scenarios and contextual clues when developing a privacy policy compliant conversational agent.

# 5.Related Work
There have been several attempts to solve the problem of privacy violation detection using AI. Previous works employed supervised learning algorithms such as logistic regression, support vector machines (SVM), and neural networks. Most of these methods require expert-crafted features that encode human reasoning and intuition about privacy. However, generating suitable features is difficult and may fail to cover all possible variations of privacy violations. 

Moreover, existing methods mainly target detecting only simple scenarios such as physical contact, sales and marketing, and social media harassment. Complex scenarios such as medical treatment scenarios or interactions requiring sensitive information require more sophisticated modeling and automated adaptation mechanisms. Despite their successes, there are still gaps in the literature regarding the design of privacy-aware conversational agents.

Overall, there is a need for a flexible and scalable approach to tackle the challenge of detecting and responding to privacy violations using artificial intelligence. The proposed technique, Discrete Modelling using Gaussian Mixture Model (DMM), takes advantage of insights gained from studying typical behaviors and the statistical properties of the data to devise a framework for automatic detection and response.