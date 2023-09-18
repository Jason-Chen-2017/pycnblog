
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Eye gaze is one of the most effective cues for human attention and decision making in many situations. Despite its importance, understanding how humans interpret eye movements has not been well addressed by researchers yet. In this paper, we aim to provide a comprehensive review on current techniques that can be used to understand eye gaze behavior. This survey will cover various topics including: (a) datasets and annotation formats, (b) feature extraction algorithms, (c) machine learning models, (d) evaluation metrics, and (e) applications. Moreover, we will briefly discuss open problems and future directions of research in this field. Finally, our goal is to inspire developers to contribute new methods for understanding eye gaze behaviour using existing resources and promote technological development in this area towards real-world benefits. 

To begin with, let's go over some basic concepts and terminologies related to eye gaze behavior: 

1. **Fixation**: When an object or part of the environment is being looked at directly without any change in position or orientation.

2. **Saccade**: The gradual movement of eyes from one fixation point to another, which is caused by the pupil shrinking and then expanding as they pass through the focal spot. 

3. **Pupil diameter**: It is a measure of the size of the iris, i.e., the inner surface of the lens that surrounds the cornea. 

4. **Pupil distance**: The distance between the center of two adjacent lens elements along their common axis, measured from the visual axis. 

5. **Pupil aspect ratio**: Ratio of width to height of the pupil relative to the vertical line passing through the midpoint of it. 

6. **Head pose**: The rotation of the head about the x, y, and z axes resulting from pitch, yaw, and roll angles respectively. Head pose information helps to correct for biases due to the lack of physical motion sensors available in all devices. 

7. **Gaze vector**: The direction of gaze points from the person's retina to the target object/scene element, also known as "velocity" or "motion". 

8. **Binocular disparity**: The horizontal difference between the left and right eyes' fixations in pixels, which indicates the relative position and depth of objects seen in both eyes. 

9. **Glance**: An unobtrusive look at something or someone, usually performed quickly before more serious action or exploration occurs.

Now, let's move onto discussing various datasets and annotation formats used to collect eye tracking data: 

1. **The Internet Movie Database (IMDB) dataset:** This dataset contains movie trailers and comments labeled with start and end times for every gazed scene during watching. We can use this dataset to train machine learning models for automated eye tracking tasks such as emotion recognition and sentiment analysis. However, since noisy annotations exist within this dataset, it may have limited generalizability. 

2. **The BIWI Hand Gesture Dataset (HG):** This dataset contains hand gestures performed by subjects while looking at different objects, scenes, or perspectives. It provides rich contextual information about each gesture and can potentially be used for advanced computer vision tasks such as activity recognition and manipulation detection. Since these gestures are made of fixed patterns rather than natural ones, however, there is no perceptible delay involved and thus accurate timing cannot be inferred automatically. Hence, these techniques should only be considered when the subject's intentions are clear and the camera setup is controlled precisely. 

3. **The National Institute of Health (NIH) Tobii Pro SDK dataset:** This dataset consists of videos captured using Tobii Pro eye trackers and includes recordings of participants performing a variety of eye gaze behaviors such as smoking, shaking hands, reading text, taking notes, and glancing around the screen. It is large and diverse enough to serve as a benchmark for developing robust and reliable eye tracking systems. However, collecting high quality data requires careful experimental design and a dedicated team of experts who carefully observe subjects throughout the experiments. 

4. **The Penn Action Lab CrowdCounting dataset:** This dataset contains a collection of images depicting people gathered at homes, schools, offices, etc., annotated with bounding boxes of individual faces along with frame-level eye tracking data corresponding to each box. We can utilize this dataset to develop deep learning based models for automatic face counting and awareness estimation. Additionally, we can explore ways to leverage synthetic data generation and self-supervised pre-training approaches to improve performance further. 

5. **The Oxford IIT Pet Dataset (IIT-P):** This dataset consists of image sequences of pet owners demonstrating different types of interaction with their animals. It covers a wide range of complex scenarios involving different body positions, facial expressions, and actions. By leveraging the timestamp information associated with each sample, we can perform continual learning and adapt our models to new domains and users as they appear. 

Let's now focus on exploring various feature extraction algorithms used to extract relevant features from eye tracking data: 

1. **Surface reconstruction from eye position data:** One of the first steps taken by many eye tracking researchers is to reconstruct the user's eye surface geometry using the recorded eye positions and interpupillary distances (IPDs). Popular surface reconstruction techniques include Principal Component Analysis (PCA), Thin Plate Spline (TPS), or Volumetric Geodesic Active Contours Model (VGA-CM). These techniques enable us to capture complex geometries like curved surfaces or wrinkles better compared to simple mesh representations. However, these techniques require additional assumptions about the shape and distribution of the eye surface geometry, which can lead to imprecise estimates in challenging scenarios like occlusions and varying illumination conditions. 

2. **Feature detectors based on pixel intensity differences:** Another approach involves extracting statistical features from the pixel intensities at specific locations across the user's eye surface. Some popular techniques include Local Binary Pattern (LBP), Histogram of Oriented Gradients (HOG), Convolutional Neural Networks (CNN), or Edge Boxes. These techniques are often less computationally expensive and simpler to implement compared to other feature extraction methods but suffer from low degree of freedom. Specifically, HOG and LBP typically generate sparse feature vectors whereas CNN and Edge Boxes tend to produce dense feature maps with little fine grained details.

3. **Dynamic programming and time series modeling:** Dynamic programming algorithms such as HMM or Viterbi decoding can be applied to identify the most likely sequence of fixations given a sequence of gaze positions and pupil diameters. Time series modeling techniques like Hidden Markov Models (HMM) and Recurrent Neural Networks (RNN) can be trained to model temporal dependencies among the extracted features, enabling us to identify distinct trajectories and intentionality among the user's eye movements. These techniques allow us to capture multi-modal interactions and sophisticated spatial dynamics.

4. **Deep neural networks for representation learning:** Deep neural networks (DNNs) can be used to learn abstract representations of the user's eye movements. These models can process raw video frames or eye tracking data streams and learn highly discriminative features that are invariant to variations in lighting, viewpoint, expression, and occlusion. DNN architectures such as Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), or Transformers can be used to capture spatiotemporal dependencies among the input features. These models can potentially outperform traditional feature extractor algorithms by leveraging the power of deep learning and vast amounts of training data.

5. **Attention mechanisms and visual memory:** Attention mechanisms and visual memory can help us focus on the salient parts of the eye surface and maintain long term memory of previously encountered scenes and events. Recent advances in Transformer-based architectures for sequential processing have shown promise for addressing these challenges. We can integrate attention mechanisms into deep neural network models to selectively attend to important regions of the user's eye surface and retain long-term memory through external memory operations. Visual memory can also be implemented through external memory units that store historic inputs and outputs and can be accessed either explicitly or implicitly to reinforce past decisions and make better future predictions.