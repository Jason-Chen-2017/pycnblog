
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


The term "genetic variation effect" refers to any type of genetic modification that changes the normal sequence of DNA in a way that leads to abnormal phenotypes or affects gene expression in an undesired manner. The goal is to identify and understand these modifications so they can be targeted by drugs for disease prevention or treatment. However, predicting the impact of genetic variants on various biological processes such as gene regulation, transcriptional control, protein synthesis, etc., remains challenging because it requires understanding of complex underlying mechanisms at multiple scales across cell types, tissues, organs, and organisms. 

In this paper, we will review four popular tools for genetic variation effect prediction: SIFT (SureSite Interference Factors), Polyphen-2, dbNSFP, and Annovar. We will analyze their performance in terms of accuracy, speed, scalability, ability to handle large datasets, and other relevant metrics. Based on our analysis, we will provide insights into how each tool works under the hood and suggest future directions for improving their performance further. In addition, we will discuss potential limitations of existing approaches with respect to handling new data types and non-model systems, and propose alternative algorithms or techniques to address them. Overall, we hope to inspire researchers, developers, and clinicians to build more powerful and accurate models for predicting the effects of genetic variants on various biological processes using novel machine learning techniques.


# 2.Core Concepts and Connections
Genetic variation effect prediction is a critical task in medical genetics, bioinformatics, and molecular biology. To date, several methods have been proposed to accomplish this task. Popular ones include SIFT (SureSite Interference Factors), Polyphen-2, dbNSFP, and Annovar. Here are brief descriptions of each method along with key features and advantages over others. 

**SIFT** - Predicts whether missense mutations are likely to cause disease based on the presence of nucleotide substitutions within certain regions of the site or the predicted effect of different mutational contexts. It is sensitive to small variations in splicing patterns, which makes it suitable for identifying single nucleotide polymorphisms (SNPs) and microdeletions that may result from spliceosome remodelling events.

**Polyphen-2** - A classification system that assigns three levels of severity to the impact of single nucleotide polymorphisms (SNPs) on protein function based on combinations of conservation, similarity, and hydrophobicity scores of affected residues. Its main advantage is its high specificity and sensitivity, making it well suited for detecting damaging SNPs in proteins and promoters but less suitable for interrogating larger sets of sequences due to its relatively slow computational time and high memory requirements.

**dbNSFP** - An updated version of Polyphen-2 that provides predictions for more than six million human genes with functional annotations. It integrates annotations from numerous sources including gene ontology, CADD phred score, mutation hotspot, population frequency, 1000 genomes allele frequency, cBioPortal risk scores, GERP++ scores, and some of the most commonly used databases like COSMIC, ClinVar, OMIM, MedGen. Unlike previous versions, dbNSFP is highly scalable, able to process millions of entries per second and utilize distributed computing resources efficiently. Despite its many benefits, however, dbNSFP's computational complexity limits its use to smaller batches of input sequences or for non-model systems where model building is not feasible.

**Annovar** - An ensemble variant annotation and selection software package consisting of several modules. Firstly, it annotates genomic variants based on multiple algorithms, including SIFT, Polyphen-2, LRT, and MutationTaster. Secondly, it filters out low-impact variants based on preset thresholds, consensus signature, and read depth profile. Thirdly, it selects specific annotated information based on user preferences and produces output files including summary statistics, gene-specific scores, and filter status. The program has become one of the most widely used tools in the field and is capable of processing tens of thousands of records per minute. The main limitation of Annovar is that it requires manual curation of results and cannot generate reliable predictions on newly discovered mutations without access to older knowledgebase. 


# 3.Algorithm Principles and Details
We now discuss the technical details behind each algorithm mentioned above. For all the below explanations, assume that we want to predict the impact of a single nucleotide change at position i in a given DNA sequence s. Let d be the deleted base, r be the replaced base, and j be the inserted base. This section contains detailed mathematical formulas and pseudocode for each algorithm.

## 3.1 SIFT Algorithm

The SureSite Interference Factor (SIFT) algorithm was developed to predict whether missense mutations are likely to cause disease based on the presence of nucleotide substitutions within certain regions of the site or the predicted effect of different mutational contexts. It is a statistical predictor based on machine learning techniques.

### Model Overview

SIFT uses a probabilistic model that takes into account the background distribution of DNA sequence and the pattern of nucleotide substitutions within a region around the affected site. Specifically, the probability distributions of nucleotide composition at each position of the motif surrounding the affected site are modeled using multinomial mixture model. In addition, we also consider the codon usage bias, the effect of the neighboring positions on the final amino acid translation, and the context-dependent nature of intronic or exonic mutations.

To obtain these probabilities, we first divide the DNA sequence into k contiguous segments of length l and extract the kmers starting at position i and ending at position i+l-k. Each segment represents a local context and is represented as a vector of frequencies of bases (A, T, C, G). These vectors represent the background distribution of DNA. Then, we train a logistic regression classifier on binary labels indicating whether a particular mutation occurred at a particular site or not. During training, we optimize the parameters of the classifier using stochastic gradient descent and cross validation technique to avoid overfitting.

### Probability Formula

Given a local context c = [c(i−j):j] and a mutation p = dri or p = rjd, the probability of observing the observed mutation m = djj after conditional sampling is calculated as follows:

Probability(m | c) = P(d|r,j)*P(j|r)*P(d,r,j)/P(d,r|c)*P(r|c) + 
                   P(d|r',j)*P(j|r')*P(d,r',j)/P(d,r'|c)*P(r'|c)
            ≈ P(d|r,j)*P(j|r)*P(d,r,j)/[P(d,r|c)+P(d,r'|c)] *
               [(P(r|c)+P(r'|c))/(P(r|c)+P(r'|c))]
             
            Note: For clarity, we assume P(d,r,j) ~ P(d|r)*P(r|j)*P(j) instead
                  of explicitly calculating all the individual components.
                  Also, for numerical stability reasons, we take logs of 
                  probabilities when performing multiplications and divisions.
                  
### Alternate Algorithmm Approach

An alternate approach to calculate the probability formula would be to directly estimate P(m|c) instead of going through the full computation chain. This could be done using Bayesian inference with prior beliefs about the nucleotide frequencies and distance between two substitutions affecting the same codon. Instead of estimating P(d,r,j), we only need to estimate P(d|r)*P(r|j)*P(j), which can be obtained using maximum likelihood estimation.

### Prediction Formula

After obtaining the probabilities for all possible mutations m, we select the top five most significant ones according to the following criteria:

 1. If the mutation involves a deletion of two or more bases, choose the one with highest probability.
 2. Choose the one with lowest absolute difference between estimated and true synonymous mutation rate.
 3. Choose the one with largest reported score from other programs (e.g. PolyPhen-2, PhastCons).
 4. Choose the one that does not involve loss of stop codons and leaves no frameshift.
 5. Randomly break ties if necessary.

Finally, we assign a category of benign or pathogenic depending on the sum of probabilities assigned to selected mutations. We set the threshold of significance at 0.05. Any mutations with lower probability are considered non-significant. Finally, we return a list of ranked mutations sorted by decreasing order of importance.