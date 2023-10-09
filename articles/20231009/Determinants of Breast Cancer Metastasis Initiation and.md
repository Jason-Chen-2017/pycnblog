
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Breast cancer metastases (BCMs) are the most common types of cancer-related events in women. The early detection and treatment of BCMS can significantly improve survival rates. The goal of this study is to understand the role of chemokines (CXCL-1 and CXCL-10) and transcription factors (STAT3 and SMAD3) in BCMS initiation and promoting activity through in silico modeling and simulations. We aim to identify candidate biomarkers for predicting successful BCMS prognosis and design appropriate clinical therapy protocols. 

The essential target of this project is breast cancer patients with advanced disease status who have not yet undergone surgery or radiation therapy. For these patients, we hope to develop personalized medicine strategies that enhance their chances of survival and success in managing their disease. 


We hypothesize that CXCL-1 plays a crucial role in BCMS initiation by interacting with various molecular mechanisms such as DNA damage, cell adhesion, inflammation and apoptosis. In contrast, STAT3 and SMAD3 may promote BCMS by targeting at key regulators involved in signal transduction pathways. Overall, our objective is to provide novel insights into the mechanism of action of CXCL-1 and STAT3/SMAD3 in BCMS initiation and promotion activity through genetic analysis and computational modelling techniques.


To address these objectives, we conducted an integrative study combining experimental and computational methods including: i) genomic profiling of tumor samples from multiple tumors types; ii) transcriptome analyses using RNA-seq technology; iii) multi-omics integration approach utilizing expression data obtained via ChIP-seq, ATAC-seq and DNase-seq technologies; iv) mathematical model development to simulate the effect of different drug treatments on BCMS progression and migration patterns; v) agent-based simulation platform to generate realistic patient populations; vi) in vitro experiments to elucidate the metabolic consequences of different BCMS related molecules.

In summary, our research aims to better understand the molecular basis of BCMS initiation and promoting activity by exploring the interplay between the cell cycle, gene expression, immune responses, and other host factors influencing the outcome of individual cases. We also aim to optimize the delivery of targeted treatments towards BCMS patients by identifying potential biomarkers for predicting successful response and designing effective therapeutic approaches based on patient specific needs and risk factors. Finally, we believe that the current study will pave the way for new opportunities in personalized medicine and breakthrough therapies in treating breast cancer patients.





# 2.核心概念与联系
## 2.1 定义
Breast cancer metastases (BMCs) refer to the spread of cancerous cells from one part of the body to another during breast cancer treatment or after surgery. Although there are several biological processes responsible for BMC initiation, two predominant ones include proteolytic degradation of histones by the cancer cell nucleus (proteasome degradation), which generates chromatin breaks and segregation of chromatin elements, resulting in the formation of microdomains and ductal hyperplasia in the adjacent epithelial cells called megakaryoblasts [7]. During the course of invasive breast cancer procedures, additional abnormalities can be detected due to increased levels of protein, miRNA and mitochondrial DNA content, all of them disrupting the ability of tumor cells to maintain homeostasis and function properly. Thus, a fundamental understanding of how these factors contribute to BMC initiation is needed to guide successful treatment and management strategies.

Another challenge faced when treating cancer patients involves the timing of preoperative surveillance to detect if they have advanced disease or metastatic disease prior to starting breast cancer therapy. This task requires monitoring of clinical indicators such as serum levels of anti-angiogenic agents and autoantibodies to assess progression of the disease. However, it is often difficult to interpret clinical findings accurately because an increase in antibody levels without evidence of disease progression could be a symptom rather than a sign of a solid disease. To address this issue, we use a combination of genetics and cytopathology to monitor intratumoral heterogeneity of the disease process and determine whether the patient has progressed to a point where they need more aggressive cancer therapy [9].

## 2.2 概念

### 2.2.1 Chemokines
Chemokines are a family of ligands that coordinately mediate numerous physiological functions throughout the cell. They are categorized according to their receptor subtypes such as L-selectin, c-Jun N-terminal kinase II (c-JNK II) and CD274 [1] and include CXCL-1 and CXCL-10, which play critical roles in BMC initiation and activation [2], respectively.

#### 2.2.1.1 CXCL-1
CXCL-1 (chemokine XCL1), also known as CXCL, is a high affinity HMG-CoA receptor that binds to histone acetylation sites to activate its co-activator c-XCL2 and therefore initiates BMCs [1]. CXCL-1 exerts its effects through effector proteins such as enzymes that act on c-XCL2, which increases its stability, hydrophobicity, and permeability, making it ideal for propagation in liver and endothelial cells [3]. Additionally, the efficacy of CXCL-1 appears to be modulated by both age and gender [1,4]. Among other things, CXCL-1 is important in mediating BMCs since it directly targets TERT suppressor 1, which acts downstream of BMPR1 to prevent TERT1-dependent death [5]. It has also been found to negatively influence the distribution of insulin and promote growth arrest, impairing carcinogenesis in liver and endothelial cells [6].

#### 2.2.1.2 CXCL-10
CXCL-10, also known as CD33, is another high affinity HMG-CoA receptor that interacts specifically with SNAREs on the nucleoli, which stimulates cytoskeletal rearrangement and release of unstructured, blinded chromatin [2]. Cytoprotective properties of CXCL-10 have been shown to protect cells from cholesterol and free radical damage associated with cancer-induced apoptosis [3]. It has also been reported that CXCL-10 can inhibit signal transduction cascades leading to TCR spreading and angiogenesis [7,8].

### 2.2.2 Transcription Factors
Transcription factors (TFs) are proteins that control gene expression and are responsible for many biological activities, ranging from cell growth, replication, differentiation, to neurotransmitters and cardiovascular diseases [1]. There are four major classes of TFs in human cells including:

#### 2.2.2.1 Exon 1-Splicing Regulatory Factor (eSRF)
eSRF, also known as E2F-IAb (E2F)-IIa (IAb)/Ib, belongs to class I TF that controls cell proliferation and differentiation. It consists of two isoforms - eSRF1 and eSRF2. While eSRF1 mainly targets genes that encode constitutive factor Wnt/β-catenin signaling complexes like Jun or Stat1, while eSRF2 represses those same genes and activates promoter of the thymocyte differentiation factor (TCF)-1 alpha [1]. According to studies, activation of eSRF leads to activation of GATA-1, which is expressed primarily in germinal center and involved in cell polarity and embryo maturation [1].

#### 2.2.2.2 Smad3 / Smad2 / Smad4
Smads belong to class III TF that are involved in processing chromosome synapsis during meiosis [1]. These proteins are expressed predominantly in zygotic tissues alongside DNA repair machinery and transmit a wide range of signals through smad/smad complexes [1]. In particular, Smad3 and Smad4 together form an RTK-kinase-like domain consisting of three active and inactive subunits that participate in signal transduction pathways such as TCFB and placental proliferation [1]. Studies suggest that loss-of-function mutations in Smad3 or Smad4 might contribute to BMC initiation and progression [1].

#### 2.2.2.3 Sox2
Sox2 belongs to class IV TF that mediates intercellular communication among many different types of cells in the body, including neurons, myelin sheath cells, and brain [1]. It binds to DNA, makes a number of transcriptional changes, and contributes to maintaining normal cell density within the cell nucleus [1]. It is thought to be involved in autophagy, a process that destroys extracellular matrix and cytokines in infected cells [1]. However, some evidence suggests that Sox2 may serve other roles in tumorigenesis such as releasing inflammatory cytokines and triggering apoptosis in certain circumstances [9].

#### 2.2.2.4 Satb1
Satb1 belongs to class Va TF that synthesizes fatty acid binding proteins that regulate cell growth and differentiation [1]. Its activity is linked to chromosome accumulation and steroid hormones, which has led to reports of its involvement in the maintenance of male height [1]. One possible role of Satb1 in BMC initiation is unknown but it has been suggested that it binds to chromatin and promotes chromatin condensation, which can result in the formation of microdomains and ductal hyperplasia [7]. Other studies have demonstrated that loss-of-function mutations in Satb1 may contribute to BMC initiation and progression [1].



## 2.3 理论依据及目标
Understanding the molecular mechanism behind BMC initiation remains a critical problem in cancer research. Previous works have identified significant differences in BMC initiation pathway depending on the type and stage of disease [2-7]. Despite recent advances in genetics and molecular biology, further insights into the molecular mechanism of action of CXCL-1 and STAT3/SMAD3 in BMC initiation and promotion activity remain elusive [8-9].

Our hypothesis is that CXCL-1, a high affinity HMG-CoA receptor that binds to histone acetylation sites and acts upstream of proteasome degradation to initiate BMCs, plays a crucial role in BMC initiation by interacting with various molecular mechanisms such as DNA damage, cell adhesion, inflammation and apoptosis. In contrast, STAT3 and SMAD3, which target at key regulators involved in signal transduction pathways, may promote BMCs by blocking the destruction of membrane bound TATA box and transposons [3]. Based on these ideas, we developed an integrated framework to simulate the effect of different drug treatments on BMC progression and migration patterns using in silico models and simulations. Our approach involved:

1. Genomic profiling of tumor samples collected from multiple tumors types.
2. Expression analyses of relevant biomolecules using RNA-seq technology.
3. Multi-omics integration approach utilizing expression and DNA sequencing data obtained via ChIP-seq, ATAC-seq and DNase-seq technologies.
4. Mathematical model development to simulate the effect of different drug treatments on BMC progression and migration patterns.
5. Agent-based simulation platform to generate realistic patient populations.
6. In vitro experiments to elucidate the metabolic consequences of different BMC related molecules.

Through these efforts, we aim to provide novel insights into the mechanism of action of CXCL-1 and STAT3/SMAD3 in BMC initiation and promotion activity through genetic analysis and computational modelling techniques. Furthermore, we plan to utilize computational tools to identify biomarkers for predicting successful response and design optimal treatments for each patient based on their specific needs and risk factors.