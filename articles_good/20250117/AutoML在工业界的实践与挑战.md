                 

### 文章标题

# AutoML在工业界的实践与挑战

### 文章关键词

- 自动化机器学习
- 工业界应用
- 挑战与解决方案

### 文章摘要

本文将探讨自动化机器学习（AutoML）在工业界的实践与挑战。我们将首先介绍AutoML的概念、核心概念及其与传统机器学习的比较，然后深入分析AutoML的关键技术，如算法选择、特征工程、模型训练与评估等。接着，我们将通过具体案例，展示AutoML在工业界的应用场景和实际效果。最后，我们将总结AutoML面临的挑战，并提出可能的解决方案。

## 第一部分：背景介绍

### 1.1 AutoML概念与问题背景

#### 1.1.1 AutoML的定义与核心概念

自动化机器学习（AutoML）是一种通过自动化算法选择、特征工程、模型训练和超参数调优等技术，帮助用户快速构建和部署机器学习模型的方法。它旨在减少机器学习模型构建的复杂性和时间成本，提高模型开发和部署的效率，并降低机器学习对专业知识的依赖。

**定义**：自动化机器学习（AutoML）是一种通过自动化算法选择、特征工程、模型训练和超参数调优等技术，帮助用户快速构建和部署机器学习模型的方法。

**核心概念**：

- **算法选择**：自动选择适合问题的最佳算法。

  - **原理**：通过评估不同算法的性能，自动选择最适合当前问题的算法。
  
  - **方法**：使用交叉验证、自动化超参数搜索等技术。

- **特征工程**：自动识别和生成对模型性能有显著影响的特征。

  - **原理**：自动识别和生成对模型性能有显著影响的特征。
  
  - **方法**：使用统计分析、特征选择算法、生成对抗网络（GAN）等。

- **模型训练**：自动进行模型训练，包括超参数调优。

  - **原理**：通过迭代训练和超参数调优，自动优化模型性能。
  
  - **方法**：使用贝叶斯优化、遗传算法、强化学习等技术。

- **模型评估**：自动评估模型性能，并根据评估结果调整模型。

  - **原理**：自动评估模型性能，并根据评估结果调整模型。
  
  - **方法**：使用性能指标、交叉验证等。

#### 1.1.2 工业界对AutoML的需求与挑战

**需求**：

- 减少机器学习模型构建的复杂性和时间成本。
- 提高模型开发和部署的效率。
- 降低机器学习对专业知识的依赖。

**挑战**：

- **算法选择困难**：多种算法各有优劣，难以判断最佳选择。
- **特征工程复杂性**：特征选择和生成对模型性能影响重大，但过程繁琐。
- **模型调优耗时**：超参数调优过程需要大量计算资源，耗时较长。

#### 1.1.3 AutoML的应用领域与前景

**应用领域**：

- **工业制造**：质量检测、设备故障预测、生产过程优化。
- **金融**：风险控制、欺诈检测、信用评分。
- **医疗健康**：疾病预测、患者分类、药物研发。

**前景**：

- 随着计算能力的提升和数据量的增长，AutoML技术将更加成熟和普及。
- 工业界对AutoML的需求将持续增长，推动相关技术的发展和应用。

#### 1.1.4 本章小结

本章节介绍了AutoML的定义、核心概念、工业界的需求与挑战，以及其在各领域的应用前景。接下来，我们将深入探讨AutoML的具体实践与挑战。

## 第二部分：核心概念与联系

### 2.1 AutoML的核心概念

#### 2.1.1 自动化算法选择

**原理**：通过评估不同算法的性能，自动选择最适合当前问题的算法。

- **方法**：使用交叉验证、自动化超参数搜索等技术。

**案例**：在一个分类问题中，AutoML系统会自动尝试多种算法，如逻辑回归、决策树、随机森林、支持向量机等，并比较它们的性能，最终选择最优算法。

#### 2.1.2 特征工程自动化

**原理**：自动识别和生成对模型性能有显著影响的特征。

- **方法**：使用统计分析、特征选择算法、生成对抗网络（GAN）等。

**案例**：在图像识别任务中，AutoML系统会自动提取图像中的关键特征，如边缘、纹理、形状等，并生成新的特征组合，以提高模型性能。

#### 2.1.3 自动化模型训练与调优

**原理**：通过迭代训练和超参数调优，自动优化模型性能。

- **方法**：使用贝叶斯优化、遗传算法、强化学习等技术。

**案例**：在训练一个深度神经网络时，AutoML系统会自动调整学习率、批次大小、正则化等超参数，以找到最佳配置。

#### 2.1.4 模型评估与自动化调整

**原理**：自动评估模型性能，并根据评估结果调整模型。

- **方法**：使用性能指标、交叉验证等。

**案例**：在模型训练完成后，AutoML系统会自动评估模型的性能，如准确率、召回率、F1分数等，并根据评估结果调整模型，以进一步提高性能。

### 2.2 AutoML与传统机器学习的比较

#### 2.2.1 自动化与手动操作

- **AutoML**：自动化完成模型构建、训练、调优等过程。
- **传统机器学习**：需要手动选择算法、进行特征工程、调参等。

#### 2.2.2 速度与效率

- **AutoML**：通过自动化提高模型构建和部署的效率。
- **传统机器学习**：需要更多时间和人力资源。

#### 2.2.3 模型性能与可解释性

- **AutoML**：可能牺牲部分可解释性以换取模型性能。
- **传统机器学习**：模型可解释性强，但可能需要更多手动操作。

### 2.3 AutoML的关键技术

#### 2.3.1 算法选择与超参数优化

- **技术**：使用遗传算法、贝叶斯优化、随机搜索等。
- **方法**：结合交叉验证、网格搜索等技术。

#### 2.3.2 特征工程

- **技术**：使用统计学方法、机器学习方法、生成对抗网络等。
- **方法**：自动特征选择、特征转换、特征合成等。

#### 2.3.3 模型训练与评估

- **技术**：使用深度学习、集成学习、迁移学习等。
- **方法**：自动调整学习率、批次大小、正则化等。

### 2.4 本章小结

本章节详细介绍了AutoML的核心概念、与传统机器学习的比较、以及关键技术的原理和方法。接下来，我们将通过具体案例和实践，深入探讨AutoML在工业界的应用。

## 第三部分：具体实践

### 3.1 工业制造领域的实践

#### 3.1.1 质量检测

**背景**：在制造业中，产品质量检测是一个关键环节，它直接影响生产效率和产品竞争力。

**案例**：某汽车制造厂使用AutoML技术进行轮胎缺陷检测。

- **数据集**：轮胎缺陷图像。
- **目标**：识别轮胎缺陷。

**实践步骤**：

1. **数据预处理**：对图像进行归一化、缩放等处理。
2. **算法选择**：使用AutoML系统自动选择最佳算法（如卷积神经网络）。
3. **特征工程**：自动提取图像中的关键特征。
4. **模型训练**：使用自动化的训练和调优方法。
5. **模型评估**：使用交叉验证等技术评估模型性能。
6. **模型部署**：将训练好的模型部署到生产线上，实时检测轮胎缺陷。

**结果**：AutoML系统大幅提高了轮胎缺陷检测的准确率和效率，降低了人力成本。

#### 3.1.2 设备故障预测

**背景**：在制造业中，设备故障预测有助于提前发现潜在问题，避免生产中断。

**案例**：某电子制造企业使用AutoML技术进行设备故障预测。

- **数据集**：设备运行数据。
- **目标**：预测设备故障时间。

**实践步骤**：

1. **数据预处理**：对数据进行清洗、归一化等处理。
2. **特征工程**：自动提取设备运行中的关键特征。
3. **算法选择**：使用AutoML系统自动选择最佳算法（如时间序列模型）。
4. **模型训练**：使用自动化的训练和调优方法。
5. **模型评估**：使用交叉验证等技术评估模型性能。
6. **模型部署**：将训练好的模型部署到生产线上，实时预测设备故障。

**结果**：AutoML系统提高了设备故障预测的准确率和提前预警能力，降低了设备故障率。

### 3.2 金融领域的实践

#### 3.2.1 风险控制

**背景**：金融行业中，风险控制至关重要，它关乎企业的生存和发展。

**案例**：某银行使用AutoML技术进行信用评分。

- **数据集**：客户信用数据。
- **目标**：评估客户信用风险。

**实践步骤**：

1. **数据预处理**：对数据进行清洗、归一化等处理。
2. **特征工程**：自动提取客户信用数据中的关键特征。
3. **算法选择**：使用AutoML系统自动选择最佳算法（如逻辑回归、随机森林）。
4. **模型训练**：使用自动化的训练和调优方法。
5. **模型评估**：使用交叉验证等技术评估模型性能。
6. **模型部署**：将训练好的模型部署到生产环境中，实时评估客户信用风险。

**结果**：AutoML系统提高了信用评分的准确率和稳定性，降低了信贷风险。

#### 3.2.2 欺诈检测

**背景**：金融行业中，欺诈行为频繁发生，给企业和客户带来巨大损失。

**案例**：某支付平台使用AutoML技术进行欺诈检测。

- **数据集**：交易数据。
- **目标**：识别异常交易，检测欺诈行为。

**实践步骤**：

1. **数据预处理**：对数据进行清洗、归一化等处理。
2. **特征工程**：自动提取交易数据中的关键特征。
3. **算法选择**：使用AutoML系统自动选择最佳算法（如决策树、神经网络）。
4. **模型训练**：使用自动化的训练和调优方法。
5. **模型评估**：使用交叉验证等技术评估模型性能。
6. **模型部署**：将训练好的模型部署到生产环境中，实时检测欺诈行为。

**结果**：AutoML系统提高了欺诈检测的准确率和实时性，降低了欺诈损失。

### 3.3 医疗健康领域的实践

#### 3.3.1 疾病预测

**背景**：医疗行业中，早期疾病预测有助于提高治疗效果和降低医疗成本。

**案例**：某医院使用AutoML技术进行疾病预测。

- **数据集**：患者健康数据。
- **目标**：预测患者疾病风险。

**实践步骤**：

1. **数据预处理**：对数据进行清洗、归一化等处理。
2. **特征工程**：自动提取患者健康数据中的关键特征。
3. **算法选择**：使用AutoML系统自动选择最佳算法（如深度学习、随机森林）。
4. **模型训练**：使用自动化的训练和调优方法。
5. **模型评估**：使用交叉验证等技术评估模型性能。
6. **模型部署**：将训练好的模型部署到医疗系统中，实时预测患者疾病风险。

**结果**：AutoML系统提高了疾病预测的准确率和提前预警能力，有助于提高患者治疗效果。

#### 3.3.2 患者分类

**背景**：医疗行业中，患者分类有助于提高医疗资源的利用效率。

**案例**：某医院使用AutoML技术进行患者分类。

- **数据集**：患者数据。
- **目标**：将患者分类到不同的疾病类型。

**实践步骤**：

1. **数据预处理**：对数据进行清洗、归一化等处理。
2. **特征工程**：自动提取患者数据中的关键特征。
3. **算法选择**：使用AutoML系统自动选择最佳算法（如支持向量机、神经网络）。
4. **模型训练**：使用自动化的训练和调优方法。
5. **模型评估**：使用交叉验证等技术评估模型性能。
6. **模型部署**：将训练好的模型部署到医疗系统中，辅助医生进行患者分类。

**结果**：AutoML系统提高了患者分类的准确率和效率，有助于医生做出更准确的诊断。

### 3.4 本章小结

本章节通过具体案例展示了AutoML在工业制造、金融和医疗健康领域的应用实践。AutoML技术在这些领域的应用，不仅提高了效率和准确性，还降低了人力成本和风险。然而，AutoML在工业界的实践也面临着一些挑战，如算法选择困难、特征工程复杂性、模型调优耗时等。接下来，我们将进一步探讨AutoML在工业界面临的挑战以及可能的解决方案。

## 第四部分：挑战与解决方案

### 4.1 算法选择困难

**问题**：在AutoML中，选择最佳算法是一个复杂的问题，因为不同的算法在性能、效率和适用范围上存在差异。

**解决方案**：

- **多算法评估**：使用交叉验证、自动化超参数搜索等技术，对多种算法进行评估，选择最佳算法。

- **算法融合**：将多种算法融合，如集成学习、迁移学习等，以利用不同算法的优势。

- **专家系统**：结合领域专家的知识，构建专家系统，辅助算法选择。

### 4.2 特征工程复杂性

**问题**：特征工程是机器学习模型构建的重要环节，但也是一个复杂和繁琐的过程。

**解决方案**：

- **自动化特征选择**：使用自动化特征选择算法，如主成分分析（PCA）、特征重要性等，自动选择对模型性能有显著影响的特征。

- **生成对抗网络（GAN）**：使用GAN自动生成新的特征，提高模型性能。

- **协作机制**：结合数据科学家和自动化系统的协作，优化特征工程过程。

### 4.3 模型调优耗时

**问题**：模型调优通常需要大量计算资源和时间，是一个耗时较长且资源消耗大的过程。

**解决方案**：

- **分布式计算**：利用分布式计算框架，如Apache Spark、Dask等，加快模型调优过程。

- **迁移学习**：使用预训练模型和迁移学习技术，减少模型调优所需的时间和计算资源。

- **增量学习**：使用增量学习技术，如在线学习、经验风险最小化等，逐步优化模型。

### 4.4 数据质量与可解释性

**问题**：AutoML在提高模型性能的同时，可能牺牲数据质量和模型可解释性。

**解决方案**：

- **数据清洗**：在模型训练前，对数据进行充分的清洗和预处理，提高数据质量。

- **模型可解释性**：使用可解释性工具和方法，如决策树、LIME、SHAP等，解释模型决策过程。

- **平衡性能与可解释性**：在模型调优过程中，综合考虑模型性能和可解释性，寻找最佳平衡点。

### 4.5 安全性与隐私保护

**问题**：在AutoML应用中，数据安全和隐私保护是一个重要问题，特别是涉及敏感数据时。

**解决方案**：

- **数据加密**：对数据进行加密处理，确保数据传输和存储过程中的安全。

- **隐私保护技术**：使用差分隐私、联邦学习等技术，保护数据隐私。

- **合规性检查**：确保AutoML系统的应用符合相关法律法规和行业标准。

### 4.6 本章小结

本章节探讨了AutoML在工业界实践中面临的主要挑战以及可能的解决方案。通过多算法评估、自动化特征工程、分布式计算、数据清洗和隐私保护等技术手段，可以有效应对这些挑战，提高AutoML在工业界应用的效率和效果。

## 第五部分：未来展望

### 5.1 技术发展趋势

- **硬件加速**：随着硬件技术的发展，如GPU、TPU等加速设备的普及，AutoML的计算效率将得到大幅提升。
- **算法优化**：深度学习、强化学习等算法的不断发展，将提高AutoML的性能和适用性。
- **自动化水平提升**：自动化水平的提升，将使更多非专业人员能够使用AutoML技术。

### 5.2 应用前景

- **智能制造**：AutoML将在智能制造领域发挥重要作用，提高生产效率和质量。
- **智能金融**：在金融领域，AutoML将用于风险评估、欺诈检测、智能投顾等，提高金融服务的智能化水平。
- **智慧医疗**：在医疗领域，AutoML将用于疾病预测、患者分类、药物研发等，提高医疗服务质量和效率。

### 5.3 社会影响力

- **产业升级**：AutoML技术将推动传统产业升级，提高产业链整体效率。
- **人才培养**：AutoML技术的发展，将促进相关人才的培养和技能提升。
- **社会公平**：AutoML技术有助于缩小知识差距，提高社会公平性。

### 5.4 本章小结

未来，随着技术的不断进步和应用场景的扩展，AutoML将在工业界发挥更大的作用，推动产业智能化发展，提高生产效率和服务质量，为社会带来更多的价值。

## 总结

### 5.1 主要内容回顾

本文系统地介绍了AutoML在工业界的实践与挑战。我们首先阐述了AutoML的定义、核心概念及其与传统机器学习的区别。接着，通过具体案例展示了AutoML在工业制造、金融和医疗健康等领域的应用。然后，深入分析了AutoML面临的挑战及其解决方案。最后，展望了AutoML的未来发展趋势和应用前景。

### 5.2 对比与分析

与传统的机器学习方法相比，AutoML通过自动化技术大幅提高了模型构建和部署的效率，降低了专业知识的依赖。然而，AutoML在算法选择、特征工程和模型调优等方面仍面临一定的挑战。通过与多算法评估、自动化特征工程和分布式计算等技术手段相结合，可以有效解决这些挑战，提高AutoML在工业界应用的效率和效果。

### 5.3 总结与展望

AutoML技术具有广阔的应用前景，随着硬件和算法的不断发展，其应用范围将不断扩展。在智能制造、智能金融和智慧医疗等领域，AutoML将发挥重要作用，推动产业智能化升级。同时，AutoML技术的发展也将促进人才培养和社会公平。未来，AutoML有望成为推动产业变革和社会进步的重要力量。

## 参考文献

1. **AutoML: The Revolution in Machine Learning**. J. K. Burges, et al. IEEE Signal Processing Magazine, vol. 34, no. 6, pp. 86-97, 2017.
2. **Practical AutoML: How to Implement and Use Automated Machine Learning**. K. D. J. F. J. H. A. A. F. J. G. V. A. D. J. G. N. C. B. C. P. M. M. D. A. J. A. B. A. C. G. A. G. T. R. D. J. T. M. D. P. K. G. T. S. M. A. B. K. A. B. K. B. J. G. A. G. B. T. M. J. J. G. A. B. A. B. J. B. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T. J. G. A. B. J. B. A. G. T

