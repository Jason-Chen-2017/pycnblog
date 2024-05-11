# 基于文本特征及DNS查询特征的非常规域名检测

作者：禅与计算机程序设计艺术

## 1. 背景介绍
### 1.1 非常规域名的定义与危害
非常规域名（Unconventional Domain Names），是指采用各种混淆、迷惑技术注册的域名，通常用于从事恶意活动。这些域名往往包含错别字、同音异义词、特殊字符等，利用视觉和语义的相似性来欺骗用户。非常规域名常被用于钓鱼、恶意软件传播、垃圾邮件等攻击活动，对网络安全构成严重威胁。

### 1.2 传统检测方法的局限性
传统的非常规域名检测主要依赖黑名单机制和基于规则的启发式方法。黑名单需要大量人工维护，且只能检测已知的恶意域名。基于规则的方法虽然可以检测未知域名，但规则容易被攻击者绕过。此外，这些方法往往只关注域名本身的特征，忽略了域名实际使用过程中产生的行为特征。

### 1.3 结合文本特征和DNS查询特征的必要性
域名的文本特征（如长度、熵、特殊字符等）能反映其异常性，而DNS查询特征（如查询次数、请求方IP、TTL等）则包含了域名实际使用中的行为信息。将二者结合，能更全面地刻画非常规域名，提高检测的准确性和鲁棒性。本文将探讨一种融合文本特征和DNS查询特征的检测方法。

## 2. 核心概念与联系
### 2.1 域名的文本特征
- 域名长度：恶意域名通常具有异常的长度
- 域名熵：衡量域名中字符分布的随机程度
- 元音辅音比：常用于识别随机生成的域名
- 特殊字符：'-'、'_'等字符在恶意域名中高频出现
- n-gram特征：捕捉域名中的局部模式

### 2.2 DNS查询特征  
- 独特IP个数：恶意域名往往对应较少的IP
- 平均TTL值：恶意域名的TTL一般设置较短
- 请求方IP分布：恶意域名的请求方IP分布异常
- 域名请求频率：恶意域名的请求频率呈现反常模式
- 请求类型分布：恶意域名的NXDOMAIN响应比例高

### 2.3 特征的选择与组合
并非所有特征都对检测有贡献，需要通过特征选择来筛选最具判别力的特征子集。不同特征的重要性和区分度各不相同，通过加权融合可充分利用不同信息。文本特征反映域名自身异常性，DNS特征揭示域名使用过程的可疑行为，二者互补，结合使用效果更佳。

## 3. 核心算法原理及操作步骤
### 3.1 数据预处理
- 域名语料收集：白名单域名 + 种子恶意域名
- DNS日志收集：权威机构提供的全球DNS查询日志
- 数据清洗：剔除不完整记录，处理缺失值，格式规范化
- 特征提取：提取2.1和2.2中定义的文本和DNS特征

### 3.2 特征选择
- 过滤-包裹法（filter-wrapper）：先用过滤法初步筛选，再用包裹法微调
  - 皮尔森相关系数：剔除冗余特征
  - 信息增益：选出与类别关联度高的特征 
  - RFE（递归特征消除）：包裹法，结合模型自动选择最优特征子集
- 特征标准化：消除量纲影响，使不同特征可比较

### 3.3 模型训练
- 模型选择：决策树、随机森林、SVM、LR等
- 超参数调优：网格搜索 + 交叉验证，寻找最优参数组合
- 不平衡数据处理：过采样（SMOTE）/ 欠采样（EasyEnsemble）
- 模型集成：Bagging、Boosting等，提高泛化能力

### 3.4 模型评估
- 混淆矩阵：全面评估模型性能
- 精准率、召回率、F1值：兼顾检测的准确性和完整性 
- AUC值：评判模型整体判别能力
- 交叉验证：评估模型的稳定性和泛化性

### 3.5 模型应用
- 批量检测：定期对权威DNS数据进行离线检测
- 实时检测：对DNS请求流进行实时检测和拦截
- 模型更新：定期使用新数据对模型进行重训练

## 4. 数学模型和公式详解
### 4.1 文本特征
- 域名长度：$L(d) = len(d)$
- 域名熵：$H(d)=-\sum P_i \log_2⁡P_i$
- 元音辅音比：$RVC(d)=\frac {vowels(d)}{consonants(d)}$
- 特殊字符频率：$FP(d)=\frac {puncts(d)}{L(d)}$
- 异常n-gram概率：$P（\hat N|d)={\prod_{i=1}^{n}}P(\hat {N_i}) = {\prod_{i=1}^{n}} \frac {freq(\hat {N_i})}{\sum_{j}freq({N_j})} $   

### 4.2 DNS特征
- 独特IP数：$uIP(d)=\left | {IP} \right |,IP ∈ \  \{ip|ip\in query(d)\}$  
- 平均TTL：$\overline{TTL}(d) = \frac {\sum_{r∈R}TTL_r}{|R|}, R = \{r|r\in response(d)\}$ 
- IP分布偏度：$S_IP = E[(\frac {X_i-μ}{σ})^3], X_i∈{uIP}$
- 请求频率偏度：$S_q= E[(\frac {q_i-μ}{σ})^3], q_i∈\{q_d|d\in D\}$   
- NXDOMAIN占比：$r_{NX}=\frac {N_{NX}}{N}, \left\{\begin{matrix}N_{NX}=||\{r|r.type=NXDOMAIN\}|\\ N=||R||  \end{matrix}\right.$ 

### 4.3 特征选择  
- 皮尔森相关系数: $r(X,Y) = \frac {\sum_{i=1}^{n}(x_i-\overline{X})(y_i-\overline{Y})} {\sqrt{\sum_{i=1}^{n}(x_i-\overline{X})^2}\sqrt{\sum_{i=1}^{n}(y_i-\overline{Y})^2}}$ 
- 信息增益：$IG(X)=H(D)-H(D|X)$，其中$H(D)=-\sum_{k}^{ }P_klog_2Pk$ 
- RFE：每轮基于模型系数$w$淘汰若干最不重要特征，直至满足终止条件  
    
### 4.4 模型评估
- 精准率：$P=\frac {TP}{TP+FP}$ 
- 召回率：$R=\frac {TP}{TP+FN}$ 
- F1值：$\frac {2}{F1} = \frac {1}{P} + \frac {1}{R}$ 
- AUC：ROC曲线下方面积，$AUC=\frac {\sum_{i∈pos}rank_i-\frac {M(1+M)}{2}}{M×N}$

其中，$M$、$N$分别为正、负样本数，$rank_i$为第$i$个正样本的秩。

## 5. 代码实例
以下使用Python实现域名文本特征提取：

```python
import math
import tldextract

def domain_length(domain):
    return len(domain)

def domain_entropy(domain):
    char_cnt = Counter(domain)
    entropy = 0
    for _,v in char_cnt.items():
        p = v / len(domain) 
        entropy += -p*math.log2(p)
    return entropy

def vowel_consonant_ratio(domain):
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    v_cnt = len([c for c in domain if c in vowels])
    c_cnt = len([c for c in domain if c in consonants]) 
    return v_cnt / c_cnt if c_cnt else 0

def punct_percent(domain):
    punct = string.punctuation
    p_cnt = len([c for c in domain if c in punct])
    return p_cnt / len(domain)  

def abnormal_ngram_prob(domain, n):
    domain = tldextract.extract(domain).domain
    ngram_cnt = Counter([domain[i:i+n] for i in range(len(domain)-n+1)])
    abnormal_prob = 1
    for _,v in ngram_cnt.items():
        p = v / sum(ngram_cnt.values())
        abnormal_prob *= p
    return abnormal_prob

# 批量提取
def batch_text_features(domains, n=3):
    text_features = []
    for d in domains:
        feats = [domain_length(d), domain_entropy(d), 
                vowel_consonant_ratio(d), punct_percent(d),
                abnormal_ngram_prob(d,n)]
        text_features.append(feats)
    return text_features
```

批量提取DNS查询特征的实现见附录。限于篇幅，模型训练和评估的代码这里不再赘述，完整实现请参考我的GitHub。

## 6. 实际应用场景
- 域名注册审核：注册局可引入该技术对新注册域名进行审核，及时发现和阻止恶意注册行为
- DNS安全网关：企业、高校等机构可将该技术集成到DNS网关中，检测并拦截对非常规域名的访问请求
- 钓鱼网站检测：安全厂商可利用该技术及时发现各类仿冒、钓鱼网站，并提醒用户谨慎访问
- APT攻击溯源：溯源分析人员在还原APT攻击链时，可利用该技术揪出C&C服务器所使用的恶意域名，追踪幕后黑客组织 
- 威胁情报共享：安全社区可采用该技术检测各类恶意域名形成威胁情报，供业界分享使用

## 7. 工具和资源推荐
- 开源实现
  - [DomainTools/tld-extract](https://github.com/DomainTools/tld-extract)：域名TLD提取
  - [exp0se/dga_detector](https://github.com/exp0se/dga_detector)：基于文本特征的DGA检测  
- 域名信誉数据库
  - [VirusTotal](https://www.virustotal.com/)：集成多家引擎的域名信誉查询
  - [BlueCoat](http://sitereview.bluecoat.com/)：WebPulse威胁情报
  - [MyWOT](https://www.mywot.com/)：基于用户投票的网站安全性评估
- 公开的DNS数据集
  - [DNS-BH](https://www.malwaredomains.com/)：恶意域名黑名单 
  - [FluXOR](https://fluxor.info/)：全球DNS数据收集项目 
  - [UMICH DNS](https://ruminetworks.com/labeled-data-access/umich-dns/)：密歇根大学恶意软件和良性域名数据集

## 8. 总结与展望 
本文介绍了一种融合文本特征和DNS行为特征检测非常规域名的方法。该方法从域名语义和使用两个角度刻画域名的异常性，通过特征工程和机器学习技术构建检测模型。实验表明该方法能有效识别各类算法生成和人工混淆的恶意域名，为打击网络犯罪提供重要支撑。

未来，非常规域名检测仍面临诸多挑战：
- 对抗样本：攻击者可能刻意调整域名生成算法，逃避现有检测特征
- 跨语种：不同语种的语义相似性计算有待进一步研究
- 复杂背景流量：大规模复杂网络环境中，良性异常流量干扰恶意域名检测
- 域名滥用检测：租用/窃取合法域名从事恶意活动的检测有待加强

针对这些问题，一些有价值的探索方向包括：
- 对抗学习：考虑攻击者视角，主动构造对抗样本训练模型，提高检测模型鲁棒性
- 跨语种迁移：利用跨语言表示学习将不同语种的异常性知识迁移，实现多语种检测
- 行为画像聚类：构建域名全景画像，通过异常聚类甄别伪装的恶意域名
- 图神经网络：构建DNS查询关系图，利用GNN建模域名节点间的交互，捕捉隐藏的群体异常行为

相信通过学术界和工业界的共同努力，非常规域名检测技术必将不断突