
作者：禅与计算机程序设计艺术                    

# 1.背景介绍



“GDPR”全称为“General Data Protection Regulation”，是欧盟于2016年通过的隐私保护法规。其宗旨是要求企业收集、存储、处理、传输和共享用户个人信息，并依据规定提供相应的个人信息保护服务。

目前很多金融科技公司也开始适应GDPR，例如微信支付，支付宝等。虽然公司已经适配了GDPR，但还是不能完全的认识到GDPR对于金融行业的影响。作为一个金融科技公司，如何合理遵守GDPR呢？这就需要了解一下GDPR各个方面的法律条款，以及如何让我们的产品或服务满足GDPR相关规定。

本文以微信支付为例，基于我对微信支付的一些理解和经验，结合GDPR的相关条款，试图回答以下几个问题：

1. GDPR对金融行业的影响是什么？为什么我们要适应GDPR？
2. 为了遵守GDPR，我们应该做哪些事情？哪些环节可以不适用GDPR？
3. 如果我们开发了一个新的产品或服务，该如何兼容GDPR？我们需要注意哪些方面？
4. 如果我们负责运营一个产品或服务，该如何收集、存储、处理、传输和共享用户个人信息？我们需要注意哪些方面？
5. 除了上述问题，我们还能从中学到哪些知识和技能？

# 2. Core Concepts and Interconnection 

## 2.1 Personal Information and Its Categories
Personal information refers to any information that can be used to identify an individual, either directly or indirectly. Examples of personal information include name, address, date of birth, email, phone number, bank account details, credit card numbers, identification documents such as drivers' licenses and passport photos, health records, etc.

Personal data is any piece of information relating to a specific or identifiable natural person who can be described in lawful ways. Personal data includes personal information and other information which could reasonably be linked back to a specific natural person without disclosing more sensitive information about them, for example:

 - Medical records related to an identified patient;

 - Financial transactions involving the same financial institution with similar transaction patterns over time;

 - Social security or tax information on a specific citizen’s income or deductions, if there are legal restrictions on sharing this information under certain circumstances.
 
In summary, personal data is any data that relates to you or your identity specifically, and may also contain information that has been combined with personal information from other sources.

Personal information categories typically consist of the following:

 - Identifying information: This category consists of information that identifies a natural person uniquely, including names, addresses, social security numbers, driver's license numbers, passport numbers, and unique biometric identifiers like fingerprints and facial recognition systems.
 
 - Contact information: This category contains information that allows someone to contact them, including email addresses, phone numbers, websites, physical mailing addresses, and telephone directories.
 
 - Device and usage information: This category comprises information about devices owned by or used by a natural person, such as IP addresses, device IDs, cookies, location data, search histories, browsing history, login sessions, transaction history, purchase history, or usage behavior data.
 
 - Account information: This category involves information associated with user accounts, such as usernames, passwords, payment methods, billing information, transaction activity, transaction history, marketing preferences, customer support interactions, and other identifying information related to user account access.

 - Professional information: This category contains information that is necessary to perform a job, professional relationship, educational pursuits, recruitment, criminal background checks, or employment applications, such as references, educational qualifications, experience, skills, language skills, certification information, and test scores.
 
 
## 2.2 Legal Basis for Collecting Personal Information
The right to collect personal information can come from different legal bases depending on the situation and purpose of collecting it. The most common legal basis for collection are:

 - Legitimate Interests: People have legitimate interests when they exercise their rights or make decisions based on a reasonable belief that those interests outweigh potential negative consequences. For instance, a website owner might want to receive a newsletter, but only if the content is relevant to them.

 - Consent: When people give their consent to a particular type of processing, their permission to do so explicitly demonstrates their agreement with the terms of the privacy policy. This often occurs when the person knows what kind of personal information will be collected and why it is being processed.

 - Contractual Obligations: When a company agrees to provide services to another party, the parties must ensure that the information provided is accurate and complete. Therefore, contractual obligations usually require explicit written permissions before the processing takes place.

 - Publicly Announced Requirements: Sometimes government regulations mandate public announcement of what types of personal information they plan to collect and how long they intend to keep it. If these requirements are unclear or not stated clearly, users are free to object to the use of their personal information.

Therefore, when we collect personal information, we should always carefully consider our legal basis and decide whether it meets the minimum required standard.

## 2.3 Sensitive Personal Information and Their Categories
Sometimes we need to handle sensitive personal information, such as medical diagnoses, family secrets, or political opinions. These are classified into four main categories:

 - Health Records: This category includes all health-related information, such as demographic information, diagnosis information, treatment plans, prescription medication orders, and insurance information. It is crucial for healthcare professionals to maintain control over their own health records, even though they are legally exempt from disclosure due to limited consumer protection laws.

 - Family Secrets: This category contains sensitive information about families and close associates, such as medical history, children’s biographies, criminal background checks, hobbies, philosophical beliefs, and romantic relationships. Family secrets are protected by various federal, state, and municipal laws, making it essential to follow proper procedures when dealing with this kind of information.

 - Political Opinions: This category includes information related to political activities and views, such as religious or ideological affiliation, political organization memberships, trade associations, current events, and industry trends. Political opinion information requires careful handling because it can expose individuals to dangerous situations and instigate violence.

 - Economic Activity Information: This category includes information gathered through market research and analyses, including stock prices, sales figures, business opportunities, investment strategies, and analyst recommendations. This category of information may be subject to stricter rules regarding data protection and security than other personal data categories.

It is important to classify and handle personal data according to its sensitivity level and protect it accordingly.

## 2.4 Access and Deletion Rights
Once we obtain the authorization of the user to process their personal information, we must give them the right to access their personal information and the ability to delete it at any time. However, some legal frameworks restrict deletion requests for specific reasons, such as:

 - Privacy offenses: If an individual breaks the law by trying to violate the privacy of others, they may face severe penalties up to and including termination of employment.

 - Protecting vital interests: Government agencies often seek to preserve sensitive personal information even after a person has left the country to safeguard vital interests.

 - Not having sufficient information: Even if we have obtained the authorization of the user to process their personal information, we cannot guarantee that they will provide us with enough information. We must therefore inform them of the expectations and challenges they will face while providing personal information.