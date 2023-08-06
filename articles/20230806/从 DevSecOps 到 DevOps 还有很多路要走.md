
作者：禅与计算机程序设计艺术                    

# 1.简介
         
19年前，最初DevSecOps并没有像今天这么火爆，只是在开源社区里被提起、被讨论。2017年，当全球出现反对DevOps的声音时，微软、亚马逊、Facebook、谷歌等公司纷纷响应，开始重新定义DevOps理念，将其作为自己的主要工作流。DevSecOps也成为这一波热潮中的一个重要词汇。
           在DevSecOps流派中，安全和合规一直都是整个流程中最核心的内容之一。因此，今年DevSecOps开发者峰会上吴晓波等人就讨论了当前 DevSecOps 发展状态及未来发展方向。DevSecOp和 DevOps之间的关系，正在慢慢地变得模糊起来。
          在DevOps和SecOps之间究竟应该怎么做？这是一个值得思考的问题。本文希望通过分享一些知识和观点，回答这个问题。
           ## 1.1.DevSecOps 到底是什么
        DevSecOps（Development Security Operations）是一组运用 Devops 和 SecOps 的方法论，帮助组织实现应用开发、集成部署、运行管理、持续集成和监控的一系列安全操作。DevSecOps倡导一种安全工程思想，强调安全至上的研发过程。它是在现代化IT运维体系的基础上发展起来的一套综合性方法论，目的是加强开发人员和安全专业人士之间的沟通和协作。
         DevSecOps是一个完整的软件开发生命周期框架，涉及DevOps的各个方面，其中包括敏捷开发、持续集成/持续交付、基于容器的虚拟化、自动化运维、配置管理、安全和合规。同时，还包括密切关注安全的安全团队，能够确保所有的应用程序的安全性和可用性。
         DevSecOps推崇“安全和合规”作为开发过程中不可或缺的一部分，始终坚持“安全第一”，以此来提高产品质量和用户体验。
         ## 1.2.DevOps vs DevSecOps
         ### 1.2.1.DevOps
         DevOps是英文单词“Development Operations”的缩写，表示的是开发运营的一种实践方法。它促进软件开发（Dev）与IT运维（Ops）两个部门之间的紧密联系，目标是提供一个统一的环境，使两者之间的沟通和协作更加顺畅、高效。

         传统上，DevOps是一个高科技词汇，由国际标准化组织“Open Group”给出定义：

         “DevOps is a software development and IT operations collaboration that aims at automating the complexities of application delivery and infrastructure management.”

         2011年，美国IBM收购CloudFoundry基金会后，把DevOps作为公司的一个创新领域，其目标是让公司内部的工具和流程更加有效、流畅，提升云计算和业务服务的能力。

         随着DevOps的兴起，越来越多的公司选择将DevOps加入到他们的软件开发流程之中。例如，谷歌推出了Google Cloud Platform（GCP），Amazon Web Services（AWS）提供了免费的DevOps基金会，RedHat推出了OpenShift，微软也推出了Azure DevOps。

         ### 1.2.2.DevSecOps
         DevSecOps 是DevOps和SecOps的组合词，用于指导软件开发、集成、测试和部署工作的全过程，着重于构建安全可靠的软件，并消除软件开发过程中的安全风险。

         DevSecOps 运用DevOps的方法论，整合安全的最佳实践和做法，以达到提升产品质量和安全水平的目标。

         2016年，Symantec宣布推出了DevSecOps，其理念是通过使用DevOps和安全专业知识、工具、方法来增强开发人员和安全人员之间的沟通和协作。

         2017年DevSecOps开发者峰会上，麻省理工学院信息安全研究所的一些优秀演讲分享了当前DevSecOps的发展状况及未来发展方向，其中有一段讲到："DevOps and security are becoming increasingly intertwined as both businesses prioritize stronger security for their customers."，这句话直击了DevSecOps与DevOps之间的互相促进关系。
         
         意味着，未来，DevOps和SecOps在一起的时代已经到来，DevSecOps将成为新的关键词。