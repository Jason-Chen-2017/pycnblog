
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1. Introduction  
         Microsoft Azure Active Directory (Azure AD) Domain Services (AD DS) is a managed service on the Azure platform that provides managed domain services such as domain join, group policy, and LDAP access for your cloud-based resources. It also offers enterprise level identity management capabilities including password policies, multi-factor authentication, and secure LDAP (LDAPS). In this article, we will discuss how you can enable single sign-on across both products using Oracle Identity Cloud Service (IDCS).
         2. Prerequisites   
         Before starting with this tutorial, make sure that you have an active subscription of either Azure or Oracle Cloud Infrastructure(OCI), having at least one resource created under Azure AD or OCI tenancy and have appropriate privileges in those environments to complete all steps mentioned below. 
         3. Outline  
        - Background Introduction 
        - Basic Concepts and Terminology
        - Core Algorithms & Operations Steps
        - Code Examples and Explanation
        - Future Development
        - Common Questions and Answers
        - Conclusion
        
        # 2.Background Introduction
         ## Introduction to Oracle Identity Cloud Service 
         Oracle Identity Cloud Service (IDCS) enables organizations to manage user identities and establish federated trust relationships between multiple applications through a centralized authentication system. This solution provides several benefits such as strong authentication, seamless federation, scalability, high availability, and support for various protocols like OAuth 2.0, OpenID Connect, SAML 2.0 etc. Oracle IDCS helps organizations meet compliance requirements by providing identity governance and security auditing tools along with multi-factor authentication (MFA). The key features of Oracle IDCS include:

            1. User provisioning and deprovisioning
            2. Group management
            3. Authorization control
            4. Credential lifecycle management
            5. Multi-tenant application integration
            6. Federated SSO and federated identity
            More details about Oracle IDCS can be found here: https://docs.oracle.com/en/cloud/paas/identity-cloud/index.html

         ## Integrating Azure AD and Oracle IDCS
         To enable cross product single sign-on (SSO) using Azure AD DS and Oracle IDCS, follow these steps:

         Step 1: Create Users in Azure AD
         Firstly, create users in Azure AD tenant which you want to use for SSO login into Oracle Applications. You need to add them as guest users so that they can see only their assigned applications without any other permissions. 

         Step 2: Configure Application Groups in Oracle IDCS
         Next, configure application groups in Oracle IDCS so that they match the application names configured in Azure AD DS. Once done, grant read-only access to these application groups to the Azure AD groups where the corresponding users are added.  

         Step 3: Link Azure AD DS to Oracle IDCS
         Finally, link Azure AD DS with Oracle IDCS so that it can perform the necessary integration tasks. During linking process, specify the mapping between application groups in Oracle IDCS and Azure AD groups and set up the desired protocol for SSO authentication. After linking is completed successfully, users can log in to Oracle Applications using their Azure AD credentials via single sign-on functionality.

         Note: Ensure that you have followed all pre-requisites mentioned above before proceeding further.

         # 3.Basic Concepts and Terminology
         A few basic concepts and terminology used while integrating Azure AD Domain Services and Oracle Identity Cloud Service should be understood. Let's get started! 

            What is Single Sign-On (SSO)?

            Single sign-on (SSO) allows users to authenticate once using a single set of credentials and gain access to multiple applications without needing to provide separate credentials. With SSO, a user can simply enter his or her username and password once, and then automatically be authenticated for each subsequent transaction.

            Types of Single Sign-On Systems

            There are two types of Single Sign-On systems: first-party and third-party. First-party systems rely on client software installed locally on individual devices, whereas third-party systems leverage existing infrastructure such as web servers, proxies, and firewalls to provide SSO capability. In our case, we will focus on implementing a third-party Single Sign-On System called Oracle Identity Cloud Service (IDCS) that uses established enterprise-level protocols such as SAML 2.0 and OpenID Connect to integrate with Azure AD Domain Services (AD DS).
            
            What is Azure Active Directory (AD)?

            Microsoft Azure Active Directory (AD) is a comprehensive identity and access management service based on modern technologies like Windows Server Active Directory (AD), but more integrated and flexible than traditional directory solutions. It provides core user, device, and app management capabilities needed to support modern business needs. The following are some highlights of Azure AD:

                1. Centralized administration
                2. Seamless single sign-on (SSO)
                3. Secure hybrid deployments
                4. Advanced analytics and reporting
                5. Role-based access control
                6. Mobile device management

            What is Azure Active Directory Domain Services (AD DS)?

            Azure AD Domain Services is a managed service on the Azure platform that provides managed domain services such as domain join, group policy, and LDAP access for your cloud-based resources. It also offers enterprise level identity management capabilities including password policies, multi-factor authentication, and secure LDAP (LDAPS). In simple terms, AD DS provides identity and access management capabilities required for a secure and reliable user environment. 

                Key Benefits of Azure AD Domain Services

                As an integral part of Azure platform, AD DS offers several key benefits including:

                    Simple deployment
                    Built-in monitoring and alerting
                    Automatic updates and patches
                    Customizable forest structure and topology
                    Support for different platforms and languages
                We will now proceed with discussing the algorithm behind enabling cross product SSO using Oracle Identity Cloud Service.


         # 4.Core Algorithm and Operational Steps
         Now let's move towards the actual algorithm and operational steps involved in enabling cross product SSO using Oracle Identity Cloud Service.

        Algorithm: Cross Product SSO Using Oracle Identity Cloud Service
        
           Oracle Identity Cloud Service enables users to sign in to Oracle Applications using their Azure AD credentials provided they belong to a mapped Azure AD group. Here are the main steps involved in enabling cross product SSO using Oracle Identity CloudService:

1. Create Users in Azure AD

   Create new users in your Azure AD Tenant. These users must be Guest Users as well.
   
2. Configure Application Groups in Oracle IDCS

    Use the Admin console of Oracle IDCS to create Application Groups that match the application names from your Azure AD Domain Services tenant. Additionally, assign Read-Only access to these Application Groups to your Azure AD Groups where the corresponding users were added during step 1.
   
3. Link Azure AD DS to Oracle IDCS
    
    Click “Link” button from your Azure AD Domain Services instance. Choose “Oracle Identity Cloud Service” option and select the application group and the desired protocol for single sign-on. Save changes.
    
4. Test Single Sign-On Login
   After configuring the integration, test logging into Oracle Applications using your Azure AD Credentials. 
   If everything was correctly implemented, you will be directed to the Oracle Applications Dashboard without prompting for additional credentials.