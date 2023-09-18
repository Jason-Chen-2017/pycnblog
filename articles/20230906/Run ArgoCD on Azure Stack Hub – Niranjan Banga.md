
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Argo CD is an open-source GitOps continuous delivery tool for Kubernetes that automates application deployment, management, and monitoring through a declarative approach. It aims to provide a git-centric approach to delivering applications to Kuberentes clusters by leveraging tools like Helm or Kustomize in the backend.

This article will guide you how to install and run Argo CD on Microsoft Azure Stack Hub (ASH). We'll be using AKS Engine to create a cluster with ASH as its underlying platform. This will also help you understand the key differences between running Argo CD on AWS EKS and AZURE AKS/ASH. 

In this article, we will cover following steps:
1. Install Prerequisites
2. Create AKS Cluster on ASH
3. Install Argo CD on AKS
4. Access Argo CD Web UI
5. Test Deployment using YAML file

Before proceeding further let's understand few terminology used in ASH:

1. Infrastructure: Physical resources such as servers, switches, and storage devices required to deploy virtual machines and cloud services. 

2. Platform: Software infrastructure that provides access to compute, network, and persistent storage resources and enables the deployment of virtual machines and containers onto them. 

3. Cloud Operator: Person who manages the infrastructure of an organization and responsible for managing all aspects of their IT landscape including security, compliance, operations, governance, cost optimization, and service level agreements. 

4. API endpoint: URL address used to communicate with various endpoints exposed by different clouds platforms. These URLs typically end with.com/.org/.net etc depending on the specific provider being used. For example: https://management.azure.com/, https://api.digitalocean.com/, https://cloud.google.com/, https://us-west-1.iaas.cloud.ovh.net/.

5. Service Administrator Account: An account with administrative privileges over a subscription, allowing it to perform tasks such as creating new subscriptions, adding users, assigning roles, modifying permissions, etc.


Let’s start installing prerequisites. 

Prerequisite:

1. Microsoft Azure Subscription - To sign up for free trial subscription visit here: https://azure.microsoft.com/en-in/free/?WT.mc_id=A261C142F

2. Enable Azure Stack Hub preview feature from Azure portal by navigating to your subscription settings > Preview features > Add "Microsoft.ContainerService" Preview Feature

3. Azure CLI version should be greater than 2.0.79

   ```
   az --version 
   ```

4. Docker should be installed and running on your machine. You can download docker from here: https://www.docker.com/get-started

5. If you have not already created an SSH Key pair to connect to the Linux VM via SSH, follow these steps:
   
  ```
  ssh-keygen -t rsa -b 4096 -f ~/.ssh/<your_private_key> 
  chmod 600 ~/.ssh/<your_private_key>
  ```

  Copy the public key content and add it to your GitHub account so that you can clone repositories securely. 

6. Clone the repository into your local environment using the command below:
  
  ```
  git clone <repository_url>
  ```

7. Create a directory named **deployment** inside argocd folder. Navigate to the deployment directory and create two files named **argocd.yaml** and **kustomization.yaml**.

    **argocd.yaml**:
     
     The `argocd.yaml` file contains all the information about our Argo CD installation which includes type of runtime(runtimeClassName), resource requests and limits, nodeSelector details, tolerations details, affinity details, hostNetwork flag value. 
     
     ```
       apiVersion: v1
       kind: ConfigMap
       metadata:
         name: argocd-cm
         labels:
           app.kubernetes.io/name: argocd-server
           app.kubernetes.io/part-of: argocd
       data:
         server.secretkey: $(openssl rand -hex 16) # Generate a unique secret key to use for basic auth passwords
         accounts.minikube: |
           url: https://localhost:8443
           password: ""
           username: admin
           insecure: true # Not recommended! Don't use self signed certs with production instances. In this case, we are using a self generated cert for testing purposes only.
         
         rbac.defaultPolicy: role:readonly    # Set default policy as readonly mode
    ```
    
    **kustomization.yaml:** 
     
     The `kustomization.yaml` file is used to specify the common configuration across multiple kustomize configurations. Here, we configure the ingress for the Argo CD server deployment using NodePort type load balancer. 
     
     ```
       namespace: argocd    
       configMapGenerator:  
        - name: argocd-cm
          behavior: merge
          literals:
            - server.adminPassword=<PASSWORD>          # Change the default admin password 
            - server.ingress.enabled=true            # Configure ingress for Argo CD Server deployment
            - server.ingress.host=argocd.<ip>.nip.io
            - server.ingress.path=/                 # Specify the path to expose Argo CD console
            - server.ingress.tls.enabled=false       # Disable TLS for demo purpose
       images:
         - name: quay.io/argoproj/argocd
           newName: ghcr.io/nirankananth11/argocd 
           digest: sha256:<image_digest>      # Replace the image digest value based on the registry where the image is stored
       
       patchesStrategicMerge:
         - patch.yaml
       
       ingress:
         - hosts:
             - argocd.<ip>.nip.io
           paths:
             - /
           serviceName: argocd-server
           servicePort: 80        
    ```

     Here, 
     * `<ip>` refers to the IP address assigned to the ASH instance. 
     * `namespace` defines the Kubernetes namespace to install Argo CD. 
     * `configMapGenerator` generates a ConfigMap object containing all the necessary configuration parameters to set up our Argo CD installation. 
     * `images` renames the container image used by Argo CD to refer to our custom private registry instead of the official one. 
     * `patchesStrategicMerge` specifies any additional modifications needed to customize the base manifest. 
     * `ingress` creates an Ingress object to allow external traffic to reach the Argo CD console. 
     
After cloning the repository and creating the appropriate files, navigate back to the root folder of the repository and run the commands mentioned below to complete the pre-requisites. 

  ```
  kubectl apply -k./deployment        # Deploy Argo CD using kustomize
  sleep 60                          # Wait for some time before accessing Argo CD Console
  kubectl port-forward svc/argocd-server -n argocd 8080:443 &    # Forward HTTPS port locally for easy access
  ```
  
  Once the above commands execute successfully, you can see that Argo CD has been deployed successfully. 

Next step is to test the deployment of an app using a sample YAML file. 