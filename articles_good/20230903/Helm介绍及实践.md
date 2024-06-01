
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Helm是一个开源的Kubernetes包管理器，可以帮助管理Kubernetes应用。Helm提供了一个功能更全面的包管理工具，包括创建、安装、升级、删除、分享和检索chart。 Helm还可以通过chart和values文件轻松实现配置管理。

Helm在发布新版本的软件时，通过版本控制系统（如GitHub）记录每个版本中的变更，这样就可以追踪软件每次的改动，方便软件的回滚。Helm的chart采用Chart.yaml文件进行定义，该文件记录了chart的名称、版本号、描述信息等。

本文介绍Helm的主要特性和优点，并通过一个示例应用案例介绍Helm的实际用法。

# 2.基本概念术语说明
## Chart
Helm管理的软件包称为Chart，它是一个目录，里面包含了运行容器所需的所有资源定义文件，例如Deployment，Service，Ingress等。

Helm的Chart组织结构如下图所示：


## Release
Chart打包后的产物称之为Release，即Chart运行实例。每个Release都有一个唯一的名字，例如myapp-v1，可以用来管理一个特定的部署。当创建一个新的Release时，Helm会将Chart对应的所有资源模板渲染成Kubernetes资源对象，然后创建这些资源。

## Repository
Repository是Helm用来存放和共享Chart的地方。一个Helm Repo可以是多个Charts仓库集合，或者单个Chart仓库。Helm客户端可以使用命令或浏览器访问不同的Helm Repositories，从而找到需要的软件包。

## Values
Values文件用于指定Release的配置参数。Values文件是一个YAML文件，定义了要传递给Chart的变量值。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 安装 Helm
在安装Helm之前，确保你已经安装好了Kubernetes环境。

1. 配置Helm仓库源地址，执行以下命令：

   ```bash
   helm repo add stable https://kubernetes-charts.storage.googleapis.com/
   ```

2. 使用 Helm install 命令安装 Helm 客户端。

   ```bash
   curl https://raw.githubusercontent.com/helm/helm/master/scripts/get | bash
   ```

3. 检查 Helm 的版本信息。

   ```bash
   helm version
   ```

4. 在 Helm 中添加 chart repository。

   ```bash
   helm repo add [repository-name] [repository-url]
   ```

5. 查看已有的 chart repository。

   ```bash
   helm repo list
   ```

6. 更新 chart repository 列表。

   ```bash
   helm repo update
   ```

7. 从 chart repository 中搜索可用的 chart。

   ```bash
   helm search [keyword]
   ```

8. 下载 chart 。

   ```bash
   helm pull [chart-repo]/[chart-name] --version=[version]
   ```

9. 查看本地 chart 仓库中所有的 chart 。

   ```bash
   helm ls
   ```

## 创建你的第一个 Helm Chart
为了创建一个简单的 Kubernetes Deployment 和 Service，我们可以按照以下步骤：

1. 初始化一个 chart 项目。

   ```bash
   helm create myapp
   ```

2. 修改 myapp/templates 中的 YAML 文件，添加 Deployment 和 Service 对象。

   ```yaml
   apiVersion: apps/v1beta2 # for versions before 1.9.0 use apps/v1beta1
   kind: Deployment
   metadata:
     name: {{.Release.Name }}-{{.Chart.Name }}
     labels:
       app.kubernetes.io/name: {{ include "myapp.fullname". }}
       helm.sh/chart: {{.Chart.Name }}-{{.Chart.Version }}
       app.kubernetes.io/instance: {{.Release.Name }}
       app.kubernetes.io/managed-by: {{.Release.Service }}
   spec:
     replicas: {{.Values.replicaCount }}
     selector:
       matchLabels:
         app.kubernetes.io/name: {{ include "myapp.fullname". }}
         app.kubernetes.io/instance: {{.Release.Name }}
     template:
       metadata:
         labels:
           app.kubernetes.io/name: {{ include "myapp.fullname". }}
           app.kubernetes.io/instance: {{.Release.Name }}
       spec:
         containers:
         - name: {{.Chart.Name }}
           image: "{{.Values.image.repository }}:{{.Values.image.tag }}"
           ports:
           - containerPort: {{.Values.service.internalPort }}
             protocol: TCP
           envFrom:
           - configMapRef:
               name: {{ include "myapp.fullname". }}-configmap
    ---
   apiVersion: v1
   kind: Service
   metadata:
     name: {{.Release.Name }}-{{.Chart.Name }}
     labels:
       app.kubernetes.io/name: {{ include "myapp.fullname". }}
       helm.sh/chart: {{.Chart.Name }}-{{.Chart.Version }}
       app.kubernetes.io/instance: {{.Release.Name }}
       app.kubernetes.io/managed-by: {{.Release.Service }}
   spec:
     type: {{.Values.service.type }}
     ports:
     - port: {{.Values.service.externalPort }}
       targetPort: {{.Values.service.internalPort }}
       protocol: TCP
     selector:
       app.kubernetes.io/name: {{ include "myapp.fullname". }}
       app.kubernetes.io/instance: {{.Release.Name }}
   ```

3. 在 values.yaml 中增加自定义的参数。

   ```yaml
   replicaCount: 1
   image:
     repository: nginx
     tag: stable
   service:
     type: ClusterIP
     externalPort: 80
     internalPort: 80
   ```

4. 生成并检查 chart。

   ```bash
   helm lint myapp
   ```

5. 安装 chart。

   ```bash
   helm install myrelease myapp
   ```

6. 查看 release status。

   ```bash
   helm status myrelease
   ```

7. 浏览 chart 服务。

   ```bash
   minikube service myrelease-myapp
   ```

8. 通过 `helm delete` 命令删除 release。

   ```bash
   helm delete myrelease
   ```

## 为你的 Helm Chart 添加更多特性
上面只是创建一个最基本的 Kubernetes Deployment 和 Service。Helm Chart 提供了很多其它特性，你可以根据自己的需求选择性地添加它们。

### Adding a Config Map
Config Maps 是 Kubernetes 的内置对象，允许将配置数据保存到键值对形式的存储中，而无需定义独立的文件。你可以把 Config Map 当作 Helm Chart 中的 values 文件一样使用，通过 Environment Variables 或 Volumes 将其注入到 Pod 中。

1. 在 templates 文件夹下新增 Config Map 模板文件 `configmap.yaml`。

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: {{ include "myapp.fullname". }}-configmap
     labels:
       app.kubernetes.io/name: {{ include "myapp.fullname". }}
       helm.sh/chart: {{.Chart.Name }}-{{.Chart.Version }}
       app.kubernetes.io/instance: {{.Release.Name }}
       app.kubernetes.io/managed-by: {{.Release.Service }}
   data:
     MY_ENV_VAR: somevalue
     OTHER_ENV_VAR: othervalue
   ```

2. 修改 Deployment 模板文件 `deployment.yaml`，添加 Config Map 的引用。

   ```yaml
  ...
   envFrom:
   - configMapRef:
       name: {{ include "myapp.fullname". }}-configmap
   volumes:
   - name: config-volume
     configMap:
        name: {{ include "myapp.fullname". }}-configmap
   volumeMounts:
   - name: config-volume
     mountPath: /etc/config
   ```

3. 在 values.yaml 中增加一个名为 `extraVolumes` 的参数，用于声明额外的卷。

   ```yaml
   extraVolumes:
   - name: config-volume
     emptyDir: {}
   ```

4. 在 values.yaml 中增加一个名为 `extraVolumeMounts` 的参数，用于声明额外的卷挂载。

   ```yaml
   extraVolumeMounts:
   - name: config-volume
     readOnly: true
     mountPath: "/etc/config"
   ```

5. 在 Deployment 中声明一个环境变量，读取 Config Map 中的值。

   ```yaml
   env:
     - name: APP_CONFIG
       valueFrom:
         configMapKeyRef:
            name: {{ include "myapp.fullname". }}-configmap
            key: MY_ENV_VAR
   ```

### Adding Secrets
Secrets 也是 Kubernetes 的内置对象，用于保存敏感的数据，例如密码、密钥等。你可以把 Secret 当作配置文件一样使用，但是注意不要把敏感数据写入代码中，而应当在部署前通过 Helm 或其他方式管理这些数据。

1. 在 templates 文件夹下新增 Secret 模板文件 `secret.yaml`。

   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: {{ include "myapp.fullname". }}-secret
     annotations:
       "helm.sh/hook": pre-install,pre-upgrade
       "helm.sh/hook-delete-policy": hook-succeeded,before-hook-creation
   stringData:
      USERNAME: admin
      PASSWORD: password123
   ```

2. 修改 Deployment 模板文件 `deployment.yaml`，添加 Secret 的引用。

   ```yaml
  ...
   env:
   - name: DB_PASSWORD
     valueFrom:
       secretKeyRef:
          name: {{ include "myapp.fullname". }}-secret
          key: PASSWORD
  ...
   secrets:
   - name: {{ include "myapp.fullname". }}-secret
     secretName: {{ include "myapp.fullname". }}-secret
   ```

3. 在 values.yaml 中删除 `USERNAME` 和 `PASSWORD` 参数。

### Using Init Containers to Run Pre-Start Scripts
Init Containers 可以用来执行一些预先定义的脚本任务，比如设置数据目录权限、导入数据等。你可以在 Deployment 中定义 Init Container，并使用 `initContainers` 来指定顺序执行。

1. 在 templates 文件夹下新增 Init Container 模板文件 `initcontainer.yaml`。

   ```yaml
   apiVersion: v1
   kind: Pod
   metadata:
     name: init-myservice
   spec:
     restartPolicy: Never
     containers:
     - name: myservice
       image: busybox
       command: ["/bin/sleep", "infinity"]
     - name: init-myservice
       image: busybox
       command: ["chown", "-R", "nobody:nogroup", "/data"]
       volumeMounts:
       - name: data-volume
         mountPath: /data
     volumes:
     - name: data-volume
       hostPath: 
         path: /var/lib/myservice
   ```

2. 修改 Deployment 模板文件 `deployment.yaml`，添加 Init Container 的定义。

   ```yaml
  ...
   initContainers:
   - name: initialize-database
     image: busybox
     command: ['sh', '-c', 'until nslookup db; do echo waiting for database; sleep 2; done;']
  ...
   containers:
   - name: myservice
     image: nginx:latest
     env:
     - name: MYSQL_URL
       value: jdbc:mysql://db/myapp
     ports:
     - containerPort: 80
     volumeMounts:
     - name: data-volume
       mountPath: /usr/share/nginx/html
     livenessProbe:
       httpGet:
         path: /healthcheck
         port: 80
         scheme: HTTP
       initialDelaySeconds: 5
       timeoutSeconds: 1
     readinessProbe:
       httpGet:
         path: /readiness
         port: 80
         scheme: HTTP
       initialDelaySeconds: 5
       periodSeconds: 10
       timeoutSeconds: 1
 ...
  volumes:
   - name: data-volume
     emptyDir: {}
  ```

3. 使用 `lifecycle` 来定义如何处理 Init Container 的退出状态码。

   ```yaml
  ...
   lifecycle:
     postStart:
       exec:
         command: ["/bin/sh","-c","echo 'Database is ready'"]
     preStop:
       exec:
         command: ["/bin/sh","-c","echo 'Shutting down...'; sleep 10"]
  ...
   ```


### Using Custom Resource Definitions (CRDs)
Helm 提供了一个机制，使得用户可以方便地创建自定义资源定义。这些 CRD 可以在模板文件中定义，也可以作为依赖项被包含进来。

1. 编写模板文件 `crds.yaml`。

   ```yaml
   apiVersion: apiextensions.k8s.io/v1beta1
   kind: CustomResourceDefinition
   metadata:
     name: mycrds.example.com
   spec:
     group: example.com
     names:
       kind: MyCRD
       plural: mycrds
     scope: Namespaced
     version: v1alpha1
     validation:
       openAPIV3Schema:
         properties:
           spec:
             type: object
             properties:
               field:
                 type: integer
     additionalPrinterColumns:
       - name: Age
         type: date
         description: The age of the resource
         JSONPath: ".metadata.creationTimestamp"
   ```

2. 使用 `crd-install` 渲染 CRD 模板。

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: crds-rendered
     namespace: default
   data:
     crd-mycrds.yaml: |-
       {{.Files.Get "crds.yaml" | indent 4 }}
   ---
   apiVersion: v1
   kind: ServiceAccount
   metadata:
     name: mycrds-sa
     namespace: default
   ---
   apiVersion: rbac.authorization.k8s.io/v1
   kind: ClusterRoleBinding
   metadata:
     name: mycrds-rb
   roleRef:
     apiGroup: rbac.authorization.k8s.io
     kind: ClusterRole
     name: cluster-admin
   subjects:
   - kind: ServiceAccount
     name: mycrds-sa
     namespace: default
   ---
   apiVersion: batch/v1
   kind: Job
   metadata:
     name: render-crds
     namespace: default
   spec:
     backoffLimit: 0
     template:
       metadata:
         name: render-crds
         namespace: default
       spec:
         serviceAccountName: mycrds-sa
         restartPolicy: OnFailure
         containers:
         - name: kubectl
           image: bitnami/kubectl
           command:
             - sh
             - -c
             - "kubectl apply -f /tmp/{{.Release.Name }}-crds/*"
             - cp
             - "/tmp/{{.Release.Name }}-crds/*"
             - "/tmp/"
           volumeMounts:
             - name: tmp
               mountPath: "/tmp"
         volumes:
           - name: tmp
             emptyDir: {}
     backoffLimit: 0
   ```

3. 在 Chart.yaml 中声明 CRD 依赖。

   ```yaml
   dependencies:
   - name: custom-resource-definitions
     version: ">0.1.0"
     alias: crds
     condition: crds.enabled
     import-values:
     - schema: disable
   ```

# 4.具体代码实例和解释说明
我们用 Helm 来发布一个 Kubernetes 搜索引擎 Elasticsearch。首先，我们创建一个新的 Chart。

```bash
$ mkdir elasticsearch
$ cd elasticsearch/
$ helm create elasticsearch
Creating elasticsearch
```

Chart 目录结构如下：

```bash
├── Chart.yaml           # 包描述文件
├── charts               # 依赖的 chart
│   └── elastic-stack    # elaticsearch chart
└── templates            # kubernetes 对象定义模板
    ├── NOTES.txt         # 安装后提示信息
    ├── _helpers.tpl      # tpl函数库
    ├── elastic-cluster   # Deployment and Service objects for the ElasticSearch clusters
    │   ├── deployment.yaml
    │   └── service.yaml
    └── elastic-master     # Deployment and Service objects for the ElasticSearch master node
        ├── deployment.yaml
        └── service.yaml
```

然后编辑 Chart.yaml 文件，修改 Chart 的描述信息。

```yaml
apiVersion: v2
name: elasticsearch
description: A Helm chart for deploying an Elasticsearch cluster on Kubernetes
type: application
version: 0.1.0
appVersion: "7.9.3"
dependencies:
- name: elastic-stack
  version: 1.8.0
  repository: https://helm.elastic.co
  condition: elastic-stack.enabled
```

我们这里使用了 elastic-stack 作为依赖，因为它提供了比较完备的 Elasticsearch 集群方案，包括 Elasticsearch 本身和 Kibana 可视化插件。

接着，编辑 elastic-stack/values.yaml ，修改默认配置，并设置 elastic-master 为只读模式，以提高集群可用性。

```yaml
antiAffinity: hard
replicas: 3
minimumMasterNodes: 2
persistence:
  enabled: false
esJavaOpts: "-Xmx1g -Xms1g"
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1"
rolesMaster: "master"
rolesData: ""
rolesIngest: ""
roles ml: ""
kibanaEnabled: true
ingress:
  enabled: false
  hosts: []
probes:
  readiness:
    failureThreshold: 3
    periodSeconds: 10
    successThreshold: 1
    timeoutSeconds: 5
  liveness:
    failureThreshold: 3
    periodSeconds: 10
    successThreshold: 1
    timeoutSeconds: 5
sysctlImage: "busybox"
fullnameOverride: ""
rbac:
  create: true
  superuserUsername: elastic
  superuserPassword: changeme
securityContext:
  enabled: true
  sysctls:
  - name: vm.max_map_count
    value: "262144"
terminationGracePeriodSeconds: 30
nodeSelector: {}
tolerations: []
affinity: {}
readiness:
  enabled: true
  periodSeconds: 10
  initialDelaySeconds: 30
liveness:
  enabled: true
  periodSeconds: 10
  initialDelaySeconds: 30
initContainers:
- name: fix-permissions
  image: busybox
  securityContext:
    privileged: true
  command:
  - chmod
  args:
  - -R
  - "u=rwX,go=rX"
  - /usr/share/elasticsearch/data
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi
```

最后，在 templates/elastic-cluster/deployment.yaml 中定义 Deployment 和 Service 对象，来创建一个 Elasticsearch 集群。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: "{{ include "elasticsearch.fullname" $ }}-es-default"
  labels:
    app: elasticsearch
spec:
  replicas: {{ $.Values.replicas }}
  strategy:
    type: RollingUpdate
  selector:
    matchLabels:
      component: es-default
  template:
    metadata:
      labels:
        app: elasticsearch
        component: es-default
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                  - {key: component, operator: In, values: [es-default]}
              topologyKey: kubernetes.io/hostname
      priorityClassName: high-priority
      terminationGracePeriodSeconds: 30
      hostname: elasticsearch-default
      initContainers:
      - name: fix-permissions
        image: "{{ $.Values.sysctlImage }}"
        imagePullPolicy: IfNotPresent
        securityContext:
          runAsUser: 0
          allowPrivilegeEscalation: true
        command:
        - chmod
        - -R
        - u=rwX,g=rX
        - /usr/share/elasticsearch/data
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi

      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:{{ $.Values.imageTag }}
        imagePullPolicy: IfNotPresent
        env:
        - name: ELASTICSEARCH_JAVA_OPTS
          value: "{{- with.Values.esJavaOpts }} {{. | quote }} {{ end }}"

        ports:
        - name: http
          containerPort: 9200
          protocol: TCP
        - name: transport
          containerPort: 9300
          protocol: TCP

        volumeMounts:
        - name: data
          mountPath: /usr/share/elasticsearch/data

        resources:
          requests:
            cpu: {{ $.Values.resources.requests.cpu }}
            memory: {{ $.Values.resources.requests.memory }}
          limits:
            cpu: {{ $.Values.resources.limits.cpu }}
            memory: {{ $.Values.resources.limits.memory }}

      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: data-vol-claim

---
apiVersion: v1
kind: Service
metadata:
  name: "{{ include "elasticsearch.fullname" $ }}-es-default"
  labels:
    app: elasticsearch
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 9200
    targetPort: 9200
    protocol: TCP
  - name: tcp-transport
    port: 9300
    targetPort: 9300
    protocol: TCP
  selector:
    component: es-default
```

至此，我们的 Elasticsearch 集群就完成了安装。

# 5.未来发展趋势与挑战
随着云原生技术的不断发展，Helm 也在跟随潮流，逐渐成为更加火热的容器编排工具。它的优势在于简单易用，能够快速发布复杂的 Kubernetes 应用程序；而且拥有良好的扩展能力，支持多种云平台。不过 Helm 有待进一步完善和发展。

Helm 的最大局限在于 Helm Chart 模板语言过于简单，无法满足复杂场景下的自动化需求。Kubernetes 社区正在探索其他方式来解决这个问题，比如编写真正的 Operators，以提供更强大的自动化能力。

# 6.附录常见问题与解答
* Q: Helm 是什么？
* A: Helm 是 Kubernetes 的包管理器。它可以帮助管理 Kubernetes 应用，基于 Chart 开发模式提供更高层次的抽象，并且具有众多优秀特性，例如版本控制、依赖管理、灰度发布等。Helm 支持跨平台安装，可以轻松集成 CI/CD 流程，提供开箱即用的最佳实践。

* Q: 为什么要使用 Helm？
* A: 在传统上，云计算平台通常为用户提供不同的服务，如数据库、消息队列、负载均衡等。这些服务一般由不同的团队维护、运维和升级。

如果有多个应用同时依赖相同的服务，那么管理这些服务就会变得非常困难。而 Helm 提供的解决方案就是将不同服务打包成不同的 Helm Chart，并统一管理。它可以降低管理复杂性、提升效率，并提供便捷的升级方案。

* Q: Helm 的安装过程有哪些步骤？
* A: Helm 的安装分为两步：第一步是安装 Helm CLI （Command Line Interface），第二步是初始化 Helm 并连接到 Kubernetes 集群。

第一步：
Helm CLI 安装非常简单，直接通过命令行工具即可完成。Linux 系统推荐通过 snap 或 apt-get 安装，Mac OS X 系统推荐通过 Homebrew 安装。

Windows 用户可以访问 Helm 的 GitHub 主页获取安装包，下载并解压后将其加入 PATH 环境变量。

第二步：初始化 Helm。

```bash
$ helm version
Client: &version.Version{SemVer:"v2.16.1", GitCommit:"bbdfe5e7803a12bbdf97e94cd847859890cf4050", GitTreeState:"clean"}
Server: &version.Version{SemVer:"v2.16.1", GitCommit:"bbdfe5e7803a12bbdf97e94cd847859890cf4050", GitTreeState:"clean"}
```

使用 `helm init` 命令初始化 Helm，它会生成必要的加密密钥和配置文件，并安装 Tiller 组件，该组件是 Helm Server 的代理。

```bash
$ helm init
$HELM_HOME has been configured at /Users/<username>/.helm.

Tiller (the Helm server-side component) has been installed into your Kubernetes Cluster.

Please note: by default, Tiller is deployed as a cluster-wide service. Access to it may be limited by firewall rules or other infrastructure settings. For more information, refer to the documentation: https://docs.helm.sh/using_helm/#securing-your-helm-installation
```

注意：如果你想使用 RBAC 授权模式，请执行 `helm init --service-account tiller --override spec.selector.matchLabels.'name'='tiller',spec.selector.matchLabels.'app'='helm' --output yaml | sed's@apiVersion: extensions/v1beta1@apiVersion: apps/v1@' | kubectl apply -f -` 命令来替换生成的 manifest 文件中的 apiVersion。

* Q: Helm Chart 应该怎么写？
* A: Helm Chart 是 Helm 包管理的基础。一个 Helm Chart 是一个目录，其中包含运行容器所需的所有资源定义文件，例如 Deployment，Service，Ingress 等。Chart 中的资源定义文件经过 Helm 解释器处理之后，生成 Kubernetes 对象，然后提交给 Kubernetes API 服务器进行创建。

Chart 目录结构如下：

```
<chart-directory>/
|-- Chart.yaml        # 描述了 Chart 的基本信息
|-- README.md         # 一个详细的使用文档
|-- values.yaml       # 默认配置参数，会被安装时的命令行参数覆盖
|-- requirements.yaml # 指定依赖的 Chart
|-- templates/        # k8s 对象定义模板
|   |-- deployment.yaml
|   `-- ingress.yaml
└── charts/           # 子 Chart，可以是依赖的外部 Chart 或同一个项目的子 Chart
```

Chart.yaml 文件中有以下字段：

```
apiVersion: APIVersion 是 Helm Chart 的版本，目前必须是 v1。
name: Chart 的名称。
description: Chart 的描述。
type: 此 Chart 的类型，默认为 “application” 。
version: Chart 的版本号。
appVersion: 此 Chart 的应用版本号。
keywords: 一组关键字，用于搜索。
home: Chart 的官网地址。
sources: Chart 的源码链接。
maintainers: 一组维护者的联系方式。
engine: 要求的 Helm 引擎版本。
icon: Chart 显示的图标 URL。
dependencies: Chart 的依赖关系，可以指定依赖 Chart 名称和版本范围，但只能指定顶级父 Chart，不能指定子 Chart。
```

values.yaml 文件中有以下字段：

```
# Default values for <chart-name>.
# This is a YAML-formatted file.
# Declare variables to be passed into your templates.

replicaCount: 1

image:
  repository: nginx
  tag: latest

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

podAnnotations: {}
podSecurityContext: {}
securityContext: {}
service:
  type: ClusterIP
  port: 80
ingress:
  enabled: false
  annotations: {}
  paths: []
  hosts:
    - chart-example.local
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 100
  targetCPUUtilizationPercentage: 50
  targetMemoryUtilizationPercentage: 50

nodeSelector: {}

tolerations: []

affinity: {}
```

模板文件中可以包含变量表达式，用以引用 values.yaml 中的参数。Helm 会解析并执行模板文件，生成最终的 Kubernetes 对象。

```
{{ include "<template_name>". }}
```