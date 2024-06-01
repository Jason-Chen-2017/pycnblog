
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 ## 1.1 Helm 是什么？
          Helm 是 Kubernetes 的包管理器工具。Helm 可以帮助用户管理复杂的 Kubernetes 应用，通过 Charts 来打包、安装和升级 Kubernetes 中的应用程序。
          ### 1.1.1 Helm 安装
          ### 1.1.2 Helm 操作命令
          Helm 提供了多个子命令用于管理charts，包括 `install`、`search`、`pull`、`template`、`delete`、`upgrade`等。可以通过 `helm <subcommand> --help` 查看具体的操作命令。
          # 2.Chart
          ## 2.1 Chart 是什么？
          Chart 是 Helm 在 Kubernetes 中应用的包文件。它包含了一组描述 Kubernetes 资源定义的 YAML 文件以及 Kubernetes 模板文件。当我们用 Helm 命令行工具进行 Chart 的安装时，Helm 会根据 Chart 中的描述信息部署出一组相关联的 Kubernetes 对象。
          ## 2.2 Chart 创建
          当我们希望编写自己的 Chart 时，需要按照一定的目录结构创建我们的 Chart 文件。以下是最简单的 Chart 目录结构：
          ```shell
          ├── Chart.yaml    # 用于定义 Chart 的一些元数据（名称、版本、图表作者、许可证）
          ├── values.yaml   # 用于设置 Chart 默认参数值
          ├── templates     # 用于定义模板文件
          │   └── deployment.yaml  # 示例模板文件
          └── README.md     # 对 Chart 的简单介绍
          ```
          ### 2.2.1 Chart.yaml
          Chart.yaml 是用于定义 Chart 的一些元数据的描述文件。它主要包含以下信息：
          * name: chart 名称
          * version: chart 版本号
          * appVersion: chart 中定义的 Kubernetes 资源的版本号
          * description: chart 的详细描述信息
          * keywords: chart 的关键字列表
          * home: chart 项目主页
          * sources: 发布 chart 源码的 git 仓库地址
          * maintainers: chart 作者相关信息列表
          ```yaml
          apiVersion: v1
          name: mychart
          version: 0.1.0
          appVersion: "1.0"
          description: A Helm chart for Kubernetes
          keywords:
            - helm
            - kubernetes
            - template
          home: https://example.com
          sources:
            - https://github.com/example/mychart
          maintainers:
            - name: example
              email: <EMAIL>
          ```
          ### 2.2.2 values.yaml
          values.yaml 文件用于设置 Chart 的默认参数值。它可以让我们在 Chart 安装或更新时传入自定义的参数。比如我们可以在 values.yaml 中指定 Docker 镜像的 tag 或数据库密码等参数。通常情况下，我们会在 Chart 中增加一个 `_helpers.tpl` 文件，用来定义模板语言中使用的辅助函数，这些辅助函数可以让 Chart 更容易被复用和扩展。
          ```yaml
          image:
            repository: nginx
            pullPolicy: IfNotPresent
            tag: latest
          replicaCount: 1

          db:
            host: localhost
            port: 5432
            user: postgres
            password: ""
            database: myapp
          ```
          ### 2.2.3 Chart 模板文件
          Chart 中的模板文件用来定义 Kubernetes 资源对象定义。每种类型的资源都对应了一个模板文件。Helm 提供了一个模板语言，允许我们将模板渲染成有效的 Kubernetes 资源清单。以下是一个例子：
          ```yaml
          apiVersion: apps/v1beta1
          kind: Deployment
          metadata:
            name: {{.Release.Name }}-{{.Chart.Name }}
            labels:
              app: {{.Chart.Name }}
              release: {{.Release.Name }}
          spec:
            replicas: {{.Values.replicaCount }}
            selector:
              matchLabels:
                app: {{.Chart.Name }}
                release: {{.Release.Name }}
            template:
              metadata:
                labels:
                  app: {{.Chart.Name }}
                  release: {{.Release.Name }}
              spec:
                containers:
                - name: {{.Chart.Name }}
                  image: "{{.Values.image.repository }}:{{ default.Chart.AppVersion.Values.image.tag }}"
                  ports:
                    - containerPort: 80
                      protocol: TCP
                  readinessProbe:
                    httpGet:
                      path: /
                      port: http
                  livenessProbe:
                    httpGet:
                      path: /healthz
                      port: http
                  resources: {}
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: {{.Release.Name }}-{{.Chart.Name }}
          labels:
            app: {{.Chart.Name }}
            release: {{.Release.Name }}
        spec:
          type: ClusterIP
          ports:
          - name: http
            port: 80
            targetPort: http
          selector:
            app: {{.Chart.Name }}
            release: {{.Release.Name }}
          sessionAffinity: None
        ```
          上面就是一个 Deployment 和一个 Service 的组合，其中 Deployment 表示 Kubernetes 中的 Deployment 资源类型，Service 表示 Kubernetes 中的 Service 资源类型。这里使用到了 Helm 提供的 `.Release.Name`，`.Chart.Name`，`.Values` 等变量，它们分别代表了 Release 的名字，Chart 名，当前 Chart 的配置参数等。
          通过模板文件，我们可以轻松地生成任意数量的 Kubernetes 资源。
          ### 2.2.4 Chart 打包
          一旦完成 Chart 的编写，就可以把它打包成一个压缩包，上传到 Chart 仓库中供其他人下载安装。Chart 仓库通常托管在 HTTP(S) URL 上，并遵循一般的软件包管理规范。可以使用 `helm package` 命令对 Chart 进行打包。
          ```shell
          $ helm package./mychart
          Successfully packaged chart and saved it to: /home/ubuntu/mychart-0.1.0.tgz
          ```
          ### 2.2.5 Chart 安装
          如果已经上传了 Chart 包到 Chart 仓库，就可以使用 `helm install` 命令安装该 Chart。
          ```shell
          $ helm install stable/prometheus --name my-release
          NAME: my-release
          LAST DEPLOYED: Mon Oct  9 17:05:24 2019
          NAMESPACE: default
          STATUS: deployed
          REVISION: 1
          TEST SUITE: None
          NOTES:
          1. Get the application URL by running these commands:
            export POD_NAME=$(kubectl get pods --namespace default -l "app=prometheus,component=server" -o jsonpath="{.items[0].metadata.name}")
            echo "Visit http://127.0.0.1:8080 to use your application"
            kubectl port-forward $POD_NAME 8080:80
          ```
          ### 2.2.6 Chart 更新
          当 Chart 有新版本可用时，我们可以使用 `helm upgrade` 命令进行更新。
          ```shell
          $ helm repo update
          Hang tight while we grab the latest from your chart repositories...
         ...Skip local chart repository
         ...Successfully got an update from the "stable" chart repository
          Update Complete. ⎈ Happy Helming!⎈ 
          $ helm search prometheus
          NAME              CHART VERSION APP VERSION DESCRIPTION
          stable/prometheus 9.3.1          8.x        Prometheus monitoring system service
          
          To update one of these charts, run the following command:
            helm upgrade [RELEASE] [CHART]
          ```
          # 3.常用参数介绍
          ## 3.1 指定 Chart 版本
          如果要安装或升级一个 Chart，但是 Chart 的版本没有指定，Helm 会默认拉取最新的版本。如果需要指定特定的版本，可以使用 `--version` 参数。
          ```shell
          $ helm install stable/mysql --version 0.1.0
          ```
          ## 3.2 指定命名空间
          如果要安装或升级一个 Chart，但不想让它安装到默认的命名空间中，可以使用 `--namespace` 参数指定其目标命名空间。
          ```shell
          $ helm install --namespace kube-system stable/mysql
          ```
          ## 3.3 安装 Chart 的前提条件
          如果 Chart 需要依赖某些 Kubernetes 功能，并且相应的资源尚不存在，则 Helm 将阻止安装。我们可以使用 `--pre` 参数跳过检查。
          ```shell
          $ helm install stable/prometheus --set server.persistentVolume.enabled=true --set alertmanager.persistentVolume.enabled=false --set server.ingress.enabled=true --pre
          Error: rendered manifests contain a resource that already exists. Unable to continue with install
          Please check the namespace or label selectors. Use --force to replace conflicting resources
          ```
          使用 `--force` 参数可以强制替换存在冲突的资源。
          ```shell
          $ helm install stable/prometheus --set server.persistentVolume.enabled=true --set alertmanager.persistentVolume.enabled=false --set server.ingress.enabled=true --force
          ```
          ## 3.4 设置参数
          每个 Chart 可能都有自己的可选参数，可以通过 `--set` 参数来指定它们的值。
          ```shell
          $ helm install stable/prometheus --set server.persistentVolume.enabled=true
          ```
          ## 3.5 使用 values.yaml 文件指定参数
          除了直接指定参数值外，我们也可以在 Chart 中提供一个 values.yaml 文件，然后通过 `-f` 参数来指定配置文件路径。
          ```shell
          $ helm install -f myvalues.yaml stable/prometheus
          ```
          ## 3.6 卸载 Chart
          删除一个 Chart 时，可以使用 `helm delete` 命令。
          ```shell
          $ helm delete my-release
          ```
          # 4.Helmfile
          ## 4.1 Helmfile 是什么？
          Helmfile 是一个声明式的 Kubernetes 应用管理器。它利用 Helm 的 Chart 和值来管理集群中的多个 Helm Releases。Helmfile 主要由两部分组成：配置模板和 Helm 操作集合。
          配置模板是 Helmfile 用 YAML 格式描述的 Helm releases 列表。每个配置模板可以包含多个 Helm releases，每个 Helm release 可指定 Chart 和配置参数。Helmfile 根据模板生成最终的 Kubernetes 资源清单，并通过 Helm 操作集合（例如 apply、diff、rollback、test）应用到集群上。
          ## 4.2 Helmfile 安装
          ## 4.3 Helmfile 配置模板
          下面的例子展示了一个简单的 Helmfile 配置模板，其中声明了一个名为 test 的 release，它使用 bitnami/nginx 作为 Chart，并且配置了两个 ingress。
          ```yaml
          helmDefaults:
            tillerless: true
            verify: false
          repositories:
            - name: stable
              url: https://kubernetes-charts.storage.googleapis.com/
          releases:
            - name: test
              chart: bitnami/nginx
              set:
                service.type: NodePort
                controller.hostPort.enabled: true
              ingress:
                enabled: true
                annotations:
                  kubernetes.io/ingress.class: nginx
                  certmanager.k8s.io/cluster-issuer: letsencrypt-prod
                paths:
                  - "/"
                hosts:
                  - test.example.com
                tls:
                  - secretName: test-tls
                    hosts:
                      - test.example.com
  ```
          从上面的配置模板可以看到，Helmfile 支持众多配置项。其中，`tillerless` 配置表示是否启用 Helm Tiller，如果设置为 true，那么所有的 Helm 操作都会被提交给 Helm 驱动 Kubernetes API（即非 Tiller）。
          `verify` 配置表示是否校验 chart 完整性。由于 Helmfile 默认拉取 chart 后本地验证，故此选项默认关闭。
          `repositories` 数组用于声明 Chart 仓库。
          `releases` 数组用于声明具体的 Helm release。其中 `name` 表示 release 名称，`chart` 表示 Chart 名称，`set` 数组用于指定配置参数，`ingress` 对象用于声明 Ingress 配置。
          ## 4.4 Helmfile 执行操作
          使用 Helmfile 可以方便地执行各种 Helm 操作。
          ### 4.4.1 检查差异
          执行 `helmfile diff` 命令，可以查看 Helmfile 计算出的待执行的 Helm 操作。
          ```shell
          $ helmfile diff
          Comparing release=test, chart=/Users/supereagle/code/bitnami/charts/bitnami/nginx
          *** Dry Run ***
          Release was not present in Helm.  Diff will show entire contents as new.
          @@ -0,0 +1,14 @@
          +# Source: nginx/templates/controller-serviceaccount.yaml
          +apiVersion: v1
          +kind: ServiceAccount
          +metadata:
          +  name: RELEASE-NAME-nginx
          +  labels:
          +    app.kubernetes.io/name: nginx
          +    helm.sh/chart: nginx-7.3.12
          +    app.kubernetes.io/instance: RELEASE-NAME
          +    app.kubernetes.io/version: "1.20.0"
          +    app.kubernetes.io/managed-by: Helm
          +spec:
          +  automountServiceAccountToken: false
          @@ -1,13 +15,6 @@
          {
            "kind": "Service",
            "apiVersion": "v1",
            "metadata": {
          -    "name": "RELEASE-NAME-nginx",
          -    "labels": {
          -      "app.kubernetes.io/name": "nginx",
          -      "helm.sh/chart": "nginx-7.3.12",
          -      "app.kubernetes.io/instance": "RELEASE-NAME",
          -      "app.kubernetes.io/version": "1.20.0",
          -      "app.kubernetes.io/managed-by": "Helm"
          -    }
          -  },
          +    "name": "test-nginx",
          +    "labels": {
          +      "app.kubernetes.io/name": "nginx",
          +      "helm.sh/chart": "nginx-7.3.12",
          +      "app.kubernetes.io/instance": "test",
          +      "app.kubernetes.io/version": "1.20.0",
          +      "app.kubernetes.io/managed-by": "Helm"
          +    }
            },
            "spec": {
              "ports": [
          @@ -14,6 +20,13 @@
                ],
                "selector": {
                  "app.kubernetes.io/name": "nginx",
          +          "app.kubernetes.io/instance": "test"
                }
              }
            }
          @@ -27,0 +39,14 @@
          +---
          +# Source: nginx/templates/controller-configmap.yaml
          +apiVersion: v1
          +kind: ConfigMap
          +metadata:
          +  name: RELEASE-NAME-nginx
          +  labels:
          +    app.kubernetes.io/name: nginx
          +    helm.sh/chart: nginx-7.3.12
          +    app.kubernetes.io/instance: RELEASE-NAME
          +    app.kubernetes.io/version: "1.20.0"
          +    app.kubernetes.io/managed-by: Helm
          +data:
          +{}
          @@ -38,0 +61,13 @@
          +---
          +# Source: nginx/templates/controller-deployment.yaml
          +apiVersion: apps/v1
          +kind: Deployment
          +metadata:
          +  name: RELEASE-NAME-nginx
          +  labels:
          +    app.kubernetes.io/name: nginx
          +    helm.sh/chart: nginx-7.3.12
          +    app.kubernetes.io/instance: RELEASE-NAME
          +    app.kubernetes.io/version: "1.20.0"
          +    app.kubernetes.io/managed-by: Helm
          +spec:
          +  replicas: 1
          +  selector:
          +    matchLabels:
          +      app.kubernetes.io/name: nginx
          +      app.kubernetes.io/instance: RELEASE-NAME
          +  template:
          +    metadata:
          +      labels:
          +        app.kubernetes.io/name: nginx
          +        app.kubernetes.io/instance: RELEASE-NAME
          +    spec:
          +      securityContext:
          +        fsGroup: 1001
          @@ -51,0 +78,3 @@
          +# Source: nginx/templates/controller-ingress.yaml
          +apiVersion: networking.k8s.io/v1beta1
          +kind: Ingress
          +metadata:
          +  name: RELEASE-NAME-nginx
          +  labels:
          +    app.kubernetes.io/name: nginx
          +    helm.sh/chart: nginx-7.3.12
          +    app.kubernetes.io/instance: RELEASE-NAME
          +    app.kubernetes.io/version: "1.20.0"
          +    app.kubernetes.io/managed-by: Helm
          +spec:
          +  rules: []
          +  tls: []
          @@ -54,0 +84,12 @@
          +{
          +  "kind": "Ingress",
          +  "apiVersion": "networking.k8s.io/v1beta1",
          +  "metadata": {
          +    "name": "test-nginx",
          +    "labels": {
          +      "app.kubernetes.io/name": "nginx",
          +      "helm.sh/chart": "nginx-7.3.12",
          +      "app.kubernetes.io/instance": "test",
          +      "app.kubernetes.io/version": "1.20.0",
          +      "app.kubernetes.io/managed-by": "Helm"
          +    }
          +  },
          +  "spec": {
          +    "rules": [],
          +    "tls": []
          +  }
          +}
          -# Source: nginx/templates/controller-serviceaccount.yaml
          -apiVersion: v1
          -kind: ServiceAccount
          -metadata:
          -  name: RELEASE-NAME-nginx
          -  labels:
          -    app.kubernetes.io/name: nginx
          -    helm.sh/chart: nginx-7.3.12
          -    app.kubernetes.io/instance: RELEASE-NAME
          -    app.kubernetes.io/version: "1.20.0"
          -    app.kubernetes.io/managed-by: Helm
          -spec:
          -  automountServiceAccountToken: false
          @@ -66,0 +99,12 @@
          -# Source: nginx/templates/controller-configmap.yaml
          -apiVersion: v1
          -kind: ConfigMap
          -metadata:
          -  name: RELEASE-NAME-nginx
          -  labels:
          -    app.kubernetes.io/name: nginx
          -    helm.sh/chart: nginx-7.3.12
          -    app.kubernetes.io/instance: RELEASE-NAME
          -    app.kubernetes.io/version: "1.20.0"
          -    app.kubernetes.io/managed-by: Helm
          -data:
          -{}
          @@ -77,0 +113,13 @@
          -# Source: nginx/templates/controller-deployment.yaml
          -apiVersion: apps/v1
          -kind: Deployment
          -metadata:
          -  name: RELEASE-NAME-nginx
          -  labels:
          -    app.kubernetes.io/name: nginx
          -    helm.sh/chart: nginx-7.3.12
          -    app.kubernetes.io/instance: RELEASE-NAME
          -    app.kubernetes.io/version: "1.20.0"
          -    app.kubernetes.io/managed-by: Helm
          -spec:
          -  replicas: 1
          -  selector:
          -    matchLabels:
          -      app.kubernetes.io/name: nginx
          -      app.kubernetes.io/instance: RELEASE-NAME
          -  template:
          -    metadata:
          -      labels:
          -        app.kubernetes.io/name: nginx
          -        app.kubernetes.io/instance: RELEASE-NAME
          @@ -89,0 +128,12 @@
          -      volumes:
          -        - name: config
          -          configMap:
          -            name: RELEASE-NAME-nginx
          -        - name: www
          -          emptyDir: {}
          -      initContainers:
          -      - name: init-chown-data
          -        image: busybox
          -        command: ['sh', '-c', 'chown -R 1001:1001 /var/lib/nginx']
          -        volumeMounts:
          -        - name: www
          -          mountPath: /var/lib/nginx
          -      containers:
          -      - name: nginx
          -        image: docker.io/bitnami/nginx:1.20.0-debian-10-r15
          @@ -101,0 +142,11 @@
          -          initialDelaySeconds: 10
          -          periodSeconds: 10
          -        ports:
          -        - name: http
          -          containerPort: 8080
          -        - name: https
          -          containerPort: 8443
          -        - name: proxy-http
          -          containerPort: 81
          @@ -112,0 +155,11 @@
          -      restartPolicy: Always
          -      terminationGracePeriodSeconds: 120
          -  strategy:
          -    type: RollingUpdate
          -    rollingUpdate:
          -      maxUnavailable: 1
          -      maxSurge: 1
          -  podDisruptionBudget:
          -    minAvailable: 1
          +diff -u /private/tmp/helmfile041161986/test-nginx-nginx/templates/controller-ingress.yaml /private/tmp/helmfile041161986/test-nginx-nginx-new/templates/controller-ingress.yaml
          --- /private/tmp/helmfile041161986/test-nginx-nginx/templates/controller-ingress.yaml	2020-09-18 11:24:15.000000000 +0800
          ++++ /private/tmp/helmfile041161986/test-nginx-nginx-new/templates/controller-ingress.yaml	2020-09-18 11:23:43.000000000 +0800
          @@ -0,0 +1,12 @@
          +{# Source: nginx/templates/controller-ingress.yaml #}
          +apiVersion: networking.k8s.io/v1beta1
          +kind: Ingress
          +metadata:
          +  name: {{ include "common.names.fullname". }}-ingress
          +  labels:
          +    app.kubernetes.io/name: {{ include "common.labels.name". }}
          +    helm.sh/chart: {{ include "common.labels.chart". }}
          +    app.kubernetes.io/instance: {{.Release.Name }}
          +    app.kubernetes.io/version: {{.Chart.AppVersion | quote }}
          +    app.kubernetes.io/managed-by: {{.Release.Service }}
          +spec:
          +  rules:
          +  - http:
          +      paths:
          +      - backend:
          +          serviceName: {{ include "common.names.fullname". }}
          +          servicePort: 80
          +      - backend:
          +          serviceName: {{ include "common.names.fullname". }}
          +          servicePort: 443
          +{{- if $.Values.metrics.enabled }}
          +---
          +# Source: nginx/templates/metrics-svc.yaml #}
          +apiVersion: v1
          +kind: Service
          +metadata:
          +  name: {{ include "common.names.fullname". }}-metrics
          +  labels:
          +    app.kubernetes.io/name: {{ include "common.labels.name". }}
          +    helm.sh/chart: {{ include "common.labels.chart". }}
          +    app.kubernetes.io/instance: {{.Release.Name }}
          +    app.kubernetes.io/version: {{.Chart.AppVersion | quote }}
          +    app.kubernetes.io/managed-by: {{.Release.Service }}
          +spec:
          +  ports:
          +  - name: metrics
          +    port: 9113
          +    targetPort: exporter-nginx
          +  selector:
          +    app.kubernetes.io/name: {{ include "common.labels.name". }}
          +    app.kubernetes.io/instance: {{.Release.Name }}
          +---
          +# Source: nginx/templates/metrics-deployment.yaml #}
          +apiVersion: apps/v1
          +kind: Deployment
          +metadata:
          +  name: {{ include "common.names.fullname". }}-metrics
          +  labels:
          +    app.kubernetes.io/name: {{ include "common.labels.name". }}
          +    helm.sh/chart: {{ include "common.labels.chart". }}
          +    app.kubernetes.io/instance: {{.Release.Name }}
          +    app.kubernetes.io/version: {{.Chart.AppVersion | quote }}
          +    app.kubernetes.io/managed-by: {{.Release.Service }}
          +spec:
          +  replicas: 1
          +  selector:
          +    matchLabels:
          +      app.kubernetes.io/name: {{ include "common.labels.name". }}
          +      app.kubernetes.io/instance: {{.Release.Name }}
          +  template:
          +    metadata:
          +      labels:
          +        app.kubernetes.io/name: {{ include "common.labels.name". }}
          +        app.kubernetes.io/instance: {{.Release.Name }}
          +    spec:
          +      containers:
          +      - name: exporter
          +        image: {{.Values.exporterImage }}
          +        imagePullPolicy: {{.Values.image.pullPolicy }}
          +        env:
          +        - name: NGINX_STATUS_PORT
          +          value: "8080"
          +        ports:
          +        - name: http
          +          containerPort: 9113
          +{{ end }}
          \ No newline at end of file
          ```
          ### 4.4.2 应用
          执行 `helmfile apply` 命令，可以应用 Helmfile 生成的 Kubernetes 资源清单。
          ```shell
          $ helmfile apply
          creating build folder "/Users/supereagle/.helm/build/"
          creating cache folder "/Users/supereagle/.cache/helm"
          Building dependency map
          Running engine.Prepare("default")
          Running hook: kubernetes before hooks [apply]
          Creating /Users/supereagle/.cache/helm/repository/local
          Building chart dependencies
          create custom resource definitions
          Installing release=test, chart=/Users/supereagle/code/bitnami/charts/bitnami/nginx
          coalesce.go:196: warning: cannot overwrite table with non table for issuers (map[])
          Created deploy/test-nginx-nginx.yaml
          Running hook: templated before hooks [apply]
          W0918 11:25:27.659952   22042 warnings.go:67] extensions/v1beta1 Ingress is deprecated in v1.14+, unavailable in v1.22+; use networking.k8s.io/v1 Ingress
          created ingress.extensions/test-nginx-nginx
          Running hook: cleanup before hooks [apply]
          Deleting release=test, chart=/Users/supereagle/code/bitnami/charts/bitnami/nginx
          Running hook: after-apply-manifests
          Updated "default" successfully
          ```
          ### 4.4.3 清除
          执行 `helmfile destroy` 命令，可以清除之前安装的所有 Helm releases。
          ```shell
          $ helmfile destroy
          deleting release=test, chart=/Users/supereagle/code/bitnami/charts/bitnami/nginx
          Release "test" does not exist. skipping deletion
          ```