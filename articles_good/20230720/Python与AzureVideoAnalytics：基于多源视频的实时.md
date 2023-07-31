
作者：禅与计算机程序设计艺术                    
                
                
近年来，随着人工智能、云计算、物联网等技术的迅速发展，视频流媒体的应用越来越广泛。而在实际场景中，需要对不同来源的视频进行融合，获取更加丰富和生动的画面效果。Azure提供了Video Analytics服务，可以对多源视频进行分析处理，生成不同的信息。本文将介绍如何使用Azure Video Analyzer API实现多源视频实时分析。

# 2.基本概念术语说明
## A.什么是Azure Video Analyzer
Azure Video Analyzer是一个视频分析平台服务，可帮助客户快速构建、部署、扩展和管理实时视频分析解决方案。它提供了一个统一的API接口，使得开发人员能够轻松地通过编程方式使用其功能。

## B.什么是多源视频分析
多源视频分析（Multi-Source Video Analysis）是指利用多种摄像头设备或视频源，以同步的方式实时分析、处理、分析并提取视频中的信息，从而能够构建出更加丰富、动态的信息图象。传统的单源视频分析只能从一个摄像头设备或视频源中获取信息，而多源视频分析则可以通过多个摄像头或视频源一起工作，共享同一份数据进行分析处理。

## C.什么是Azure Video Analyser Edge Module
Edge模块是一种与Azure IoT Edge运行时的IoT模块。它与Azure Video Analyzer Cloud Service建立了通信连接，通过Azure IoT Hub在边缘端节点与云端交互。它的作用是对视频流进行实时分析处理，输出分析结果并将结果写入到Azure Blob Storage或其他云存储中。

## D.Azure Video Analyzer API
Azure Video Analyzer API提供的功能包括：

- **创建帐户** - 创建视频分析账号。
- **上传视频** - 将本地视频文件上传至云端。
- **创建资产** - 在视频分析账号中定义多源视频资产。
- **设置管道模板** - 使用预设的流水线模板快速建立视频分析管道。
- **调整管道配置** - 根据视频需求调整视频分析管道配置。
- **启动管道** - 通过调用管道方法来启动视频分析管道。
- **监视管道状态** - 通过调用管道方法获取管道状态。
- **检索分析结果** - 从Blob Storage检索分析结果。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 一、安装Azure CLI
要在计算机上安装Azure命令行界面(CLI)，请按照以下链接：https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

## 二、安装Azure Video Analyzer CLI扩展
运行以下命令安装Azure Video Analyzer CLI扩展：
```
az extension add --name video-analyzer
```

## 三、设置Azure Subscription
要设置Azure Subscription，请登录Azure Portal，然后在顶部导航栏中选择“订阅”。点击“+添加”，选择需要使用的Subscription ID。

## 四、创建Azure Resource Group
运行以下命令创建Azure Resource Group：
```
az group create -l <location> -n <resource_group_name>
```

例如：
```
az group create -l eastus2 -n ava-demo
```

## 五、创建Azure Video Analyzer Account
运行以下命令创建Azure Video Analyzer Account：
```
az video analyzer account create \
    -g <resource_group_name> \
    -n <account_name> \
    --storage-account "<storage_account_id>" \
    --location "centraluseuap" \
    --mi-system-assigned
```

其中，`--storage-account`参数用于指定Azure Storage Account的ID。若已有一个Storage Account可以直接使用该Account的ID，也可以创建一个新的Storage Account，然后将其ID用双引号包裹起来输入。另外， `--mi-system-assigned`参数用于为系统分配托管标识，该标识由Azure自动创建并管理，不需要手动创建或导入任何证书。

例子：
```
az video analyzer account create \
    -g ava-demo \
    -n ava-account \
    --storage-account "/subscriptions/<subscription_id>/resourceGroups/ava-demo/providers/Microsoft.Storage/storageAccounts/avasystemtest" \
    --location "centraluseuap" \
    --mi-system-assigned
```

## 六、创建Azure Video Analyzer Edge Module
运行以下命令创建Azure Video Analyzer Edge Module：
```
az iot edge set-modules \
    --hub-name <iot_hub_name> \
    --device-id <edge_device_id> \
    --content "{
        \"modules\":[{
            \"name\":\"avaedge\",
            \"version\":\"1.0\",
            \"type\":\"docker\",
            \"settings\":{\"image\":\"mcr.microsoft.com/media/video-analyzer:1\"},
            \"env\":{
                \"LOCAL_USER_ID\": \"1000\",
                \"LOCAL_GROUP_ID\": \"1000\",
                \"VIDEO_INPUT_FOLDER_ON_DEVICE\": \"/var/lib/videoanalyzer/inputs\",
                \"VIDEO_OUTPUT_FOLDER_ON_DEVICE\": \"/var/media\"",
                \"VAMANA_CONFIG_APPDATA_PATH\": \"/var/lib/videoanalyzer\"
            },
            \"provisioningOptions\":{}
        }]
    }"
```

其中，`--hub-name`参数用于指定Azure IoT Hub的名称，`--device-id`参数用于指定Azure IoT Edge Device的ID。另外，`IMAGE`环境变量设置为Azure Video Analyzer Edge Module镜像地址。

## 七、上传本地视频文件
要上传本地视频文件，请在目标计算机上找到视频文件所在的目录，运行以下命令将视频文件上传至Cloud Storage：
```
az storage blob upload \
  --account-name <storage_account_name> \
  --container-name input \
  --file <local_video_path> \
  --name <video_file_name>
```

其中，`--account-name`参数用于指定Azure Storage Account的名称，`--container-name`参数用于指定上传到的Container Name，`--file`参数用于指定本地视频文件的路径，`--name`参数用于指定上传视频的文件名。

例子：
```
az storage blob upload \
  --account-name avasystemtest \
  --container-name input \
  --file ~/Videos/test.mp4 \
  --name test.mp4
```

## 八、创建Azure Video Analyzer Asset
运行以下命令创建Azure Video Analyzer Asset：
```
az video analyzer asset create \
    --account-name <account_name> \
    --resource-group <resource_group_name> \
    --name <asset_name> \
    --alternate-id <alternative_id> \
    --description <asset_description> \
    --endpoint "<rtsp_url>" \
    --batch-size=<batch_size>
```

其中，`--account-name`参数用于指定Azure Video Analyzer Account的名称，`--resource-group`参数用于指定Azure Resource Group的名称，`--name`参数用于指定Asset的名称，`--alternate-id`参数用于指定Asset的备选ID，`--description`参数用于指定Asset的描述信息，`--endpoint`参数用于指定RTSP URL。若有多个摄像头或视频源需要加入Asset，则可以通过指定多个RTSP URL来完成。`--batch-size`参数用于指定每个批次上传的最大视频帧数目。

例子：
```
az video analyzer asset create \
    --account-name ava-account \
    --resource-group ava-demo \
    --name camera1 \
    --endpoint rtsp://admin:password@camera1.contoso.net:554/Streaming/Channels/101 \
    --batch-size 50
```

## 九、设置Azure Video Analyzer Pipeline Template
运行以下命令设置Azure Video Analyzer Pipeline Template：
```
az video analyzer pipeline topology set \
   --account-name <account_name> \
   --resource-group <resource_group_name> \
   --name <pipeline_topology_name> \
   --set-param '[ {"name": "rtspUserName", "value": "<user_name>"},
                  {"name": "rtspPassword", "value": "<password>"}]' 
```

其中，`--account-name`参数用于指定Azure Video Analyzer Account的名称，`--resource-group`参数用于指定Azure Resource Group的名称，`--name`参数用于指定Pipeline Topology的名称。使用前一步设置的Asset，需要设置`--add`选项，指定Pipeline Topology所需的所有Assets。

例子：
```
az video analyzer pipeline topology set \
   --account-name ava-account \
   --resource-group ava-demo \
   --name pipeline1 \
   --set-param '[{"name":"rtspUserName","value":"admin"},{"name":"rtspPassword","value":"password"}]' 
   --add "[ {
      \"name\": \"camera1\", 
      \"type\": \"RtspSource\", 
      \"transport\": \"tcp\" 
    } ]"
```

## 十、调整Azure Video Analyzer Pipeline Configuration
运行以下命令调整Azure Video Analyzer Pipeline Configuration：
```
az video analyzer pipeline job set \
   --account-name <account_name> \
   --resource-group <resource_group_name> \
   --job-name <job_name> \
   --pipeline-name <pipeline_name> \
   --topology-name <pipeline_topology_name> \
   [--parameters '<key>=<value>' ['...']]
```

其中，`--account-name`参数用于指定Azure Video Analyzer Account的名称，`--resource-group`参数用于指定Azure Resource Group的名称，`--job-name`参数用于指定Job的名称，`--pipeline-name`参数用于指定Pipeline的名称，`--topology-name`参数用于指定Pipeline Topology的名称。使用前一步设置的Pipeline Topology，需要设置`--pipeline-topology-name`参数。此外，还可以设置自定义的参数，可以使用`--parameters`参数传入，形如`<key>=<value>`的形式。

例子：
```
az video analyzer pipeline job set \
   --account-name ava-account \
   --resource-group ava-demo \
   --job-name myfirstjob \
   --pipeline-name myfirstpipline \
   --topology-name pipeline1 \
   --parameters "[{\"name\":\"rtspUrl\",\"value\":\"rtsp://admin:password@camera1.contoso.net:554/Streaming/Channels/101\"}]"
```

## 十一、启动Azure Video Analyzer Pipeline
运行以下命令启动Azure Video Analyzer Pipeline：
```
az video analyzer pipeline job start \
   --account-name <account_name> \
   --resource-group <resource_group_name> \
   --name <job_name>
```

其中，`--account-name`参数用于指定Azure Video Analyzer Account的名称，`--resource-group`参数用于指定Azure Resource Group的名称，`--job-name`参数用于指定Job的名称。

例子：
```
az video analyzer pipeline job start \
   --account-name ava-account \
   --resource-group ava-demo \
   --name myfirstjob
```

## 十二、监视Azure Video Analyzer Pipeline Status
运行以下命令监视Azure Video Analyzer Pipeline Status：
```
az video analyzer pipeline job list \
   --account-name <account_name> \
   --resource-group <resource_group_name> \
   [--top]
```

其中，`--account-name`参数用于指定Azure Video Analyzer Account的名称，`--resource-group`参数用于指定Azure Resource Group的名称。此外，可以设置`--top`参数显示最新几个Job信息。

例子：
```
az video analyzer pipeline job list \
   --account-name ava-account \
   --resource-group ava-demo \
   --top
```

## 十三、检索Azure Video Analyzer Pipeline Result
运行以下命令检索Azure Video Analyzer Pipeline Result：
```
az video analyzer pipeline job get-pipeline-instance-runtime-details \
   --account-name <account_name> \
   --resource-group <resource_group_name> \
   --job-name <job_name> \
   --name <pipeline_name>
```

其中，`--account-name`参数用于指定Azure Video Analyzer Account的名称，`--resource-group`参数用于指定Azure Resource Group的名称，`--job-name`参数用于指定Job的名称，`--name`参数用于指定Pipeline的名称。

例子：
```
az video analyzer pipeline job get-pipeline-instance-runtime-details \
   --account-name ava-account \
   --resource-group ava-demo \
   --job-name myfirstjob \
   --name myfirstpipeline
```

# 4.具体代码实例和解释说明
我们准备了一段Python代码演示如何使用Azure Video Analyzer API进行多源视频实时分析。代码如下：

```python
import json
from azure.identity import DefaultAzureCredential
from azure.mgmt.videoanalyzer import VideoAnalyzerManagementClient
from azure.mgmt.media import AzureMediaServices

# Create Azure Media Services client
credential = DefaultAzureCredential()
client = AzureMediaServices(credential)

# Get a list of existing media services in the subscription
media_services = list(client.list())

if not media_services:
    print("No media service exists in this subscription.")

else:

    # Use the first media service in the subscription
    resource_group_name = media_services[0].resource_group_name
    account_name = media_services[0].name
    
    # Set up the Video Analyzer Management Client with the current credentials and subscription
    location = media_services[0].location
    credential = DefaultAzureCredential()
    client = VideoAnalyzerManagementClient(credential, subscription_id, base_url="https://" + location + ".api.cognitive.microsoft.com")

    # Set up the parameters for creating an access policy
    policy_name ='myAccessPolicy'
    permissions ='read'

    # Create an access policy
    print('Creating an Access Policy')
    client.access_policies.create_or_update(resource_group_name, account_name, policy_name, permissions)

    # Get details of the default streaming endpoint on the account
    print('Getting details of the default Streaming Endpoint')
    default_streaming_endpoint = next((se for se in client.streaming_endpoints if se.name == "default"), None)

    # Check if there is already a live event on the account, otherwise create one
    print('Checking for Live Event')
    live_events = [e for e in client.live_events if e.name == "avaevent"]
    if len(live_events) == 0:

        # If no live event found, create one with a unique name by adding a random number at the end
        live_event_name = f"{account_name}-avaevent-{random.randint(1000,9999)}"

        # Create a live event with RTMP as the ingest protocol
        print(f"Creating a new Live Event named '{live_event_name}' using the RTMP ingest protocol...")
        properties = {'description': '',
                      'input': {'streamingProtocol': 'RTMP',
                                'accessControl': {'ip': {'allow': []}}},
                      'preview': {'accessControl': {'ip': {'allow': ['*']}}}}
        live_event = client.live_events.begin_create(resource_group_name, account_name, live_event_name, properties).result()
        print(f"Live Event created successfully with id '{live_event.id}'.
")

    else:
        # If a live event was found, use that one instead
        live_event_name = live_events[0].name
        print(f"Using existing Live Event '{live_event_name}'.")

    # Define the recording policies to record from both sources simultaneously
    archive_enabled = True
    preview_lifetime_seconds = 60 * 60 * 24  # 1 day
    recording_policy1 = {
        "@odata.type": "#Microsoft.VideoAnalyzer.RecordingSink",
        "name": "recordingsink1",
        "inputs": ["camera1"],
        "archiveWindowLength": "PT5M",
        "filenamePattern": "${systemSequenceNumber}-${systemDateTime}.mkv"
    }
    recording_policy2 = {
        "@odata.type": "#Microsoft.VideoAnalyzer.RecordingSink",
        "name": "recordingsink2",
        "inputs": ["camera2"],
        "archiveWindowLength": "PT5M",
        "filenamePattern": "${systemSequenceNumber}-${systemDateTime}.mkv"
    }
    sinks = {"sinks": [recording_policy1, recording_policy2]}

    # Set up the parameters for creating a live output to record the stream
    live_output_name = "recording1"
    encoding_type = "Premium1080p"

    # Configure the Live Output to use the custom recordings settings defined earlier
    print('
Creating a new Live Output named {}.'.format(live_output_name))
    properties = {'description': "",
                  'priority': "Low",
                  'archiveEnabled': archive_enabled,
                 'minLatencySeconds': 5,
                 'maxLatencySeconds': 10,
                  'encodingType': encoding_type,
                  'videoName': '${System.Runtime.GraphInstanceName}',
                 'sourceDirectories': ["/var/media/*"],
                 'sinks': [
                      {
                          '@odata.type': '#Microsoft.VideoAnalyzer.VideoSink',
                          'baseDirectory': '/var/media/',
                          'fileNameFormat': '%Y/%m/%d/'
                      },
                      sinks['sinks'][0],
                      sinks['sinks'][1]
                  ]}
    live_output = client.live_outputs.begin_create(resource_group_name, account_name, live_event_name, live_output_name, properties).result()

    # Start the live event once it has been created and wait until it starts running before continuing with the script execution
    print("
Starting the live event...")
    client.live_events.begin_start(resource_group_name, account_name, live_event_name).wait()
    state = client.live_events.get(resource_group_name, account_name, live_event_name).state
    while state!= "Running":
        time.sleep(10)
        state = client.live_events.get(resource_group_name, account_name, live_event_name).state
        print(".", end='')
    print("
Live event started.
")

    # Wait for the live output to be in running state before starting the analysis process
    print("Waiting for the live output to reach 'Running' state...")
    live_output_status = ''
    while live_output_status!= "Running":
        time.sleep(10)
        live_output_status = client.live_outputs.get(resource_group_name, account_name, live_event_name, live_output_name).status
        print('.', end='')
    print("
Live output running.
")

    # Once the live output is running, initiate the stream analytics job
    print("Initiating the stream analytics job...")
    sa_client = StreamAnalyticsManagementClient(credential, subscription_id, base_url='https://management.azure.com/')
    sa_jobs = list(sa_client.stream_analytics_functions.list_by_resource_group(resource_group_name))
    if not any(j for j in sa_jobs if j.properties.name=='stream-analytics-job'):
        try:

            # Deploy the stream analytics job template using ARM deployment template
            deploy_template_uri = 'https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/quickstarts/microsoft.stream-analytics/asa-videoindexer-livepeer/azuredeploy.json'
            
            arm_params = {
                    '_artifactsLocation': 'https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/quickstarts/microsoft.stream-analytics/asa-videoindexer-livepeer',
                    'liveEventName': live_event_name,
                    'accessTokenSecretName': 'ava-access-token-secret'
                }
            
            scope = str(resource_group_name)
            DeploymentProperties = namedtuple('DeploymentProperties', ['mode', 'template', 'parameters'])
            props = DeploymentProperties('Incremental', { '$schema': 'https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#', 'contentVersion': '1.0.0.0','resources': [{'apiVersion': '2017-04-01-preview', 'type': 'Microsoft.StreamAnalytics/streamingjobs', 'name':'stream-analytics-job', 'location': 'West US 2','sku': {'name': 'Standard'}, 'properties': {'compatibilityLevel': '1.3', 'dataLocale': 'en-US', 'inputs': [{'dataType': 'Stream', 'name':'streamInput1','serialization': {'type': 'Json'}}, {'dataType': 'Reference', 'name':'metadata','referenceGroupName': 'videoIndexersMetadataRefGroup'}], 'transformation': {'name': 'Stream Analytics Query', 'query': 'SELECT MetadataCacheOutput INTO [videoIndexersMetadataRefGroup] FROM [streamInput1]; SELECT Camera1Output INTO [Camera1Output] FROM [streamInput1] TIMESTAMP BY Timestamp SELECT Camera2Output INTO [Camera2Output] FROM [streamInput1] TIMESTAMP BY Timestamp','streamingUnits': 3}}, {'apiVersion': '2017-04-01-preview', 'type': 'Microsoft.KeyVault/vaults', 'name': 'keyvault', 'location': 'westus2', 'properties': {'tenantId': '{{ subscription().tenantId }}','sku': {'family': 'A', 'name':'standard'}, 'accessPolicies': [], 'enableSoftDelete': True}}], 'outputs': {}, 'parameters': {'liveEventName': {'type':'string', 'defaultValue': live_event_name}, 'accessTokenSecretName': {'type':'secureString'}}} ), arm_params )
            template = props.template
            
            params = props.parameters
            params = { k: {'value': v} for k, v in params.items()}
            
            result = deployment_client.deployments.validate(scope,'stream-analytics-arm-template', props.template, params)
            
        except Exception as ex:
            print('Failed deploying resources.')
            raise ex
        
        try:
            # Create or update the KeyVault secret containing the access token required for authenticating with the Video Indexer API
            kv_secrets = vault_client.secrets.list(resource_group_name, account_name)
            token_secret = next((s for s in kv_secrets if s.name==props.parameters['accessTokenSecretName']['value']), None)
            if token_secret is None:
                secret_bundle = VaultSecretBundle(value='', attributes=SecretAttributes(enabled=True), tags=None)
                secret_bundle = vault_client.secrets.create_or_update(resource_group_name, account_name, props.parameters['accessTokenSecretName']['value'], secret_bundle)
            else:
                secret_bundle = vault_client.secrets.get(resource_group_name, account_name, props.parameters['accessTokenSecretName']['value'])
                
            # Start the stream analytics job
            job_properties = JobCreateOrUpdateParameters(input=Input(type='Stream'), output=[Output(datasource=OutputDataSource(type='Microsoft.ServiceBus/EventHub', property_columns={'Column1': 'Column1'}), serialization=Serialization(type='Json', format='LineSeparated')), Output(datasource=OutputDataSource(type='PowerBI'), name='Camera1PBI')], transformation=Transformation(name='Query', query='SELECT MetadataCacheOutput INTO [videoIndexersMetadataRefGroup] FROM [streamInput1]; SELECT Camera1Output INTO [Camera1Output] FROM [streamInput1] TIMESTAMP BY Timestamp SELECT Camera2Output INTO [Camera2Output] FROM [streamInput1] TIMESTAMP BY Timestamp'))
            job_creation_result = sa_client.stream_analytics_functions.create_or_update(resource_group_name,'stream-analytics-job', job_properties)
            print("Successfully deployed resources and started the stream analytics job.")
        
        except Exception as ex:
            print('Failed initiating the stream analytics job.')
            raise ex
        
    else:
        print("The stream analytics job already exists. Skipping deployment and just starting it.")
    
    # Connect to the Video Analyzer and Stream Analytics clients and run queries against their APIs
    vi_client = VideoIndexer(api_key=os.environ["VIDEOINDEXER_KEY"])
    sa_jobs = list(sa_client.stream_analytics_functions.list_by_resource_group(resource_group_name))
    sa_job = next((j for j in sa_jobs if j.properties.name=='stream-analytics-job'), None)
    token_secret = vault_client.secrets.get(resource_group_name, account_name, props.parameters['accessTokenSecretName']['value']).value
    video_indexers_metadata_ref_group = next((i for i in sa_job.properties.inputs if i.name=='videoIndexersMetadataRefGroup'), None)
    video_indexer_api_url = os.getenv('VIDEOINDEXER_API_URL', '')
    assert video_indexer_api_url!= '', 'Please provide VIDEOINDEXER_API_URL environment variable.'
    
    def get_access_token():
        return VideoIndexerAuthentication(os.environ["VIDEOINDEXER_ACCOUNT"]).generate_access_token()
    
    
    # Run a few sample queries against the Video Indexer API
    print('Fetching insights about each recorded video...')
    videos = vi_client.get_all_videos()
    for video in videos:
        insights = vi_client.get_video_insights(video.id, access_token=get_access_token(), refresh=True)['videos'][0]['insights']
        metadata_cache = []
        for i in range(len(insights)):
            insight = insights[i]
            if insight['type'] == 'KeyFrame':
                break
            elif (insight['type'] == 'Shot' or insight['type'] == 'Scene') and i > 0:
                start = datetime.datetime.utcfromtimestamp(int(insights[i-1]['instances'][0]['end']))
                end = datetime.datetime.utcfromtimestamp(int(insight['instances'][0]['start']))
                duration = int((end - start).total_seconds())
                shot = {
                    'cameraId': video.id,
                   'startTimeUtc': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'endTimeUtc': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'durationInSeconds': duration
                }
                metadata_cache.append(shot)
        cache_output = Message({
            'MetadataCacheOutput': metadata_cache, 
            'TimeStamp': datetime.datetime.utcnow().isoformat()
        })
        iothub_service_client.send_message_to_output(str(cache_output), video_indexers_metadata_ref_group.serialization.format, video_indexers_metadata_ref_group.datasource)
    
    # Run another sample query against the Stream Analytics job
    print('Fetching statistics about each analyzed video...')
    query_results = iothub_registry_manager.get_twin(sa_job.id)
    cameras = [(item['value']['Camera1Output'], item['value']['Camera2Output']) for item in query_results['properties']['outputs'][-1]['properties']['datasource']['properties']['values']]
    for cam1, cam2 in cameras:
        stats = {
            'cameraId': video.id,
            'camera1DurationInMinutes': round(float(cam1['timestamps'][1]) / 60, 2),
            'camera2DurationInMinutes': round(float(cam2['timestamps'][1]) / 60, 2),
            'overlapDurationInMinutes': round(((abs(int(cam1['timestamps'][1]) - int(cam2['timestamps'][1]))) / 60), 2)
        }
        sb_client.send_message(QueueMessage(content=str(stats)))
        
```

以上代码涵盖了Azure Video Analyzer API中最常用的一些功能。具体操作流程请参考注释。

