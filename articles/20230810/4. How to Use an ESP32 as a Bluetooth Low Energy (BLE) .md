
作者：禅与计算机程序设计艺术                    

# 1.简介
         

Bluetooth Low Energy（以下简称 BLE），是一种低功耗、低数据速率、短距离、双向通信的无线传输协议。它使用 2.4GHz 的低频率蓝牙电波进行通讯。通过引入可靠传输层协议 (RTCP)，该协议可以确保数据可靠送达，并且可以检测到接收方丢失的数据包。BLE 是继 Wi-Fi 和 Zigbee 之后，第三个领先的低功耗蓝牙技术。它的广泛应用使得其成为物联网技术的核心。

ESP32 是一个基于模组、嵌入式系统微控制器的开源 Wi-Fi/BT/BLE SoC 芯片。本教程将介绍如何使用 ESP32 搭建一个 BLE 透传器设备和一个 BLE 收发器设备。透传器设备将可接受到的待发送数据的类型转换成 BLE 数据格式并发送出去；收发器设备则将收到的 BLE 数据格式转换成对应类型的原始数据。这样，BLE 设备就可以作为 BT / BLE 模块实现应用之间的通信。

在此之前，需要对 BLE 有一定的了解。BLE 的工作方式类似于 TCP/IP 报文，分为广播模式、点对点连接模式和 GATT 访问模式。其中，GATT (Generic Attribute Profile，通用属性配置文件) 是 BLE 设备交流的基础。GATT 中定义了服务 (Service)、特征 (Characteristics)、描述符 (Descriptors) 和属性值 (Attribute Values)。


# 2.基本概念术语说明
## 2.1 BLE 设备角色及通信模式
BLE 在不同的角色之间切换时，存在两种主要的通信模式：广播模式（也叫透射模式）和点对点连接模式。如下图所示：




如上图所示，BLE 的角色包括：初始化器（Initiator），响应器（Responder）。初始器通常是手机或其他远程终端，启动扫描，搜索 BLE 设备。当找到响应器设备后，就会自动建立点对点连接，双方可以互相传输数据。另外，BLE 还支持多种链接方式，包括主从链路（Primary-Secondary Linking）、广播链路（Broad-Cast Linking）和周期广播链路（Cycle Broad-Cast Linking）。这些链接方式共同构成了 BLE 协议栈的复杂性。

除了角色外，BLE 还存在着不同的数据传输方式。广播模式下，所有参与者都能收到消息，点对点模式下，只有两个设备才可以通信。因此，BLE 分为两种传输方式，分别为不可靠传输和可靠传输。

## 2.2 GATT 协议结构
GATT 协议主要由服务、特征和描述符三部分构成。服务是 BLE 设备中最基本的组件，它代表了一类相关的功能集合。例如，身体感应服务（Body Sensor Service，BSS）就是一个典型的服务。每一个服务都有一个 UUID 来唯一标识。

特征和描述符都是服务的组成部分，它们定义了该服务提供的具体数据。特征可以理解为数据采集点，它提供给用户用于收集数据的值或者指令。每个特征都有一个 UUID 来唯一标识。

最后，描述符则用来描述特征的值，比如特征值的单位、测量方法等信息。GATT 协议提供的能力是灵活的，可以通过配置描述符实现自定义的业务逻辑，满足各类特殊需求。

## 2.3 BLE 协议栈
在 BLE 的传输过程中，涉及到多个协议层次。如下图所示：



如上图所示，蓝色部分是主机（Host）协议栈，它处理的是客户端-服务器模型中的客户端协议。红色部分是从机（Peripheral）协议栈，它处理的是客户端-服务器模型中的服务器协议。蓝色部分还包括主机控制器接口（HCI）、基本传输单元（L2CAP）、串行连接管理器（SMP）、动态内存分配（MEM）、媒体访问控制（MAC）等子协议。从机协议栈的主要组件有 GAP （ Generic Access Profile，通用访问配置文件），GATT ，连接管理器（CM），安全管理器（SM），低功耗蓝牙（LL），物理层（PHY）。

一般来说，BLE 协议栈采用命令查询响应机制。客户端发送命令给从机，然后从机返回执行结果。命令包括读取特征值、写入特征值、订阅通知等，查询命令就是告诉从机要做什么事情。举例来说，主机向从机请求一个特征值的过程可以分为两步：第一步，主机向从机发送命令，要求得到某个特征值的最新值。第二步，从机把最新值返回给主机。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 BLE 透传器的设计流程
### 3.1.1 配置开发环境
1. 安装 Keil 工具链。Keil 是 IAR Embedded Workbench 的一个增强版本。如果没有安装，可以下载免费版 Keil IDE for ARM from https://www.keil.com/demo/eval.html。

2. 配置 STM32F4 Discovery 上电后连接调试端口所需串口号。

方法一：
- 使用板载 ST-Link 转接板，将 STM32F4 Discovery 连接到电脑上的 USB 口。

方法二：
- 使用纽扣串行总线连接 STM32F4 Discovery 和电脑。
- 将 STM32F4 Discovery 上电。
- 用数据线连接电脑的 TXD、RXD 和 GND 引脚。
- 查看 Windows Device Manager 中的 COMx 对应的串口号（其中 x 为数字）。

方法三：
- 在 STM32F4xx_DFP_Eval 库目录下新建一个名为 USARTx_Config.h 的文件，并打开编辑。
- 在文件中添加 `#define STM32F4_DISCOVERY` 宏定义，如下图所示。


方法四：直接修改源码。将 `usb_conf.c` 文件中的第 117 行的 `else if(USB_OTG_FS == Dev) {}` 修改为 `#if defined (STM32F4_DISCOVERY)` 。

```c
#elif defined (STM32F4_DISCOVERY) // STM32F4 Discovery
#define USE_USB_OTG_HS    __HAL_RCC_USB_OTG_HS_CLK_ENABLE()
#define OTG_FS            __USB_OTG_HS
#define PCD_FS            hpcd_USB_OTG_FS
#define ID                USB_OTG_HS_ID
#define DP_PULLUP         GPIO_PULLUP_ENABLE
#define DM_PULLDOWN       GPIO_NOPULL
#define DATA_LINE_TX      GPIO_PIN_6
#define DATA_LINE_RX      GPIO_PIN_7
#define POWER_LINE        GPIO_PIN_14
#define VBUS_SENSE_PORT   GPIOB
#define VBUS_SENSE_PIN    GPIO_PIN_5
#define PHY_CLOCK         RCC_PLLSAI2_USBCLKSOURCE_PLL
#else
/*... */
}
```

3. 配置 STM32F4xx_StdPeriph_Driver 库。打开项目属性页，选择“Debug”标签，在设置选项卡中选择加载目标并确认选择为 STM32F407VGTx，然后保存。

4. 添加 STM32F4 HAL Library。使用 STM32CubeMX 从头开始创建项目。右键点击 STM32F4 项目文件夹，选择 “Add Existing Hardware” -> “Manage All Components”。选择 STM32F4xx_HAL_Driver 和 STM32F4xx_HAL_GPIO 库。

5. 添加任务和事件。在源文件 main.c 中，定义三个任务和三个事件：

- LED 任务：一个无限循环，一直循环输出 LED 状态。
- BLE 透传任务：一个等待接收数据的线程。
- BLE 回调任务：一个等待事件发生的线程。

```c
int led_task(void *arg) {
while(1) {
HAL_GPIO_TogglePin(GPIOD, LD4_Pin); // Toggle the LED on PD4
osDelay(500);                      // Delay of 500ms
}
return 0;
}

void ble_rx_task(void const * argument) {
uint8_t buffer[DATA_LEN];
int len = sizeof(buffer);

while (1) {
err_code = sd_ble_gattc_read(conn_handle, rx_handle, offset, &len, buffer);
CHECK_ERROR(err_code);

// Process received data
//...

event_set(event_data_received);
}
}

void ble_callback_task(void const * argument) {
while (1) {
uint32_t e = event_wait(EVENT_ALL);

switch (e) {
case EVENT_CONNECTED:
// Connected to device
break;

case EVENT_DISCONNECTED:
// Disconnected from device
break;

case EVENT_DATA_RECEIVED:
// Data received
break;

default: 
break;
}
}
}

int app_main(void) {
TaskHandle_t tasks[] = {
NULL,                  // LED task not created yet
osThreadCreate(osPriorityHigh, 0, ble_rx_task, "ble_rx", NULL),
osThreadCreate(osPriorityHigh, 0, ble_callback_task, "ble_cb", NULL),
};

// Create tasks
tasks[LED_TASK] = osThreadCreate(osPriorityLow, 1, led_task, "led", NULL);

// Start scheduler
osKernelStart();
while (1) {
// Just hang here...
}
return 0;
}
```

### 3.1.2 创建服务和特征
为了让 STM32F4 Discovery 可以作为 BLE 透传器，首先需要创建一个 GATT 服务。服务由多个特征组合而成，每一个特征代表了一个可读写的变量。

1. 定义服务和特征的 UUID。在源文件 main.c 中，定义两个 UUID。

```c
static uint8_t tx_uuid[UUID_SIZE] = {
0xAB, 0xBC, 0xDE, 0xEF, 0xAA, 0xBB, 0xCC, 0xD0, 0xEE, 0xF1
};

static uint8_t rx_uuid[UUID_SIZE] = {
0xAD, 0xBE, 0xED, 0xFE, 0xCA, 0xBA, 0xDB, 0xDC, 0xFA, 0xEB
};
```

注意：UUID 的长度必须为 16 个字节。这里使用的 UUID 只是示例，实际应用中可以使用自定义的 UUID。

2. 创建服务和特征。在源文件 main.c 中，创建服务和特征。

```c
// Add services and characteristics to BLE database
ble_add_service(&tx_uuid[0], THRIFT_TX_CHAR_UUID_LEN, &thrift_tx_char);
ble_add_service(&rx_uuid[0], THRIFT_RX_CHAR_UUID_LEN, &thrift_rx_char);
```

以上函数会将上面定义的 UUID 作为参数传入，创建服务，并在服务下创建特征。

3. 配置服务特性的属性。在源文件 main.c 中，配置服务特性的属性。

```c
static struct ble_gatts_char_handles thrift_rx_char = {0};
static ble_gatts_char_md_t char_md = {
.char_props.write = 1,                     // Characteristic is writable by remote client
.char_props.indicate = 1                   // Client will receive notifications when characteristic changes
};

static ble_gatts_attr_md_t cccd_md = {
.vloc = BLE_GATTS_VLOC_STACK,              // Attribute will be stored in attribute table on stack
};

static uint8_t value[VALUE_LEN] = {0};          // Value to send over BLE
static uint8_t cccd_value[2] = {0xC0, 0x00};     // Notification enabled by default

static ble_gatts_attr_t attrs[] = {
[THRIFT_TX_CHAR_HANDLE] = {
.p_uuid = &rx_uuid[0],                 // Pointer to UUID array
.max_len = VALUE_LEN,                   // Maximum length of this attribute
.init_len = sizeof(value),             // Length set during initialization
.p_value = &value[0],                  // Pointer to initial value
.flags = BLE_GATT_CHRC_WRITE | BLE_GATT_CHRC_NOTIFY,
},
[THRIFT_RX_CHAR_CCCD_HANDLE] = {
.p_uuid = &cccd_uuid,                  // Pointer to CCCD's UUID
.max_len = sizeof(cccd_value),          // Maximum length of this attribute
.init_len = sizeof(cccd_value),         // Length set during initialization
.p_value = cccd_value,                  // Pointer to initial value
.flags = BLE_GATT_CHRC_READ | BLE_GATT_CHRC_INDICATE,
.p_cccd_md = &cccd_md                    // Pointer to CCCD metadata structure
},
};
```

以上代码定义了两个特征：第一个特征负责接收数据，第二个特征负责配置特性是否接收通知。

4. 注册服务和特征。在源文件 main.c 中，注册服务和特征。

```c
// Register service with GATT server
ble_enable_services(attrs, sizeof(attrs));
```

此函数会将上面定义的服务和特征注册到 GATT 数据库。

### 3.1.3 配置和运行 BLE 栈
1. 初始化系统时钟。在源文件 system_stm32f4xx.c 中，设置系统时钟。

```c
Clock_InitTypeDef clock_init = {0};

clock_init.ClockType = CLK_TYPE_SYSCLK;
clock_init.SYSCLKSource = CLK_SRC_PLLCLK;
clock_init.APB1CLKDivider = APB1_DIV1;
clock_init.AHBCLKDivider = AHB_DIV1;
SystemClock_Config(&clock_init);
```

以上代码会配置系统时钟为 72 MHz。

2. 配置 PLL 时钟。在源文件 system_stm32f4xx.c 中，配置 PLL 时钟。

```c
RCC_OscInitTypeDef oscillator_init = {0};
RCC_ClkInitTypeDef clock_init = {0};

oscillator_init.OscillatorType = RCC_OSCILLATORTYPE_HSE;
oscillator_init.HSEState = RCC_HSE_ON;
oscillator_init.PLL.PLLState = RCC_PLL_NONE;
if (HAL_RCC_OscConfig(&oscillator_init)!= HAL_OK) {
Error_Handler();
}

clock_init.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
clock_init.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
clock_init.AHBCLKDivider = RCC_SYSCLK_DIV1;
clock_init.APB1CLKDivider = RCC_HCLK_DIV2;
clock_init.APB2CLKDivider = RCC_HCLK_DIV1;
if (HAL_RCC_ClockConfig(&clock_init, FLASH_LATENCY_2)!= HAL_OK) {
Error_Handler();
}
```

以上代码配置了 PLL 时钟。

3. 设置 STM32F4 Discovery 引脚。在源文件 stm32f4xx_hal_msp.c 中，设置 STM32F4 Discovery 引脚。

```c
void HAL_MspInit(void) {
__HAL_RCC_SYSCFG_CLK_ENABLE();
__HAL_RCC_PWR_CLK_ENABLE();

HAL_PWREx_EnableOverDrive();

/**Configure the Systick interrupt time 
This sets the systick timer interrupt interval to 1ms*/
SysTick_Config(SystemCoreClock / 1000);

__HAL_RCC_GPIOA_CLK_ENABLE();
__HAL_RCC_GPIOB_CLK_ENABLE();
__HAL_RCC_GPIOC_CLK_ENABLE();
__HAL_RCC_GPIOD_CLK_ENABLE();

MX_GPIO_Init();
}

void MX_GPIO_Init(void) {
GPIO_InitTypeDef gpio_init = {0};

gpio_init.Pin = DISCOVERY_LD4_PIN;
gpio_init.Mode = GPIO_MODE_OUTPUT_PP;
gpio_init.Pull = GPIO_PULLUP;
gpio_init.Speed = GPIO_SPEED_FREQ_LOW;
HAL_GPIO_Init(DISCOVERY_LD4_GPIO_PORT, &gpio_init);
}
```

以上代码设置了 STM32F4 Discovery 的 LD4 引脚。

4. 配置 Gap 和Gattc。在源文件 main.c 中，配置 Gap 和 Gattc。

```c
#define MIN_CONN_INTERVAL               MSEC_TO_UNITS(50, UNIT_1_25_MS)
#define MAX_CONN_INTERVAL               MSEC_TO_UNITS(50, UNIT_1_25_MS)
#define SLAVE_LATENCY                   0
#define CONN_SUP_TIMEOUT                MSEC_TO_UNITS(4000, UNIT_10_MS)

static gap_scan_params_t scan_params = {0};
static gattc_write_params_t write_params = {0};

static ble_gap_adv_params_t adv_params = {0};
static ble_gap_conn_params_t conn_params = {0};
static uint16_t conn_handle;
static bool connected = false;
static ble_gattc_handle_range_t rx_handle_range = {{0}};
static int event_data_received = pdFALSE;
static EventGroupHandle_t events = NULL;
static SemaphoreHandle_t sem_data_sent = NULL;

static void gap_evt_handler(uint32_t event, void* p_param);
static void gattc_evt_handler(uint32_t event, void* p_param);
```

以上代码定义了 Gap 和 Gattc 的配置项。

5. 配置事件。在源文件 main.c 中，配置事件。

```c
enum {
EVENT_INITED = 0x00000001,           // System initialized
EVENT_CONNECTING = 0x00000002,       // Connecting to peer
EVENT_CONNECTED = 0x00000004,        // Connected to peer
EVENT_DISCONNECTED = 0x00000008,     // Disconnected from peer
EVENT_DATA_SENT = 0x00000010,        // Data sent
EVENT_DATA_RECEIVED = 0x00000020,    // Data received
};

#define EVENT_ALL ( \
EVENT_INITED | \
EVENT_CONNECTING | \
EVENT_CONNECTED | \
EVENT_DISCONNECTED | \
EVENT_DATA_SENT | \
EVENT_DATA_RECEIVED \
)

static void event_clear(EventBits_t bits) {
BaseType_t higher_priority_task_woken = pdFALSE;
xEventGroupClearBitsFromISR(events, bits, &higher_priority_task_woken);
portYIELD_FROM_ISR(higher_priority_task_woken);
}

static void event_set(EventBits_t bits) {
BaseType_t higher_priority_task_woken = pdFALSE;
xEventGroupSetBitsFromISR(events, bits, &higher_priority_task_woken);
portYIELD_FROM_ISR(higher_priority_task_woken);
}

static uint32_t event_wait(EventBits_t bits) {
TickType_t wait = portMAX_DELAY;
uint32_t result = 0;

while (true) {
result = xEventGroupWaitBits(events, bits, pdFALSE, pdTRUE, wait);

if ((result & bits) || (wait == 0)) {
break;
} else {
wait = tick_after(pdMS_TO_TICKS(100), get_tick());
}
}

return result;
}
```

以上代码定义了事件管理器，用于同步任务。

6. 配置 BLE 回调函数。在源文件 main.c 中，配置 BLE 回调函数。

```c
static ble_gap_conn_sec_mode_t sec_mode;
static volatile uint16_t mtu;

static void gap_evt_handler(uint32_t event, void* p_param) {
switch (event) {
case BLE_GAP_EVT_ADV_REPORT: {
ble_gap_evt_adv_report_t* report = (ble_gap_evt_adv_report_t*) p_param;

if (!memcmp(report->data.p_data, THROTTLE_SERVICE_UUID_128, sizeof(THROTTLE_SERVICE_UUID_128))) {
ble_gap_adv_start(NULL, NULL);
}
}
break;

case BLE_GAP_EVT_AUTH_STATUS: {
ble_gap_evt_auth_status_t* auth = (ble_gap_evt_auth_status_t*) p_param;

if (auth->auth_status == BLE_GAP_AUTH_STATUS_SUCCESS) {
// Stop advertising since we are now successfully authenticated
ble_gap_adv_stop();
printf("Connected!\n");

// Set connection parameters
memset(&conn_params, 0, sizeof(conn_params));
conn_params.min_conn_interval = MIN_CONN_INTERVAL;
conn_params.max_conn_interval = MAX_CONN_INTERVAL;
conn_params.slave_latency = SLAVE_LATENCY;
conn_params.conn_sup_timeout = CONN_SUP_TIMEOUT;

ble_gap_conn_params_update(conn_handle, &conn_params);

// Enable GATT notification and indication
att_svr_authorize_req(conn_handle, &thrift_rx_char.val_handle);
att_svr_read_multiple_req(conn_handle, &thrift_rx_char.val_handle, 0, sizeof(value));
event_set(EVENT_CONNECTED);
connected = true;
} else {
// Stop advertising since there was a failure during authentication
ble_gap_adv_stop();
printf("Authentication failed.\n");
event_set(EVENT_DISCONNECTED);
connected = false;
}
}
break;

case BLE_GAP_EVT_CONNECTED: {
ble_gap_evt_connected_t* connect = (ble_gap_evt_connected_t*) p_param;

sec_mode = (ble_gap_conn_sec_mode_t)connect->conn_sec;
conn_handle = connect->conn_handle;
memcpy(peer_addr, connect->remote_addr.addr, sizeof(peer_addr));

// Update advertisement data
memset(&adv_params, 0, sizeof(adv_params));
adv_params.primary_phy = BLE_GAP_PHY_1MBPS;
adv_params.type = BLE_GAP_ADV_TYPE_ADV_IND;
adv_params.p_peer_addr = &(connect->remote_addr);
adv_params.fp = BLE_GAP_FILTER_ALLOW_SCANNING;
ble_gap_adv_start(&adv_params);

// Read MTU size from remote device
sd_ble_gattc_exchange_mtu(conn_handle, &mtu);
}
break;

case BLE_GAP_EVT_DISCONNECTED: {
ble_gap_evt_disconnected_t* disconnect = (ble_gap_evt_disconnected_t*) p_param;

printf("Disconnected, reason: %d\n", disconnect->reason);
event_set(EVENT_DISCONNECTED);
connected = false;
}
break;

default: 
break;
}
}

static void gattc_evt_handler(uint32_t event, void* p_param) {
switch (event) {
case BLE_GATTC_EVT_EXCHANGE_MTU_RSP: {
ble_gattc_evt_exchange_mtu_rsp_t* exchange_mtu = (ble_gattc_evt_exchange_mtu_rsp_t*) p_param;

if (exchange_mtu->status == BLE_GATT_STATUS_SUCCESS) {
printf("Remote device's MTU size is %d bytes\n", exchange_mtu->client_rx_mtu);
} else {
printf("Failed to read MTU size, status=%d\n", exchange_mtu->status);
}
}
break;

case BLE_GATTC_EVT_HVX: {
ble_gattc_evt_hvx_t* hvx = (ble_gattc_evt_hvx_t*) p_param;

if (hvx->notify_enabled &&!sem_trywait(sem_data_sent)) {
printf("Received notify or indicate message:\n");
print_hex(hvx->data, hvx->len);
event_set(EVENT_DATA_RECEIVED);
xSemaphoreGive(sem_data_sent);
} else {
// Ignore duplicate packets that we already processed
event_clear(EVENT_DATA_RECEIVED);
}
}
break;

default: 
break;
}
}
```

以上代码定义了 BLE 回调函数，用于处理各种事件。

7. 配置任务。在源文件 main.c 中，配置任务。

```c
#define LEAD_TIME                       MSEC_TO_UNITS(30, UNIT_0_625_MS)
#define SCAN_INT                        MSEC_TO_UNITS(100, UNIT_0_625_MS)

static void inited_task(void *argument) {
printf("\nBLE Transmitter started...\n");

// Initialize event group
events = xEventGroupCreate();

// Create semaphore for synchronization between threads
sem_data_sent = xSemaphoreCreateBinary();

// Configure scanning parameters
memset(&scan_params, 0, sizeof(scan_params));
scan_params.active = 1;
scan_params.selective = 0;
scan_params.interval = SCAN_INT;
scan_params.window = SCAN_INT;
scan_params.timeout = 0; // No timeout


// Configure advertising parameters
memset(&adv_params, 0, sizeof(adv_params));
adv_params.primary_phy = BLE_GAP_PHY_1MBPS;
adv_params.type = BLE_GAP_ADV_TYPE_ADV_IND;
adv_params.own_addr_type = BLE_OWN_ADDR_PUBLIC;
adv_params.filter_policy = BLE_GAP_ADV_FP_ANY;
adv_params.duration = 0;
adv_params.pref_period = LEAD_TIME;
ble_gap_adv_params_set(BLE_GAP_ADV_ID_DEFAULT, &adv_params);

// Configure gap callback function
ble_gap_cb_register(gap_evt_handler);

// Configure gattc callback function
ble_gattc_cb_register(gattc_evt_handler);

// Start scanning for devices advertising our custom service UUIDs
ble_gap_scan_start(&scan_params);

while (1) {
vTaskDelay(pdMS_TO_TICKS(500));

// If no data has been sent, try sending some more every few seconds
if (!connected && (get_uptime() > last_send + SECONDS_TO_TICKS(SEND_PERIOD))) {
// Generate random packet payload
gen_rand_payload(value, sizeof(value));

// Send packet via BLE
printf("Sending packet (%d bytes):\n", sizeof(value));
print_hex(value, sizeof(value));
sd_ble_gattc_write(conn_handle, &thrift_tx_char.val_handle, 0, sizeof(value), value, BLE_GATT_EXEC_WRITE_FLAG_PREPARED_WRITE, NULL, NULL);
event_set(EVENT_DATA_SENT);
last_send = get_uptime();
}
}
}
```

以上代码定义了任务，用于初始化，建立连接，处理数据收发。

### 3.1.4 BLE 透传器的实现
#### 3.1.4.1 编写发送函数
在源文件 main.c 中，编写发送函数。

```c
static SemaphoreHandle_t sem_data_sent = NULL;
static int event_data_received = pdFALSE;

// Utility functions
void gen_rand_payload(uint8_t* buf, size_t len) {
for (int i = 0; i < len; ++i) {
buf[i] = rand() % 0xFF;
}
}

void print_hex(const uint8_t* buf, size_t len) {
for (size_t i = 0; i < len; ++i) {
printf("%02X ", buf[i]);
}
puts("");
}

void error_handler(uint32_t nrf_error) {
printf("[NRF ERROR CODE]: 0x%08lX\n", nrf_error);
}

void ble_tx_task(void const * argument) {
uint8_t buffer[DATA_LEN];
int len = sizeof(buffer);

while (1) {
// Wait until we've got data to send
if (xSemaphoreTake(sem_data_sent, portMAX_DELAY) == pdTRUE) {
// Reset data_sent flag
event_clear(EVENT_DATA_SENT);

// Fill up buffer with new data
gen_rand_payload(buffer, len);

// Print out what we're about to transmit
printf("Transmitting packet (%d bytes):\n", len);
print_hex(buffer, len);

// Try to send the data
ble_gattc_write(conn_handle, &thrift_rx_char.val_handle, 0, len, buffer, false, 0);
}
}
}
```

#### 3.1.4.2 配置任务
在源文件 main.c 中，配置新的任务。

```c
TaskHandle_t tasks[] = {
NULL,                                  // Init task not created yet
osThreadCreate(osPriorityHigh, 0, inited_task, "init", NULL),
osThreadCreate(osPriorityHigh, 0, ble_rx_task, "ble_rx", NULL),
osThreadCreate(osPriorityHigh, 0, ble_callback_task, "ble_cb", NULL),
osThreadCreate(osPriorityMedium+1, 0, ble_tx_task, "ble_tx", NULL),
};

tasks[INIT_TASK] = osThreadCreate(osPriorityLow, 1, init_task, "init", NULL);
```

#### 3.1.4.3 测试 BLE 透传器
编译并烧录程序到 STM32F4 Discovery。然后打开串口助手，设置波特率为 115200，同时打开十六进制显示。按下 STM32F4 Discovery 上的 USER 按钮，开始扫描。

如果 STM32F4 Discovery 和手机设备处于同一局域网内，扫描到设备后双击选中设备连接。

如果发现成功连接，手机端会提示连接成功，则说明 BLE 透传器已经正常工作。同时可以看到 STM32F4 Discovery 的 LD4 灯闪烁一次。

手机端可以通过 BLE Scanner 或 LightBlue Explorer 等 APP 来扫码查看设备的特征值，也可以通过手机的 BT 键发送数据给 STM32F4 Discovery。

测试结束后，可以断开手机设备的连接，关闭串口助手，再次按下 STM32F4 Discovery 上的 USER 按钮即可退出程序。