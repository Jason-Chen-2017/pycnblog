                 

### AI 大模型驱动的创业产品设计创新

#### 一、相关领域的典型问题

**1. 什么是AI大模型？它有哪些常见的类型？**

AI大模型（Large-scale AI Models）是指那些具有数十亿甚至万亿参数的深度学习模型。它们通过在大规模数据集上训练来学习复杂的模式和知识。常见的AI大模型类型包括：

- **生成对抗网络（GANs）**：通过生成器和判别器的对抗训练生成数据。
- **变分自编码器（VAEs）**：通过编码器和解码器学习数据的概率分布。
- **Transformer**：广泛应用于自然语言处理领域，通过自注意力机制处理序列数据。

**答案：** AI大模型是一种具有大量参数的深度学习模型，通过在大规模数据集上训练来学习复杂的模式和知识。常见的类型包括生成对抗网络（GANs）、变分自编码器（VAEs）和Transformer等。

**2. 如何设计一个基于AI大模型的创业产品？**

设计基于AI大模型的创业产品需要考虑以下几个方面：

- **市场需求**：分析目标用户的需求，确定AI大模型的应用场景。
- **数据集**：准备或收集足够的数据集，确保模型能够有效训练。
- **模型架构**：选择合适的AI大模型架构，如Transformer、GANs等。
- **训练与优化**：训练模型并优化参数，提高模型性能。
- **产品化**：将模型集成到产品中，提供用户友好的界面和功能。

**答案：** 设计一个基于AI大模型的创业产品需要考虑市场需求、数据集、模型架构、训练与优化以及产品化等方面。具体步骤包括分析目标用户需求、准备或收集数据集、选择模型架构、训练与优化模型以及将模型集成到产品中。

**3. AI大模型在创业产品中的应用有哪些？**

AI大模型在创业产品中的应用非常广泛，以下是一些常见的应用场景：

- **自然语言处理**：文本分类、机器翻译、情感分析等。
- **计算机视觉**：图像分类、目标检测、图像生成等。
- **推荐系统**：个性化推荐、商品推荐等。
- **语音识别**：语音识别、语音合成等。

**答案：** AI大模型在创业产品中的应用包括自然语言处理、计算机视觉、推荐系统和语音识别等领域。例如，在自然语言处理领域，可以应用于文本分类、机器翻译和情感分析等；在计算机视觉领域，可以应用于图像分类、目标检测和图像生成等。

#### 二、算法编程题库

**4. 编写一个基于Transformer的文本分类算法。**

**题目描述：** 编写一个Python程序，使用Transformer模型对文本数据进行分类。要求包括数据处理、模型训练和模型评估。

**答案：** 

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

# 数据处理
def preprocess_data(texts, tokenizer, max_length=128):
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']

# 模型定义
class TextClassifier(nn.Module):
    def __init__(self, num_labels):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

# 训练
def train(model, dataloader, criterion, optimizer, device):
    model.to(device)
    model.train()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs.input_ids, inputs.attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

# 评估
def evaluate(model, dataloader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs.input_ids, inputs.attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 实际使用
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TextClassifier(num_labels=2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 3

for epoch in range(num_epochs):
    train(model, train_dataloader, criterion, optimizer, device)
    val_loss = evaluate(model, val_dataloader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")
```

**5. 编写一个基于GAN的图像生成算法。**

**题目描述：** 编写一个Python程序，使用生成对抗网络（GAN）生成图像。要求包括数据处理、模型定义、训练和图像生成。

**答案：** 

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# 模型定义
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# 训练
generator = Generator()
discriminator = Discriminator()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

num_epochs = 5

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        # 更新D
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size,), 1, device=device)
        optimizer_D.zero_grad()
        outputs = discriminator(real_images)
        loss_D_real = criterion(outputs, labels)
        loss_D_real.backward()

        # 生成假图像
        z = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(z)
        labels.fill_(0)
        optimizer_D.zero_grad()
        outputs = discriminator(fake_images.detach())
        loss_D_fake = criterion(outputs, labels)
        loss_D_fake.backward()
        optimizer_D.step()

        # 更新生成器
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)
        labels.fill_(1)
        loss_G = criterion(outputs, labels)
        loss_G.backward()
        optimizer_G.step()

        # 打印信息
        if i % 100 == 0:
            print(f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}] Loss_D: {loss_D_real + loss_D_fake:.4f}, Loss_G: {loss_G:.4f}")

# 生成图像
z = torch.randn(100, 100, 1, 1, device=device)
with torch.no_grad():
    fake_images = generator(z).cpu()

fake_images = fake_images.view(100, 1, 28, 28)
torch.save(fake_images, 'fake_images.pth')
```

**6. 编写一个基于VAE的图像去噪算法。**

**题目描述：** 编写一个Python程序，使用变分自编码器（VAE）对图像进行去噪。要求包括数据处理、模型定义、训练和图像去噪。

**答案：** 

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# 模型定义
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # 隐含变量均值
        self.fc22 = nn.Linear(400, 20) # 隐含变量方差
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 训练
def train(vae, dataloader, optimizer, criterion, num_epochs=10):
    vae.to('cuda' if torch.cuda.is_available() else 'cpu')
    vae.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            inputs = data[0].to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(inputs)
            loss = criterion(recon_batch, inputs) + 1e-6 * (mu ** 2 + logvar ** 2 - 1).mean()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss: {loss.item()}')

# 实际使用
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train(vae, train_loader, optimizer, criterion)

# 去噪
def denoise_image(image, vae):
    image = torch.FloatTensor(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    _, mu, logvar = vae.encode(image)
    z = vae.reparameterize(mu, logvar)
    recon_image = vae.decode(z)
    return recon_image.cpu().numpy()

# 读取图像
with open('noisy_image.png', 'rb') as f:
    image = Image.open(f)
    image = image.convert('L') # 转为灰度图
    image = np.array(image)

# 去噪
recon_image = denoise_image(image, vae)

# 显示结果
plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(np.reshape(image[i], (28, 28)), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(np.reshape(recon_image[i], (28, 28)), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
```

**7. 编写一个基于深度强化学习的游戏AI算法。**

**题目描述：** 编写一个Python程序，使用深度强化学习（DRL）算法训练一个智能体（agent）玩某个游戏。要求包括环境定义、智能体定义、训练和评估。

**答案：** 

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 环境定义
env = gym.make("CartPole-v0")

# 智能体定义
class DRLAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DRLAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 训练
def train_drl_agent(agent, env, num_episodes, gamma=0.99, alpha=0.001):
    agent.to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(agent.parameters(), lr=alpha)
    running_score = 0.0

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        while not done:
            action = agent(state).argmax()
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32)
            Q_values = agent(state)
            next_Q_values = agent(next_state)
            target_Q = reward + (1 - int(done)) * gamma * next_Q_values.max()
            loss = (Q_values[0][action] - target_Q).pow(2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

        running_score += reward

    print(f"Episode {episode+1}, Average Score: {running_score/num_episodes:.2f}")

# 评估
def evaluate_drl_agent(agent, env, num_episodes):
    scores = []
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        score = 0
        while not done:
            action = agent(state).argmax()
            next_state, reward, done, _ = env.step(action.item())
            score += reward
            state = next_state
        scores.append(score)
    print(f"Average Score over {num_episodes} Episodes: {np.mean(scores):.2f}")

# 实际使用
agent = DRLAgent(input_size=4, hidden_size=64, output_size=2)
train_drl_agent(agent, env, num_episodes=1000)
evaluate_drl_agent(agent, env, num_episodes=10)
```

**8. 编写一个基于迁移学习的文本分类算法。**

**题目描述：** 编写一个Python程序，使用预训练的语言模型进行文本分类。要求包括数据处理、模型加载、文本分类和模型评估。

**答案：** 

```python
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# 数据处理
def load_data():
    # 加载数据集
    # 这里使用一个示例数据集
    texts = ["This is a great product.", "I don't like this movie."]
    labels = [1, 0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, labels

input_ids, attention_mask, labels = load_data()
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, attention_mask, labels, test_size=0.2, random_state=42)

# 模型加载
class TextClassifier(nn.Module):
    def __init__(self, model_name):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits

model_name = 'bert-base-uncased'
model = TextClassifier(model_name)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 训练
train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
optimizer = Adam(model.parameters(), lr=2e-5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        logits = model(inputs, inputs)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            logits = model(inputs, inputs)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {100 * correct / total:.2f}%")
```

**9. 编写一个基于强化学习的推荐系统。**

**题目描述：** 编写一个Python程序，使用强化学习（RL）算法训练一个推荐系统。要求包括用户行为建模、智能体定义、训练和评估。

**答案：** 

```python
import gym
import numpy as np
import random
import torch
from torch import nn
from torch.optim import Adam

# 环境定义
class RecommendationEnv(gym.Env):
    def __init__(self, num_items, discount_factor=0.9):
        super(RecommendationEnv, self).__init__()
        self.num_items = num_items
        self.discount_factor = discount_factor

    def step(self, action):
        reward = 0
        if action == self.current_user_action:
            reward = 1
        next_state = random.randint(0, self.num_items - 1)
        done = True
        return next_state, reward, done, {}

    def reset(self):
        self.current_user_action = random.randint(0, self.num_items - 1)
        return self.current_user_action

# 智能体定义
class RLAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RLAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 训练
def train_recommender(agent, env, num_episodes, gamma=0.99, alpha=0.001):
    agent.to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(agent.parameters(), lr=alpha)
    running_reward = 0.0

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        while not done:
            action_probs = agent(state)
            action = torch.argmax(action_probs).item()
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32)
            Q_values = agent(state)
            target_Q = reward + (1 - int(done)) * gamma * Q_values.max()
            loss = (Q_values[0][action] - target_Q).pow(2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

        running_reward += reward

    print(f"Episode {episode+1}, Average Reward: {running_reward/num_episodes:.2f}")

# 评估
def evaluate_recommender(agent, env, num_episodes):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        score = 0
        while not done:
            action_probs = agent(state)
            action = torch.argmax(action_probs).item()
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
        rewards.append(score)
    print(f"Average Score over {num_episodes} Episodes: {np.mean(rewards):.2f}")

# 实际使用
num_items = 10
env = RecommendationEnv(num_items)
agent = RLAgent(input_size=1, hidden_size=64, output_size=num_items)
train_recommender(agent, env, num_episodes=1000)
evaluate_recommender(agent, env, num_episodes=10)
```

**10. 编写一个基于卷积神经网络的图像分类算法。**

**题目描述：** 编写一个Python程序，使用卷积神经网络（CNN）进行图像分类。要求包括数据处理、模型定义、训练和评估。

**答案：** 

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 数据处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 模型定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 评估
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total:.2f}%")

# 可视化
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

inputs, labels = next(iter(test_loader))
images = inputs[0:10].to('cuda' if torch.cuda.is_available() else 'cpu')
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
for i in range(10):
    imshow torchvision.utils.make_grid(images[i].cpu(), nrow=5, padding=1)
    print(f"Predicted: {predicted[i].item()}, True: {labels[i].item()}")
```

