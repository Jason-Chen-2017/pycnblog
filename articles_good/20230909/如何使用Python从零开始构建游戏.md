
作者：禅与计算机程序设计艺术                    

# 1.简介
  

游戏开发是一个庞大的领域，其涉及到许多领域如数学、计算机图形学、音视频等等，本文只讨论游戏编程，不涉及到其他相关领域知识。

游戏编程可以分成以下几步：

1. 游戏逻辑
2. 渲染
3. 物理引擎
4. AI
5. 用户交互

游戏中可能会用到不同的技术或框架，如：Python（Pygame）、C++（Cocos2d-x）、Java（libGDX）等。

本文将以Pygame作为示例，展示如何利用Python语言从零开始构建一个简单的游戏。

# 2.基本概念术语说明
## 2.1 Pygame概述

Pygame是一款开源的Python游戏编程库，提供了各种游戏引擎组件，比如：窗口管理器、声音效果、图像渲染、用户输入处理等功能，可以帮助游戏开发者快速实现游戏程序。

Pygame最初由Guido van Rossum编写，于2000年发布了第一个版本。它是基于SDL的跨平台框架，可以运行在Windows、Linux、Mac OS X、BSD系统上。

最新版Pygame是1.9.4。

## 2.2 Pygame安装

Pygame支持Python2.7和Python3.X版本。如果您的系统没有安装，请先按照您的操作系统进行安装。

推荐使用Anaconda Python环境安装Pygame，Anaconda是一个包含了众多数据科学、机器学习、深度学习等库的Python发行版。可以非常方便地安装和使用Pygame。

Anaconda安装命令如下：

```bash
conda install -c anaconda pygame
```

如果您还没有安装Anaconda，也可以直接通过pip安装：

```bash
pip install pygame
```

安装完成后，就可以开始编写游戏代码了。

## 2.3 游戏编程中的基本元素

游戏编程中主要有以下几种基本元素：

1. 游戏世界：游戏世界包括地图、角色、怪物等元素构成，这些元素构成了游戏中完整的环境。
2. 游戏机制：游戏机制描述了玩家的游戏方式，如按键控制、鼠标点击、碰撞检测等规则。
3. 游戏画面：游戏画面是指屏幕上显示的图像，包括角色、怪物、建筑、道具等在游戏世界里的动态效果。
4. 播放音效：播放音频有助于增强游戏的 immersion 和 enjoyment。
5. 用户输入：用户可以通过键盘、鼠标、触摸屏等方式与游戏进行交互。
6. 数据存储：保存游戏状态数据，可用于记录游戏得分、存档等信息。
7. AI：AI 可以根据游戏世界和玩家的行为，自动执行一些特定的任务。

# 3.核心算法原理和具体操作步骤以及数学公式讲解

本节主要介绍一些游戏编程中的基本概念。

## 3.1 创建游戏窗口

创建游戏窗口并设置宽高、标题和背景色的代码如下：

```python
import pygame

pygame.init()

width = 800
height = 600
title = "My Game"
bg_color = (255, 255, 255) # white background color

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(title)
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(bg_color)

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.blit(background, (0, 0))

    clock.tick(60)
    pygame.display.flip()
    
pygame.quit()
```

创建游戏窗口需要调用`pygame.display.set_mode()`函数，该函数返回一个Surface对象，用来绘制游戏画面。

然后设置游戏窗口的标题、宽度高度、背景颜色等属性。

## 3.2 定义角色、怪物类

游戏中会存在很多角色、怪物等元素，为了更方便地对这些元素进行管理，可以创建一个类，来封装它们共同拥有的属性和方法。

例如，我们可以定义一个类来表示一个角色：

```python
class Player:
    def __init__(self):
        self.image = None
        self.rect = None
        
    def set_image(self, image_file):
        self.image = pygame.image.load(image_file).convert_alpha()
        self.rect = self.image.get_rect()
        
    def update(self, dx=0, dy=0):
        new_x = self.rect.x + dx
        new_y = self.rect.y + dy
        self.rect = self.rect.move(dx, dy)
        
        if self.is_out_of_bounds():
            self.rect = self.reset_position()
            
    def is_out_of_bounds(self):
        return not (0 <= self.rect.left < width and \
                    0 <= self.rect.top < height and \
                    width >= self.rect.right > 0 and \
                    height >= self.rect.bottom > 0)
        
    def reset_position(self):
        self.rect = self.image.get_rect().move(400, 300)
```

这个Player类有一个构造函数 `__init__`，里面初始化了一个image和一个Rect。其中image通过`pygame.image.load()`加载图片文件并转换为 alpha 模式的 Surface 对象。

还有两个成员方法：

1. `set_image(self, image_file)`：设置image和rect。
2. `update(self, dx=0, dy=0)`：移动 rect 的位置并检查是否越界，越界时重置位置。

创建一个角色的代码如下：

```python
player = Player()
players.add(player)
```

这里我们创建了一个 Player 类的实例 player，并设置他的初始位置为 `(400, 300)` 。

当然，除了角色，游戏世界也会有怪物，所以我们还需要定义怪物类：

```python
class Monster:
    def __init__(self):
        pass
        
    def move(self):
        pass
```

这个Monster类也是有一个构造函数 `__init__`，但是没什么用。

## 3.3 实现游戏循环

游戏循环（Game Loop）是在游戏运行期间持续发生的一系列事件，负责动画绘制、物理计算、AI决策、用户输入等功能。游戏循环的目的是让游戏看起来像是一个真正的程序而不是静态的动画画面。

游戏循环的每一步都由事件驱动，即每次事件结束后才会更新游戏状态。

游戏循环的代码如下：

```python
def game_loop():
    global players
    while running:
        time_passed = clock.tick(60) / 1000.0

        handle_events()

        update_objects()

        draw_frame()

        fps = int(clock.get_fps())
        print("FPS:", fps)

def handle_events():
    global keys_pressed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            keys_pressed[event.key] = True
        elif event.type == pygame.KEYUP:
            keys_pressed[event.key] = False

def update_objects():
    pass

def draw_frame():
    global screen, players, background
    
    screen.blit(background, (0, 0))

    for p in players:
        screen.blit(p.image, p.rect)
    
    pygame.display.flip()
```

游戏循环的各个阶段：

1. 初始化游戏变量：这里包括定义游戏窗口的大小、标题、背景色；创建一个列表 players 来保存所有的角色；准备好一些其他变量。
2. 处理事件：遍历所有事件并判断是否要退出游戏。
3. 更新对象：这里只是简单地对角色进行更新，实际应用时可能还需要更多复杂的更新操作。
4. 绘制帧：调用`draw_frame()`函数，将角色图像绘制到屏幕上。
5. FPS显示：获取当前的FPS值并打印出来。

## 3.4 设置键盘事件

为了让角色可以移动，我们需要监听键盘事件。

键盘事件的处理在`handle_events()`函数内。

首先导入一些必要的模块：

```python
import pygame
from pygame.locals import *
import sys

keys_pressed = {}

def get_key_pressed(key):
    return keys_pressed.get(key, False)

def wait_for_keypress():
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == KEYDOWN or e.type == JOYBUTTONDOWN:
                continue
            else:
                break
```

这里定义了一个字典 keys_pressed 来保存哪些按键被按下过。`get_key_pressed()`函数用来查询某一个按键是否被按下。

当某个按键被按下时，`wait_for_keypress()`函数停止等待，进入下一轮循环。这样，就可以使得角色在接收到键盘输入之后再开始移动。

例子：

```python
if get_key_pressed(K_LEFT) and player.rect.left > 0:
    player.update(-10, 0)

if get_key_pressed(K_RIGHT) and player.rect.right < width:
    player.update(10, 0)

if get_key_pressed(K_UP) and player.rect.top > 0:
    player.update(0, -10)

if get_key_pressed(K_DOWN) and player.rect.bottom < height:
    player.update(0, 10)
```

这里的代码判断左右上下四个方向上的箭头键是否被按下，并且把相应的坐标偏移量传给 player 的 update 方法，让他移动。

## 3.5 实现游戏物理引擎

游戏中的物理引擎可以模拟现实世界中的物体运动。

设想一下，如果没有物理引擎，一个角色只能靠自己的意志力来控制它的移动，这很难满足游戏的需求。

实现游戏物理引擎需要一些数学技巧，比如重力加速度、速度限制、碰撞检测等。

## 3.6 添加碰撞检测

碰撞检测可以检测角色之间的碰撞情况，并作出相应的反应。

实现碰撞检测需要遍历所有角色的 Rect，然后求出他们之间的碰撞区域，最后判断这些区域是否有重叠部分。

实现的方法就是：

1. 获取所有角色的 Rect。
2. 遍历所有角色，对于每个角色 A，遍历所有角色 B，如果 A 和 B 不重叠，则跳过。否则，计算两个角色之间重叠的部分，并判断是否有重合点（即角色是否相撞）。
3. 如果有相撞，就处理相应的事情，比如删除一个角色或者改变两者的位置。

例子：

```python
collision = []

for p in players:
    overlaps = [q for q in players if p!= q and p.rect.colliderect(q.rect)]
    collision.extend([(p, q) for q in overlaps])
    
if len(collision) > 0:
    for pair in collision:
        a, b = pair
        distance_x = abs(a.rect.centerx - b.rect.centerx)
        distance_y = abs(a.rect.centery - b.rect.centery)
        
        if distance_x > a.rect.width/2 + b.rect.width/2 or \
           distance_y > a.rect.height/2 + b.rect.height/2:
               continue
        
        middle_x = (a.rect.centerx + b.rect.centerx)/2
        middle_y = (a.rect.centery + b.rect.centery)/2
        
        angle = math.atan2(middle_y - center_y, middle_x - center_x)
        xspeed = random.uniform(200, 400)*math.cos(angle)
        yspeed = random.uniform(200, 400)*math.sin(angle)
        
        a.xspeed += xspeed
        a.yspeed += yspeed
        
        b.xspeed -= xspeed
        b.yspeed -= yspeed
```

这里代码实现了一个简单的碰撞检测。

首先，遍历所有角色，把它们和其他角色的碰撞情况保存在列表 collision 中。

然后，遍历 collision 中的所有组合，计算它们之间的距离和角度差。

如果距离超过两者半径之和，就认为它们不会相撞。

如果满足条件，就随机产生一个速度向量，并添加到两者的速度上，以模拟重力作用。

## 3.7 添加障碍物

障碍物可以让游戏变得更有趣一些。

实现障碍物需要几个步骤：

1. 定义障碍物的位置和尺寸。
2. 把障碍物的 Rect 添加到一个列表中。
3. 在游戏循环中，绘制所有障碍物。
4. 检测角色是否与障碍物相撞。

例子：

```python
obstacles = [pygame.Rect(100, 100, 100, 100), 
             pygame.Rect(500, 100, 100, 100),
             pygame.Rect(300, 300, 100, 100)]
             
for o in obstacles:
    screen.blit(some_image, o)
        
overlaps = [o for o in obstacles if player.rect.colliderect(o)]
if len(overlaps) > 0:
    # do something...
```

这里定义了一个列表 obstacles 来保存三个矩形障碍物的 Rect。

然后在游戏循环中，绘制所有障碍物。

接着，遍历所有障碍物，查看角色是否与之重叠。如果重叠，就做出相应的反应，比如结束游戏或者倒退。

## 3.8 添加音效

音效可以增加游戏的 immersion 和 enjoyment。

音效的实现比较简单，只需要加载 mp3 文件并播放即可。

```python
sound = pygame.mixer.Sound('sound.mp3')
sound.play()
```

这里加载了一个名为 sound.mp3 的音乐文件，并调用 play() 方法播放。

## 3.9 AI（Artificial Intelligence）

人工智能（Artificial Intelligence，AI）可以让游戏具有更高级的互动性。

目前有很多 AI 算法可以用到游戏编程中，包括决策树、神经网络、遗传算法等。

不过，由于 AI 的性能和复杂度都超出了本文范围，所以不在此展开。

# 4.具体代码实例和解释说明

## 4.1 代码实例

### 4.1.1 空白游戏窗口

```python
import pygame

pygame.init()

width = 800
height = 600
title = "My Game"
bg_color = (255, 255, 255) # white background color

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(title)
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(bg_color)

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # TODO: add more game logic here...

    clock.tick(60)
    pygame.display.flip()
    
pygame.quit()
```

这个代码创建了一个空白游戏窗口，还没有加入任何的游戏逻辑。可以用来测试游戏窗口是否正确显示。

### 4.1.2 角色移动

```python
import pygame
from pygame.locals import *
import sys

keys_pressed = {}

def get_key_pressed(key):
    return keys_pressed.get(key, False)

def wait_for_keypress():
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == KEYDOWN or e.type == JOYBUTTONDOWN:
                continue
            else:
                break

class Player:
    def __init__(self):
        self.image = None
        self.rect = None
        
    def set_image(self, image_file):
        self.image = pygame.image.load(image_file).convert_alpha()
        self.rect = self.image.get_rect()
        
    def update(self, dx=0, dy=0):
        new_x = self.rect.x + dx
        new_y = self.rect.y + dy
        self.rect = self.rect.move(dx, dy)
        
        if self.is_out_of_bounds():
            self.rect = self.reset_position()
            
    def is_out_of_bounds(self):
        return not (0 <= self.rect.left < width and \
                    0 <= self.rect.top < height and \
                    width >= self.rect.right > 0 and \
                    height >= self.rect.bottom > 0)
        
    def reset_position(self):
        self.rect = self.image.get_rect().move(400, 300)

width = 800
height = 600
title = "My Game"
bg_color = (255, 255, 255) # white background color

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(title)
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(bg_color)

clock = pygame.time.Clock()

player = Player()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            
            keys_pressed[event.key] = True
            
    if get_key_pressed(K_LEFT) and player.rect.left > 0:
        player.update(-10, 0)

    if get_key_pressed(K_RIGHT) and player.rect.right < width:
        player.update(10, 0)

    if get_key_pressed(K_UP) and player.rect.top > 0:
        player.update(0, -10)

    if get_key_pressed(K_DOWN) and player.rect.bottom < height:
        player.update(0, 10)

    screen.blit(background, (0, 0))
    screen.blit(player.image, player.rect)

    clock.tick(60)
    pygame.display.flip()
    
pygame.quit()
```

这个代码实现了一个角色的移动功能，可以用方向键来控制角色的前进方向。

同时，代码还实现了按 Esc 键退出游戏的功能。

### 4.1.3 播放音效

```python
import pygame
from pygame.locals import *
import sys

keys_pressed = {}

def get_key_pressed(key):
    return keys_pressed.get(key, False)

def wait_for_keypress():
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == KEYDOWN or e.type == JOYBUTTONDOWN:
                continue
            else:
                break

class Player:
    def __init__(self):
        self.image = None
        self.rect = None
        
    def set_image(self, image_file):
        self.image = pygame.image.load(image_file).convert_alpha()
        self.rect = self.image.get_rect()
        
    def update(self, dx=0, dy=0):
        new_x = self.rect.x + dx
        new_y = self.rect.y + dy
        self.rect = self.rect.move(dx, dy)
        
        if self.is_out_of_bounds():
            self.rect = self.reset_position()
            
    def is_out_of_bounds(self):
        return not (0 <= self.rect.left < width and \
                    0 <= self.rect.top < height and \
                    width >= self.rect.right > 0 and \
                    height >= self.rect.bottom > 0)
        
    def reset_position(self):
        self.rect = self.image.get_rect().move(400, 300)

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.init()
pygame.mixer.init()

sound = pygame.mixer.Sound('sound.mp3')
sound.play()

width = 800
height = 600
title = "My Game"
bg_color = (255, 255, 255) # white background color

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(title)
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(bg_color)

clock = pygame.time.Clock()

player = Player()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            
            keys_pressed[event.key] = True
            
    if get_key_pressed(K_LEFT) and player.rect.left > 0:
        player.update(-10, 0)

    if get_key_pressed(K_RIGHT) and player.rect.right < width:
        player.update(10, 0)

    if get_key_pressed(K_UP) and player.rect.top > 0:
        player.update(0, -10)

    if get_key_pressed(K_DOWN) and player.rect.bottom < height:
        player.update(0, 10)

    screen.blit(background, (0, 0))
    screen.blit(player.image, player.rect)

    clock.tick(60)
    pygame.display.flip()
    
pygame.quit()
```

这个代码演示了如何播放音效，并利用 pre_init 函数预先初始化音频参数。

### 4.1.4 添加障碍物

```python
import pygame
from pygame.locals import *
import sys

keys_pressed = {}

def get_key_pressed(key):
    return keys_pressed.get(key, False)

def wait_for_keypress():
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == KEYDOWN or e.type == JOYBUTTONDOWN:
                continue
            else:
                break

class Player:
    def __init__(self):
        self.image = None
        self.rect = None
        
    def set_image(self, image_file):
        self.image = pygame.image.load(image_file).convert_alpha()
        self.rect = self.image.get_rect()
        
    def update(self, dx=0, dy=0):
        new_x = self.rect.x + dx
        new_y = self.rect.y + dy
        self.rect = self.rect.move(dx, dy)
        
        if self.is_out_of_bounds():
            self.rect = self.reset_position()
            
    def is_out_of_bounds(self):
        return not (0 <= self.rect.left < width and \
                    0 <= self.rect.top < height and \
                    width >= self.rect.right > 0 and \
                    height >= self.rect.bottom > 0)
        
    def reset_position(self):
        self.rect = self.image.get_rect().move(400, 300)

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.init()
pygame.mixer.init()

sound = pygame.mixer.Sound('sound.mp3')
sound.play()

width = 800
height = 600
title = "My Game"
bg_color = (255, 255, 255) # white background color

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(title)
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(bg_color)

clock = pygame.time.Clock()

player = Player()

obstacles = [pygame.Rect(100, 100, 100, 100), 
             pygame.Rect(500, 100, 100, 100),
             pygame.Rect(300, 300, 100, 100)]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            
            keys_pressed[event.key] = True
            
    if get_key_pressed(K_LEFT) and player.rect.left > 0:
        player.update(-10, 0)

    if get_key_pressed(K_RIGHT) and player.rect.right < width:
        player.update(10, 0)

    if get_key_pressed(K_UP) and player.rect.top > 0:
        player.update(0, -10)

    if get_key_pressed(K_DOWN) and player.rect.bottom < height:
        player.update(0, 10)

    for o in obstacles:
        screen.blit(some_image, o)

    overlaps = [o for o in obstacles if player.rect.colliderect(o)]
    if len(overlaps) > 0:
        running = False

    screen.blit(background, (0, 0))
    screen.blit(player.image, player.rect)

    clock.tick(60)
    pygame.display.flip()
    
pygame.quit()
```

这个代码演示了如何实现障碍物。

 obstacle 的绘制代码可以在游戏循环的 draw_frame() 函数中找到。

游戏循环中的 overlaps 代码可以检测角色是否与障碍物相撞，如果发生这种情况，游戏循环就会结束。

### 4.1.5 添加碰撞检测

```python
import pygame
from pygame.locals import *
import sys

keys_pressed = {}

def get_key_pressed(key):
    return keys_pressed.get(key, False)

def wait_for_keypress():
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == KEYDOWN or e.type == JOYBUTTONDOWN:
                continue
            else:
                break

class Player:
    def __init__(self):
        self.image = None
        self.rect = None
        
    def set_image(self, image_file):
        self.image = pygame.image.load(image_file).convert_alpha()
        self.rect = self.image.get_rect()
        
    def update(self, dx=0, dy=0):
        new_x = self.rect.x + dx
        new_y = self.rect.y + dy
        self.rect = self.rect.move(dx, dy)
        
        if self.is_out_of_bounds():
            self.rect = self.reset_position()
            
    def is_out_of_bounds(self):
        return not (0 <= self.rect.left < width and \
                    0 <= self.rect.top < height and \
                    width >= self.rect.right > 0 and \
                    height >= self.rect.bottom > 0)
        
    def reset_position(self):
        self.rect = self.image.get_rect().move(400, 300)

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.init()
pygame.mixer.init()

sound = pygame.mixer.Sound('sound.mp3')
sound.play()

width = 800
height = 600
title = "My Game"
bg_color = (255, 255, 255) # white background color

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(title)
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill(bg_color)

clock = pygame.time.Clock()

player = Player()

obstacles = [pygame.Rect(100, 100, 100, 100), 
             pygame.Rect(500, 100, 100, 100),
             pygame.Rect(300, 300, 100, 100)]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            
            keys_pressed[event.key] = True
            
    if get_key_pressed(K_LEFT) and player.rect.left > 0:
        player.update(-10, 0)

    if get_key_pressed(K_RIGHT) and player.rect.right < width:
        player.update(10, 0)

    if get_key_pressed(K_UP) and player.rect.top > 0:
        player.update(0, -10)

    if get_key_pressed(K_DOWN) and player.rect.bottom < height:
        player.update(0, 10)

    collision = []

    for p in players:
        overlaps = [q for q in players if p!= q and p.rect.colliderect(q.rect)]
        collision.extend([(p, q) for q in overlaps])
        
    if len(collision) > 0:
        for pair in collision:
            a, b = pair
            distance_x = abs(a.rect.centerx - b.rect.centerx)
            distance_y = abs(a.rect.centery - b.rect.centery)
            
            if distance_x > a.rect.width/2 + b.rect.width/2 or \
               distance_y > a.rect.height/2 + b.rect.height/2:
                   continue
            
            middle_x = (a.rect.centerx + b.rect.centerx)/2
            middle_y = (a.rect.centery + b.rect.centery)/2
            
            angle = math.atan2(middle_y - center_y, middle_x - center_x)
            xspeed = random.uniform(200, 400)*math.cos(angle)
            yspeed = random.uniform(200, 400)*math.sin(angle)
            
            a.xspeed += xspeed
            a.yspeed += yspeed
            
            b.xspeed -= xspeed
            b.yspeed -= yspeed

    for o in obstacles:
        screen.blit(some_image, o)

    overlaps = [o for o in obstacles if player.rect.colliderect(o)]
    if len(overlaps) > 0:
        running = False

    screen.blit(background, (0, 0))
    screen.blit(player.image, player.rect)

    clock.tick(60)
    pygame.display.flip()
    
pygame.quit()
```

这个代码演示了如何实现碰撞检测。

碰撞检测代码在游戏循环的 update_objects() 函数中。

碰撞检测会根据两个角色的 Rect 判断是否发生了碰撞。如果发生了碰撞，会修改两个角色的速度方向，以模拟物理的相撞。