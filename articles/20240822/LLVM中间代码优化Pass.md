                 

# LLVM中间代码优化Pass

> 关键词：LLVM,中间代码,优化Pass,代码生成,编译器优化

## 1. 背景介绍

### 1.1 问题由来
随着程序规模的不断增大，编译器对代码的优化变得愈发重要。传统的编译器优化技术主要集中在源代码级别，但源代码级别的优化往往需要复杂的语法分析，而且优化效果受源代码质量的影响较大。相比之下， LLVM (LLVM) 中间代码优化Pass则可以以一种更通用的方式，对源代码进行全局优化，从而在保证代码质量和可读性的同时，提升执行效率。

### 1.2 问题核心关键点
LLVM (LLVM) 是当今最流行的编译器基础设施之一，广泛应用于开源、商业和工业领域。LLVM 的核心在于其IR（中间表示），IR 提供了一种抽象层次，使得优化Pass能够更高效地进行操作。LLVM IR 的优化Pass 主要包括如下几类：

- **指令级优化Pass**：通过各种指令重排和组合技术，提升性能。
- **控制流优化Pass**：通过优化控制流结构，减少分支和循环开销。
- **内存优化Pass**：通过优化内存访问和数据布局，减少内存使用和数据移动。
- **并行化优化Pass**：通过并行执行技术，提升执行速度。
- **代码生成优化Pass**：通过生成高效的机器码，提升执行效率。

这些Pass 为开发者提供了强大的工具，以提升代码执行效率，同时保持源代码的可读性和可维护性。

## 2. 核心概念与联系

### 2.1 核心概念概述

为更好地理解LLVM中间代码优化Pass，本节将介绍几个密切相关的核心概念：

- **LLVM (LLVM)**：由Mozilla开发的高级编译器工具，提供了一套完整、高效的编译器基础设施。
- **中间表示 (IR)**：LLVM 的IR 提供了一种抽象层次，用于描述程序的中间状态，使得优化Pass能够更高效地进行操作。
- **优化Pass (Optimization Passes)**：LLVM 中的优化Pass 是一种可以运行在IR 上的程序，用于优化代码结构，提升执行效率。
- **代码生成 (Code Generation)**：将优化后的IR 转换为可执行的机器码的过程。
- **控制流 (Control Flow)**：程序中的分支、循环等结构。
- **指令 (Instructions)**：LLVM IR 中的基本操作单元。

这些核心概念之间的逻辑关系可以通过以下Mermaid流程图来展示：

```mermaid
graph TB
    A[LLVM] --> B[中间表示 (IR)]
    A --> C[优化Pass]
    B --> D[指令]
    D --> E[控制流]
    C --> F[代码生成]
    F --> G[可执行机器码]
```

这个流程图展示了一个典型的LLVM IR 优化流程：

1. LLVM 将源代码编译成IR。
2. 通过优化Pass 对IR 进行优化。
3. 优化后的IR 生成指令。
4. 指令组成控制流结构。
5. 最终生成可执行的机器码。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

LLVM中间代码优化Pass主要通过分析程序的结构和数据流，识别出潜在的优化机会。优化Pass 可以分类为如下几类：

- **基于控制流的优化Pass**：例如loop unrolling (循环展开)，switch/if 替换等。
- **基于数据的优化Pass**：例如死代码消除 (DCE)，常量折叠 (constant folding)，循环展开 (loop unrolling) 等。
- **基于指令的优化Pass**：例如CSE (循环不变代码提取)，合并寄存器 (Register Combining)，替换为更优指令等。

每个优化Pass 都会分析程序的结构和数据流，通过变换指令、控制流或数据结构来提升性能。

### 3.2 算法步骤详解

LLVM中间代码优化Pass的具体操作流程如下：

**Step 1: 构建IR**

将源代码转换为LLVM IR。这个过程称为前端编译，使用编译器将源代码转换为IR，并生成中间表示。

**Step 2: 插入优化Pass**

将各个优化Pass 按照一定顺序插入到IR 中，按照顺序依次执行。

**Step 3: 运行优化Pass**

每个优化Pass 都会分析IR，并进行一系列的变换操作，如指令重排、控制流优化、死代码消除等。

**Step 4: 生成代码**

优化后的IR 通过后端编译器生成可执行的机器码，并进行代码优化和并行化处理。

**Step 5: 输出可执行文件**

将生成的机器码输出为可执行文件，供用户使用。

### 3.3 算法优缺点

LLVM中间代码优化Pass 具有以下优点：

1. **优化效果显著**：通过分析IR，可以全局性优化代码结构，提升执行效率。
2. **灵活性高**：可以按照需求选择和组合不同的优化Pass，实现更精细的优化。
3. **可移植性强**：优化Pass 可以在多种平台上运行，支持跨语言和跨平台的优化。
4. **效率高**：通过IR 分析，可以在编译时进行优化，减少运行时开销。

同时，这些Pass 也存在一些缺点：

1. **复杂度高**：IR 分析的复杂度较高，需要一定的计算资源。
2. **优化成本高**：需要编写和维护大量的优化Pass，开发成本较高。
3. **代码可读性差**：优化后的IR 可能变得难以阅读和理解。
4. **依赖性大**：优化Pass 的效率依赖于IR 的结构和数据流分析的准确性。

尽管存在这些局限性，但就目前而言，LLVM中间代码优化Pass 仍然是提升代码执行效率的重要手段。未来相关研究的重点在于如何进一步降低优化成本，提高优化效果，同时兼顾代码可读性和可维护性。

### 3.4 算法应用领域

LLVM中间代码优化Pass 广泛应用于以下几个领域：

- **高性能计算**：通过优化Pass，提升计算密集型代码的执行效率。
- **嵌入式系统**：优化Pass 可以帮助嵌入式设备在资源受限的环境下运行高效代码。
- **游戏引擎**：优化Pass 可以提升游戏的渲染和执行效率，改善游戏体验。
- **云服务器**：通过优化Pass，提升云服务器的资源利用率和性能。
- **物联网设备**：优化Pass 可以帮助物联网设备高效运行，提升用户体验。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

本节将使用数学语言对LLVM中间代码优化Pass 进行更加严格的刻画。

记LLVM IR 为 $I = \{I_1, I_2, ..., I_n\}$，其中 $I_i$ 表示第 $i$ 条指令。记优化Pass 为 $P = \{P_1, P_2, ..., P_m\}$，其中 $P_j$ 表示第 $j$ 个优化Pass。优化Pass 的输出为 $I' = P(I)$。

定义优化Pass $P_j$ 的效果函数为 $f_j: I \rightarrow I$，表示在指令 $I$ 上应用优化Pass $P_j$ 后，生成的优化后的指令。则整个优化Pass 的效果函数为：

$$
f_P = \prod_{j=1}^m f_j
$$

即所有优化Pass 的效果函数为所有Pass 效果的组合。

### 4.2 公式推导过程

以下我们以循环展开 (Loop Unrolling) 为例，推导循环优化Pass 的效果。

假设循环的IR 如下：

```plaintext
%0 = call i32 @foo()
%1 = icmp slt %0, 1000000
br i1 %1, label %L2, label %L3
%L2:
  %2 = load i32, ptr %0, align 4
  %3 = icmp slt %2, 1000000
  br i1 %3, label %L2, label %L3
%L3:
  %4 = call i32 @bar()
  store i32 %4, ptr %0, align 4
  br label %L1
```

循环展开后的IR 如下：

```plaintext
%0 = call i32 @foo()
%1 = icmp slt %0, 1000000
br i1 %1, label %L2, label %L3
%L2:
  %2 = load i32, ptr %0, align 4
  %3 = icmp slt %2, 1000000
  br i1 %3, label %L2, label %L3
%L3:
  %4 = call i32 @bar()
  store i32 %4, ptr %0, align 4
  br label %L1

%5 = call i32 @foo()
%6 = icmp slt %5, 1000000
br i1 %6, label %L4, label %L5
%L4:
  %7 = load i32, ptr %5, align 4
  %8 = icmp slt %7, 1000000
  br i1 %8, label %L4, label %L5
%L5:
  %9 = call i32 @bar()
  store i32 %9, ptr %5, align 4
  br label %L1

%10 = call i32 @foo()
%11 = icmp slt %10, 1000000
br i1 %11, label %L6, label %L7
%L6:
  %12 = load i32, ptr %10, align 4
  %13 = icmp slt %12, 1000000
  br i1 %13, label %L6, label %L7
%L7:
  %14 = call i32 @bar()
  store i32 %14, ptr %10, align 4
  br label %L1

%15 = call i32 @foo()
%16 = icmp slt %15, 1000000
br i1 %16, label %L8, label %L9
%L8:
  %17 = load i32, ptr %15, align 4
  %18 = icmp slt %17, 1000000
  br i1 %18, label %L8, label %L9
%L9:
  %19 = call i32 @bar()
  store i32 %19, ptr %15, align 4
  br label %L1

%20 = call i32 @foo()
%21 = icmp slt %20, 1000000
br i1 %21, label %L10, label %L11
%L10:
  %22 = load i32, ptr %20, align 4
  %23 = icmp slt %22, 1000000
  br i1 %23, label %L10, label %L11
%L11:
  %24 = call i32 @bar()
  store i32 %24, ptr %20, align 4
  br label %L1
```

可以看出，通过循环展开，循环体被重复执行多次，从而减少了循环内的分支和控制流开销，提升了循环的执行效率。

### 4.3 案例分析与讲解

以Switch/If 替换 (Switch/If Replacement) 为例，分析如何通过优化Pass 优化Switch/If 结构。

假设Switch/If 结构的IR 如下：

```plaintext
%0 = load i32, ptr %arg0
%1 = switch i32 %0, { [0 -> label %L2], [1 -> label %L3], [2 -> label %L4], [3 -> label %L5], [4 -> label %L6], [5 -> label %L7], [6 -> label %L8], [7 -> label %L9], [8 -> label %L10], [9 -> label %L11], [10 -> label %L12], [11 -> label %L13], [12 -> label %L14], [13 -> label %L15], [14 -> label %L16], [15 -> label %L17], [16 -> label %L18], [17 -> label %L19], [18 -> label %L20], [19 -> label %L21], [20 -> label %L22], [21 -> label %L23], [22 -> label %L24], [23 -> label %L25], [24 -> label %L26], [25 -> label %L27], [26 -> label %L28], [27 -> label %L29], [28 -> label %L30], [29 -> label %L31], [30 -> label %L32], [31 -> label %L33], [32 -> label %L34], [33 -> label %L35], [34 -> label %L36], [35 -> label %L37], [36 -> label %L38], [37 -> label %L39], [38 -> label %L40], [39 -> label %L41], [40 -> label %L42], [41 -> label %L43], [42 -> label %L44], [43 -> label %L45], [44 -> label %L46], [45 -> label %L47], [46 -> label %L48], [47 -> label %L49], [48 -> label %L50], [49 -> label %L51], [50 -> label %L52], [51 -> label %L53], [52 -> label %L54], [53 -> label %L55], [54 -> label %L56], [55 -> label %L57], [56 -> label %L58], [57 -> label %L59], [58 -> label %L60], [59 -> label %L61], [60 -> label %L62], [61 -> label %L63], [62 -> label %L64], [63 -> label %L65], [64 -> label %L66], [65 -> label %L67], [66 -> label %L68], [67 -> label %L69], [68 -> label %L70], [69 -> label %L71], [70 -> label %L72], [71 -> label %L73], [72 -> label %L74], [73 -> label %L75], [74 -> label %L76], [75 -> label %L77], [76 -> label %L78], [77 -> label %L79], [78 -> label %L80], [79 -> label %L81], [80 -> label %L82], [81 -> label %L83], [82 -> label %L84], [83 -> label %L85], [84 -> label %L86], [85 -> label %L87], [86 -> label %L88], [87 -> label %L89], [88 -> label %L90], [89 -> label %L91], [90 -> label %L92], [91 -> label %L93], [92 -> label %L94], [93 -> label %L95], [94 -> label %L96], [95 -> label %L97], [96 -> label %L98], [97 -> label %L99], [98 -> label %L100], [99 -> label %L101], [100 -> label %L102], [101 -> label %L103], [102 -> label %L104], [103 -> label %L105], [104 -> label %L106], [105 -> label %L107], [106 -> label %L108], [107 -> label %L109], [108 -> label %L110], [109 -> label %L111], [110 -> label %L112], [111 -> label %L113], [112 -> label %L114], [113 -> label %L115], [114 -> label %L116], [115 -> label %L117], [116 -> label %L118], [117 -> label %L119], [118 -> label %L120], [119 -> label %L121], [120 -> label %L122], [121 -> label %L123], [122 -> label %L124], [123 -> label %L125], [124 -> label %L126], [125 -> label %L127], [126 -> label %L128], [127 -> label %L129], [128 -> label %L130], [129 -> label %L131], [130 -> label %L132], [131 -> label %L133], [132 -> label %L134], [133 -> label %L135], [134 -> label %L136], [135 -> label %L137], [136 -> label %L138], [137 -> label %L139], [138 -> label %L140], [139 -> label %L141], [140 -> label %L142], [141 -> label %L143], [142 -> label %L144], [143 -> label %L145], [144 -> label %L146], [145 -> label %L147], [146 -> label %L148], [147 -> label %L149], [148 -> label %L150], [149 -> label %L151], [150 -> label %L152], [151 -> label %L153], [152 -> label %L154], [153 -> label %L155], [154 -> label %L156], [155 -> label %L157], [156 -> label %L158], [157 -> label %L159], [158 -> label %L160], [159 -> label %L161], [160 -> label %L162], [161 -> label %L163], [162 -> label %L164], [163 -> label %L165], [164 -> label %L166], [165 -> label %L167], [166 -> label %L168], [167 -> label %L169], [168 -> label %L170], [169 -> label %L171], [170 -> label %L172], [171 -> label %L173], [172 -> label %L174], [173 -> label %L175], [174 -> label %L176], [175 -> label %L177], [176 -> label %L178], [177 -> label %L179], [178 -> label %L180], [179 -> label %L181], [180 -> label %L182], [181 -> label %L183], [182 -> label %L184], [183 -> label %L185], [184 -> label %L186], [185 -> label %L187], [186 -> label %L188], [187 -> label %L189], [188 -> label %L190], [189 -> label %L191], [190 -> label %L192], [191 -> label %L193], [192 -> label %L194], [193 -> label %L195], [194 -> label %L196], [195 -> label %L197], [196 -> label %L198], [197 -> label %L199], [198 -> label %L200], [199 -> label %L201], [200 -> label %L202], [201 -> label %L203], [202 -> label %L204], [203 -> label %L205], [204 -> label %L206], [205 -> label %L207], [206 -> label %L208], [207 -> label %L209], [208 -> label %L210], [209 -> label %L211], [210 -> label %L212], [211 -> label %L213], [212 -> label %L214], [213 -> label %L215], [214 -> label %L216], [215 -> label %L217], [216 -> label %L218], [217 -> label %L219], [218 -> label %L220], [219 -> label %L221], [220 -> label %L222], [221 -> label %L223], [222 -> label %L224], [223 -> label %L225], [224 -> label %L226], [225 -> label %L227], [226 -> label %L228], [227 -> label %L229], [228 -> label %L230], [229 -> label %L231], [230 -> label %L232], [231 -> label %L233], [232 -> label %L234], [233 -> label %L235], [234 -> label %L236], [235 -> label %L237], [236 -> label %L238], [237 -> label %L239], [238 -> label %L240], [239 -> label %L241], [240 -> label %L242], [241 -> label %L243], [242 -> label %L244], [243 -> label %L245], [244 -> label %L246], [245 -> label %L247], [246 -> label %L248], [247 -> label %L249], [248 -> label %L250], [249 -> label %L251], [250 -> label %L252], [251 -> label %L253], [252 -> label %L254], [253 -> label %L255], [254 -> label %L256], [255 -> label %L257], [256 -> label %L258], [257 -> label %L259], [258 -> label %L260], [259 -> label %L261], [260 -> label %L262], [261 -> label %L263], [262 -> label %L264], [263 -> label %L265], [264 -> label %L266], [265 -> label %L267], [266 -> label %L268], [267 -> label %L269], [268 -> label %L270], [269 -> label %L271], [270 -> label %L272], [271 -> label %L273], [272 -> label %L274], [273 -> label %L275], [274 -> label %L276], [275 -> label %L277], [276 -> label %L278], [277 -> label %L279], [278 -> label %L280], [279 -> label %L281], [280 -> label %L282], [281 -> label %L283], [282 -> label %L284], [283 -> label %L285], [284 -> label %L286], [285 -> label %L287], [286 -> label %L288], [287 -> label %L289], [288 -> label %L290], [289 -> label %L291], [290 -> label %L292], [291 -> label %L293], [292 -> label %L294], [293 -> label %L295], [294 -> label %L296], [295 -> label %L297], [296 -> label %L298], [297 -> label %L299], [298 -> label %L300], [299 -> label %L301], [300 -> label %L302], [301 -> label %L303], [302 -> label %L304], [303 -> label %L305], [304 -> label %L306], [305 -> label %L307], [306 -> label %L308], [307 -> label %L309], [308 -> label %L310], [309 -> label %L311], [310 -> label %L312], [311 -> label %L313], [312 -> label %L314], [313 -> label %L315], [314 -> label %L316], [315 -> label %L317], [316 -> label %L318], [317 -> label %L319], [318 -> label %L320], [319 -> label %L321], [320 -> label %L322], [321 -> label %L323], [322 -> label %L324], [323 -> label %L325], [324 -> label %L326], [325 -> label %L327], [326 -> label %L328], [327 -> label %L329], [328 -> label %L330], [329 -> label %L331], [330 -> label %L332], [331 -> label %L333], [332 -> label %L333], [333 -> label %L334], [334 -> label %L335], [335 -> label %L336], [336 -> label %L337], [337 -> label %L338], [338 -> label %L339], [339 -> label %L340], [340 -> label %L341], [341 -> label %L342], [342 -> label %L343], [343 -> label %L344], [344 -> label %L345], [345 -> label %L346], [346 -> label %L347], [347 -> label %L348], [348 -> label %L349], [349 -> label %L350], [350 -> label %L351], [351 -> label %L352], [352 -> label %L353], [353 -> label %L354], [354 -> label %L355], [355 -> label %L356], [356 -> label %L357], [357 -> label %L358], [358 -> label %L359], [359 -> label %L360], [360 -> label %L361], [361 -> label %L362], [362 -> label %L363], [363 -> label %L364], [364 -> label %L365], [365 -> label %L366], [366 -> label %L367], [367 -> label %L368], [368 -> label %L369], [369 -> label %L370], [370 -> label %L371], [371 -> label %L372], [372 -> label %L373], [373 -> label %L374], [374 -> label %L375], [375 -> label %L376], [376 -> label %L377], [377 -> label %L378], [378 -> label %L379], [379 -> label %L380], [380 -> label %L381], [381 -> label %L382], [382 -> label %L383], [383 -> label %L384], [384 -> label %L385], [385 -> label %L386], [386 -> label %L387], [387 -> label %L388], [388 -> label %L389], [389 -> label %L390], [390 -> label %L391], [391 -> label %L392], [392 -> label %L393], [393 -> label %L394], [394 -> label %L395], [395 -> label %L396], [396 -> label %L397], [397 -> label %L398], [398 -> label %L399], [399 -> label %L400], [400 -> label %L401], [401 -> label %L402], [402 -> label %L403], [403 -> label %L404], [404 -> label %L405], [405 -> label %L406], [406 -> label %L407], [407 -> label %L408], [408 -> label %L409], [409 -> label %L410], [410 -> label %L411], [411 -> label %L412], [412 -> label %L413], [413 -> label %L414], [414 -> label %L415], [415 -> label %L416], [416 -> label %L417], [417 -> label %L418], [418 -> label %L419], [419 -> label %L420], [420 -> label %L421], [421 -> label %L422], [422 -> label %L423], [423 -> label %L424], [424 -> label %L425], [425 -> label %L426], [426 -> label %L427], [427 -> label %L428], [428 -> label %L429], [429 -> label %L430], [430 -> label %L431], [431 -> label %L432], [432 -> label %L433], [433 -> label %L434], [434 -> label %L435], [435 -> label %L436], [436 -> label %L437], [437 -> label %L438], [438 -> label %L439], [439 -> label %L440], [440 -> label %L441], [441 -> label %L442], [442 -> label %L443], [443 -> label %L444], [444 -> label %L445], [445 -> label %L446], [446 -> label %L447], [447 -> label %L448], [448 -> label %L449], [449 -> label %L450], [450 -> label %L451], [451 -> label %L452], [452 -> label %L453], [453 -> label %L454], [454 -> label %L455], [455 -> label %L456], [456 -> label %L457], [457 -> label %L458], [458 -> label %L459], [459 -> label %L460], [460 -> label %L461], [461 -> label %L462], [462 -> label %L463], [463 -> label %L464], [464 -> label %L465], [465 -> label %L466], [466 -> label %L467], [467 -> label %L468], [468 -> label %L469], [469 -> label %L470], [470 -> label %L471], [471 -> label %L472], [472 -> label %L473], [473 -> label %L474], [474 -> label %L475], [475 -> label %L476], [476 -> label %L477], [477 -> label %L478], [478 -> label %L479], [479 -> label %L480], [480 -> label %L481], [481 -> label %L482], [482 -> label %L483], [483 -> label %L484], [484 -> label %L485], [485 -> label %L486], [486 -> label %L487], [487 -> label %L488], [488 -> label %L489], [489 -> label %L490], [490 -> label %L491], [491 -> label %L492], [492 -> label %L493], [493 -> label %L494], [494 -> label %L495], [495 -> label %L496], [496 -> label %L497], [497 -> label %L498], [498 -> label %L499], [499 -> label %L500], [500 -> label %L501], [501 -> label %L502], [502 -> label %L503], [503 -> label %L504], [504 -> label %L505], [505 -> label %L506], [506 -> label %L507], [507 -> label %L508], [508 -> label %L509], [509 -> label %L510], [510 -> label %L511], [511 -> label %L512], [512 -> label %L513], [513 -> label %L514], [514 -> label %L515], [515 -> label %L516], [516 -> label %L517], [517 -> label %L518], [518 -> label %L519], [519 -> label %L520], [520 -> label %L521], [521 -> label %L522], [522 -> label %L523], [523 -> label %L524], [524 -> label %L525], [525 -> label %L526], [526 -> label %L527], [527 -> label %L528], [528 -> label %L529], [529 -> label %L530], [530 -> label %L531], [531 -> label %L532], [532 -> label %L533], [533 -> label %L534], [534 -> label %L535], [535 -> label %L536], [536 -> label %L537], [537 -> label %L538], [538 -> label %L539], [539 -> label %L540], [540 -> label %L541], [541 -> label %L542], [542 -> label %L543], [543 -> label %L544], [544 -> label %L545], [545 -> label %L546], [546 -> label %L547], [547 -> label %L548], [548 -> label %L549], [549 -> label %L550], [550 -> label %L551], [551 -> label %L552], [552 -> label %L553], [553 -> label %L554], [554 -> label %L555], [555 -> label %L556], [556 -> label %L557], [557 -> label %L558], [558 -> label %L559], [559 -> label %L560], [560 -> label %L561], [561 -> label %L562], [562 -> label %L563], [563 -> label %L564], [564 -> label %L565], [565 -> label %L566], [566 -> label %L567], [567 -> label %L568], [568 -> label %L569], [569 -> label %L570], [570 -> label %L571], [571 -> label %L572], [572 -> label %L573], [573 -> label %L574], [574 -> label %L575], [575 -> label %L576], [576 -> label %L577], [577 -> label %L578], [578 -> label %L579], [579 -> label %L580], [580 -> label %L581], [581 -> label %L582], [582 -> label %L583], [583 -> label %L584], [584 -> label %L585], [585 -> label %L586], [586 -> label %L587], [587 -> label %L588], [588 -> label %L589], [589 -> label %L590], [590 -> label %L591], [591 -> label %L592], [592 -> label %L593], [593 -> label %L594], [594 -> label %L595], [595 -> label %L596], [596 -> label %L597], [597 -> label %L598], [598 -> label %L599], [599 -> label %L600], [600 -> label %L601], [601 -> label %L602], [602 -> label %L603], [603 -> label %L604], [604 -> label %L605], [605 -> label %L606], [606 -> label %L607], [607 -> label %L608], [608 -> label %L609], [609 -> label %L610], [610 -> label %L611], [611 -> label %L612], [612 -> label %L613], [613 -> label %L614], [614 -> label %L615], [615 -> label %L616], [616 -> label %L617], [617 -> label %L618], [618 -> label %L619], [619 -> label %L620], [620 -> label %L621], [621 -> label %L622], [622 -> label %L623], [623 -> label %L624], [624 -> label %L625], [625 -> label %L626], [626 -> label %L627], [627 -> label %L628], [628 -> label %L629], [629 -> label %L630], [630 -> label %L631], [631 -> label %L632], [632 -> label %L633], [633 -> label %L634], [634 -> label %L635], [635 -> label %L636], [636 -> label %L637], [637 -> label %L638], [638 -> label %L639], [639 -> label %L640], [640 -> label %L641], [641 -> label %L642], [642 -> label %L643], [643 -> label %L644], [644 -> label %L645], [645 -> label %L646], [646 -> label %L647], [647 -> label %L648], [648 -> label %L649], [649 -> label %L650], [650 -> label %L651], [651 -> label %L652], [652 -> label %L653], [653 -> label %L654], [654 -> label %L655], [655 -> label %L656], [656 -> label %L657], [657 -> label %L658], [658 -> label %L659], [659 -> label %L660], [660 -> label %L661], [661 -> label %L662], [662 -> label %L663], [663 -> label %L664], [664 -> label %L665], [665 -> label %L666], [666 -> label %L667], [667 -> label %L668], [668 -> label %L669], [669 -> label %L670], [670 -> label %L671], [671 -> label %L672], [672 -> label %L673], [673 -> label %L674], [674 -> label %L675], [675 -> label %L676], [676 -> label %L677], [677 -> label %L678], [678 -> label %L679], [679 -> label %L680], [680 -> label %L681], [681 -> label %L682], [682 -> label %L683], [683 -> label %L684], [684 -> label %L685], [685 -> label %L686], [686 -> label %L687], [687 -> label %L688], [688 -> label %L689], [689 -> label %L690], [690 -> label %L691], [691 -> label %L692], [692 -> label %L693], [693 -> label %L694], [694 -> label %L695], [695 -> label %L696], [696 -> label %L697], [697 -> label %L698], [698 -> label %L699], [699 -> label %L700], [700 -> label %L701], [701 -> label %L702], [702 -> label %L703], [703 -> label %L704], [704 -> label %L705], [705 -> label %L706], [706 -> label %L707], [707 -> label %L708], [708 -> label %L709], [709 -> label %L710], [710 -> label %L711], [711 -> label %L712], [712 -> label %L713], [713 -> label %L714], [714 -> label %L715], [715 -> label %L716], [716 -> label %L717], [717 -> label %L718], [718 -> label %L719], [719 -> label %L720], [720 -> label %L721], [721 -> label %L722], [722 -> label %L723], [723 -> label %L724], [724 -> label %L725], [725 -> label %L726], [726 -> label %L727], [727 -> label %L728], [728 -> label %L729], [729 -> label %L730], [730 -> label %L731], [731 -> label %L732], [732 -> label %L733], [733 -> label %L734], [734 -> label %L735], [735 -> label %L736], [736 -> label %L737], [737 -> label %L738], [738 -> label %L739], [739 -> label %L740], [740 -> label %L741], [741 -> label %L742], [742 -> label %L743], [743 -> label %L744], [744 -> label %L745], [745 -> label %L746], [746 -> label %L747], [747 -> label %L748], [748 -> label %L749], [749 -> label %L750], [750 -> label %L751], [751 -> label %L752], [752 -> label %L753], [753 -> label %L754], [754 -> label %L755], [755 -> label %L756], [756 -> label %L757], [757 -> label %L758], [758 -> label %L759], [759 -> label %L760], [760 -> label %L761], [761 -> label %L762], [762 -> label %L763], [763 -> label %L764], [764 -> label %L765], [765 -> label %L766], [766 -> label %L767], [767 -> label %L768], [768 -> label %L769], [769 -> label %L770], [770 -> label %L771], [771 -> label %L772], [772 -> label %L773], [773 -> label %L774], [774 -> label %L775], [775 -> label %L776], [776 -> label %L777], [777 -> label %L778], [778 -> label %L779], [779 -> label %L780], [780 -> label %L781], [781 -> label %L782], [782 -> label %L783], [783 -> label %L784], [784 -> label %L785], [785 -> label %L786], [786 -> label %L787], [787 -> label %L788], [788 -> label %L789], [789 -> label %L790], [790 -> label %L791], [791 -> label %L792], [792 -> label %L793], [793 -> label %L794], [794 -> label %L795], [795 -> label %L796], [796 -> label %L797], [797 -> label %L798], [798 -> label %L799], [799 -> label %L800], [800 -> label %L801], [801 -> label %L802], [802 -> label %L803], [803 -> label %L804], [804 -> label %L805], [805 -> label %L806], [806 -> label %L807], [807 -> label %L808], [808 -> label %L809], [809 -> label %L810], [810 -> label %L811], [811 -> label %L812], [812 -> label %L813], [813 -> label %L814], [814 -> label %L815], [815 -> label %L816], [816 -> label %L817], [817 -> label %L818], [818 -> label %L819], [819 -> label %L820], [820 -> label %L821], [821 -> label %L822], [822 -> label %L823], [823 -> label %L824], [824 -> label %L825], [825 -> label %L826], [826 -> label %L827], [827 -> label %L828], [828 -> label %L829], [829 -> label %L830], [830 -> label %L831], [831 -> label %L832], [832 -> label %L833], [833 -> label %L834], [834 -> label %L

