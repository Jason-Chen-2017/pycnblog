
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## Genetic programming (GP) is a powerful technique for evolving programs that can solve complex problems with high accuracy and adaptiveness. It uses principles of evolutionary computation to generate diverse solutions by combining genetic operators such as crossover and mutation. GP has been applied successfully to various fields including artificial intelligence, robotics, control systems, and finance. 

In the past few years, multi-population GP algorithms have shown promise for achieving better performance compared to single-population GP methods. These algorithms utilize multiple populations where each population represents different types of individuals in the problem space and operates independently from other populations. Therefore, these techniques help reduce convergence time, increase exploration, and enhance generalization ability of GP models. 


The traditional parallel evolutionary algorithm (PEA) is based on a simple two-layered approach where an initial population is generated randomly, followed by repeated iterations until convergence or stagnation occurs. In this paper, we present an extension of PEA called MP-PEA that involves multiple populations. This extension incorporates several strategies for improving the convergence rate and exploratory capability of PEA. The proposed method maintains multiple independent populations that evolve simultaneously towards more fit individuals. We also propose new fitness functions that are tailored for handling continuous variables and constraints.


We implement our MP-PEA framework in Python and demonstrate its effectiveness through three case studies: optimizing a binary classification problem, solving constrained optimization problems, and finding optimal trading signals using financial data. Finally, we conclude on future research directions for further advancing the field of parallel evolutionary algorithms for large-scale applications.




# 2.背景介绍

Evolutionary computing (EC), which is the study of adaptive processes inspired by natural selection, has proven to be effective for many practical purposes such as solving optimization problems, generating machine learning models, and designing computer architectures. However, evolutionary algorithms (EAs) are known to be sensitive to hyperparameters, making it challenging to tune them accurately. On the contrary, genetic programming (GP) is a powerful technique for evolving programs that can solve complex problems with high accuracy and adaptiveness. GP uses principles of evolutionary computation to generate diverse solutions by combining genetic operators such as crossover and mutation. GP has been applied successfully to various fields including artificial intelligence, robotics, control systems, and finance. 

In recent years, multi-population GP algorithms have shown promise for achieving better performance compared to single-population GP methods. These algorithms utilize multiple populations where each population represents different types of individuals in the problem space and operates independently from other populations. Therefore, these techniques help reduce convergence time, increase exploration, and enhance generalization ability of GP models. The traditional parallel evolutionary algorithm (PEA) is based on a simple two-layered approach where an initial population is generated randomly, followed by repeated iterations until convergence or stagnation occurs. In this paper, we present an extension of PEA called MP-PEA that involves multiple populations. This extension incorporates several strategies for improving the convergence rate and exploratory capability of PEA. The proposed method maintains multiple independent populations that evolve simultaneously towards more fit individuals. We also propose new fitness functions that are tailored for handling continuous variables and constraints.



# 3.基本概念术语说明
## Individuals (I):
Individuals in a GP program represent candidate solutions to the problem being optimized. Each individual consists of one parent node and zero or more child nodes. The number of nodes in an individual determines how complicated the solution may be. For example, an individual representing a polynomial function consisting of four terms could have between two and eight nodes depending on whether certain coefficients were fixed during training. An individual's structure is determined by the gene pool used to create it, which contains all possible combinations of operator choices and constant values within the allowed ranges. At the beginning of the search process, the entire gene pool is evaluated to produce an initial set of random individuals.

## Gene Pool (GP):
A set of alleles (i.e., instruction sets, constants, etc.) that can be combined into specific instructions to construct expressions. The combination of alleles creates an expression tree that defines the behavior of the program being optimized. The length and complexity of an expression tree determine the level of detail required to define an accurate representation of the problem being solved. As the size and complexity of the problem increases, so does the size and complexity of the available gene pool, which requires careful management to avoid creating overly complex trees that become unmaintainable. Moreover, appropriate selection mechanisms must be employed to ensure that only useful and interesting individuals survive in the long run. A good starting point would be to use real-world domain knowledge and prior experience to guide the construction of the gene pool.

## Fitness Function (FF):
A measure of how well an individual fits a particular problem. In a binary classification setting, the fitness function measures the probability of correctly classifying a test instance given its features. In a constrained optimization problem, the fitness function evaluates the quality of a solution subject to specified constraints. In our experiments, we use objective functions specifically designed for dealing with categorical variables and constraint satisfaction problems. Our implementation includes some popular fitness functions such as the mean squared error (MSE) for regression tasks and the area under the ROC curve (AUC) for binary classification tasks. We have also experimented with novel fitness functions such as the weighted k-nearest neighbor (WKNN) distance metric that takes into account the importance of different categories when evaluating the correctness of predictions. Additional fitness functions could be added to handle additional types of problems or improve their performance.

## Operator (Op):
An elementary operation performed inside a GP individual. Operators include arithmetic operations like addition, subtraction, multiplication, division, and comparison. Additionally, there are terminal operators, which represent leaf nodes in the expression tree and represent input variables, output variables, and constant values. Terminal operators do not require any parameters, while non-terminal operators take arguments from previous nodes in the tree. Non-terminal operators enable greater flexibility and expressivity than conventional programs because they allow constructs such as if-else statements and loops to be expressed compactly.

## Crossover (Cx):
A type of genetic recombination technique that combines parents to form offspring. In a GP context, crossover generates descendants by selecting pairs of parent individuals and swapping sections of their expression trees at predefined points. Crossover allows the creation of new solutions that exhibit similarities to the existing ones but still maintain distinct characteristics due to the unique nature of genetic information. By introducing diversity into the population, crossover helps to prevent premature convergence and explore areas of the search space that might otherwise remain poorly explored.

## Mutation (Mut):
A type of genetic modification technique that modifies the genetic material of an individual without changing its overall shape or functionality. In a GP context, mutations involve adding, removing, or modifying elements of an individual's expression tree. Mutations occur at defined rates and introduce small changes to the expression tree that preserve its functionality but result in a modified version that may be fitter than the original. Mutations typically alter the expression tree by adding or deleting random subtrees or shuffling the order of the remaining components. With enough mutations, even highly complex programs can be perturbed and improved upon, thus providing a promising mechanism for finding better solutions to optimization problems.

## Selection (Sel):
A process of choosing a subset of individuals from a population based on their fitness scores to pass down to the next generation. Selection involves ranking the individuals based on their performance and then selecting a portion of the best performing individuals to reproduce and pass onto the next generation. Different approaches exist for determining the proportion of selected individuals and for applying elitism or tournament selection strategies to promote genetic diversity among the population. Tournament selection selects pairs of individuals randomly, compares their performance, and keeps the winner. Elitism selects the most fit individual(s) and passes them on unchanged to the next generation while the rest of the population undergoes regular selection procedures.

## Population (P):
A collection of individuals that interact with one another to find better solutions to a problem. The population size controls the number of individuals in the pool, which affects both the exploration and exploitation capabilities of the GP model. Large populations tend to converge faster and to obtain better results, while smaller populations offer better fine-grained control and can provide insight into the underlying problem landscape. There are generally two types of populations in GP: superpopulations and subpopulations. Superpopulations consist of multiple subpopulations, each with its own set of individuals that operate independently from the others. Subpopulations are organized into different groups according to different criteria such as age, fitness score, or specialization. Subpopulations interact with each other during the course of a GP iteration and exchange individuals based on their fitness levels to achieve efficient exploration and exploitability.

## Iteration (It):
One cycle of the Evolutionary Computation (EC) loop, where a group of individuals, referred to as a population, breeds to produce offspring, mutates, and receives feedback from the environment. In a GP context, an iteration involves executing the following steps:

1. Evaluation - Evaluate the fitness of the current population members. Assign a fitness score to each member based on the degree of its match to the target goal.

2. Selection - Select the fittest members of the population to be carried forward into the next generation. There are various selection methods that choose a portion of the population to be passed on to the next generation, either for reproduction or for survival.

3. Reproduction - Create new offspring by combining selectively chosen parents. Offspring inherit the traits of their parents' genes and combine them with variations introduced by mutations.

4. Mutation - Introduce modifications to the offspring by substituting parts of the expression tree or replacing entire subtrees. This step aims to enforce genetic diversity within the population, allowing for a wider range of solutions to be considered.

5. Survival - Remove excess members from the population that fall below a certain threshold of fitness, ensuring that the population remains diverse and healthy. This can be achieved using various survival techniques, such as truncation, retention, or removal.

## Convergence:
A state of the EC loop where no significant improvement is observed in the fitness of the population. In a GP context, convergence occurs when no substantial change in the fitness of the population occurs after a sufficient number of iterations. If convergence cannot be achieved despite increasing the number of iterations, the algorithm may become stuck in a local minimum or oscillate around a fixed point. Overly aggressive termination criteria, such as a maximum number of generations, may also cause convergence to be premature.

## Stagnation:
A temporary state of the EC loop where little progress is made in improving the fitness of the population. When stagnation starts, the population may have reached a plateau or a saddle point. In contrast to convergence, stagnation usually resolves itself quickly, resulting in a population with low variance and much lower fluctuations in fitness. However, some cases of stagnation may persist even though the population has already converged, leading to deteriorating performance or instability.