
作者：禅与计算机程序设计艺术                    

# 1.简介
  


Birds are social animals that have been flying for millennia and sharing their wings with other birds to create collective flight patterns. Understanding how birds fly together is essential in understanding the world around us and improving our living conditions.

In this article, we will introduce you to the flocking behavior of birds and learn how they combine different types of behaviors to form complex flocks. We will also discuss some key concepts such as:

1. Separation (Separation or cohesion) - The ability of a group of boids to keep their distance from each other.
2. Alignment (Alignment or centering) - The tendency of individuals in a group to move towards the same direction without being too far away from one another.
3. Cohesion (Coherence or gathering) - The tendency of individuals in a group to move closer to each other, forming tight clusters.
4. Leader-Follower Model - A way of selecting leaders and followers within groups to influence their movement and improve overall performance.
5. Turning Patterns - Bird migration patterns based on individual's relative position and velocity. 

Let’s start by discussing what flocks are. In general terms, flocks refer to small groups of bird species that migrate together following a set of rules known as “flocking” behavior. These rules include separation, alignment, and cohesion, among others. Each rule acts independently but together they create complex migration patterns that give rise to different flock forms, including scattered, ravaging, or parasitic flocks. 

Flocks can be classified into three main categories: 

1. Centric flocks (also called colony flocks or homing flocks): This type of flock mimics the behaviour of single swarms of prey birds which feed on nectar provided by other flockmates. Examples of these flocks include common starlings, hens, or crows.

2. Collinear flocks (also called kite flocks or lure flocks): These flocks consist of a pair of closely related males and females, often separated from each other at a significant distance, performing migratory movements through direct dispersion only. Example of such flocks includes kites and tadpoles.

3. Orbiter flocks (or swooping flocks): These flocks use large numbers of tiny sensors placed throughout the body of the boid, like ants, to sense nearby obstacles and avoid collisions with them. They rely heavily on leader-follower relationships to select the best navigational paths while maintaining constant speed. An example of such flocks include herons, gulls, and magpies.

The success of any flocking behavior depends on a combination of factors, including the strength of the interactions between members of the flock, the complexity of the environment, and the specifics of the migration pattern being used. Therefore, it is important to carefully consider all aspects of flock formation and behavior before deciding on an approach that works well for your particular problem.



# 2. Basic Concepts and Terminology

Before going further let me first clarify some basic concepts and terminology regarding flocking behavior of birds.

1. Group (Flock) - A collection of individuals of similar characteristics who exhibit flocking behavior because of their relationship to other members of the group.

2. Migrant (Individual/Boid) - Any individual or agent that moves through the environment under the control of a coordinated group of other entities.

3. Waypoint (Target Point/Heading) - A fixed point in space where the bird wants to travel. This point may change frequently due to sudden changes in the environment or to emulate natural decision making.

4. Velocity Vector (Velocity) - The rate at which an entity (migrant/boid) is moving through the environment. It is expressed in units per time and measures the magnitude and direction of its motion. 

5. Positional Data (Location/Position) - Relative coordinates of a migrant or boid in space, typically represented by x, y, z axes respectively.

6. Behaviour (Behavioral Parameter) - Dynamic attribute of an individual that determines its physical, psychological, or neural responses to various stimuli or situations. Examples of behaviours include flight speed, size, color, shape, etc.

7. Genetic Diversity (Genotype) - Set of allelic variations that make up an organism's phenotype. Two individuals of the same species will have the same genotypes.

8. Phenotype (Organismal Characteristics) - The actual appearance and function of an organism, including morphology, behavior, and underlying biology.

9. Aberration (Attractiveness or Aggressiveness) - A deviation from normal flocking behavior. For instance, aberrations could come about when two individuals attempt to covertly join a flock to gain advantageous positions or avoid detection.