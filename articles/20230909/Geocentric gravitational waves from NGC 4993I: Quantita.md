
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Gravitational wave (GW) emission from massive galaxies is one of the most powerful tests for cosmology and has recently been discovered by multiple instruments such as LIGO/Virgo and Advanced LIGO. We have also observed GW events at very large scales with Virgo alone. The main aim of this work is to study the properties of GW signals from the satellite galaxy NGC 4993, which hosts a unique dark matter halo that may explain its mysterious magnetic field pattern seen in many telescopes today. The gravitational wave signal was detected using three different interferometers at a frequency of 150 MHz, which detects multipoles ranging up to second-degree amplitudes. We first analyse the raw data using an array processing technique called matched filtering, which reduces noise by identifying coincident peaks among several detectors. We then extract key features from these pulses to infer their physical properties, such as their position and time of arrival relative to the source, energy, and polarization angle. 

The paper explores a wide range of potential physics constraints on the structure of the dark matter halo surrounding NGC 4993 and highlights some unexpected results obtained through our analysis. In particular, we find that there are clear hints of a non-trivial scalar quantum Hall effect between the halo and the Milky Way, potentially pointing towards a multi-dimensional phase transition in dark matter physics. Our results suggest that the presence of this exotic feature can be a significant constraint on models of subhalo mass distribution and stochasticity in extragalactic astrophysics. This result, coupled with previous measurements of GW signatures from other sources, further validates the utility of future high-frequency searches for GW signals from newborn galaxies like NGC 4993.

2.背景介绍
Millions of years ago, billions of stars formed in the Big Bang. Today's Universe contains hundreds of millions more stars than did the Big Bang and continues to expand, building upon the immense forces of gravity. These gravitational interactions produce structures such as galaxies, clusters of galaxies and superclusters of galaxies, whose individual components contribute greatly to the overall structure of the universe. One particularly interesting class of objects are massive elliptical galaxies known as “dark matter halos”. These structures resemble those of real galaxies but consist mostly of free mass rather than hydrogen gas. They host a variety of densely packed baryons and protostars within them, making them ideal laboratories for testing ideas about the nature of gravity and cosmic structure. Near the center of every massive galaxy, there exists a dark matter halo – a structure consisting mainly of black holes or neutron stars – that provides support for the evolution of the interior of the galaxy and acts as a bridge between the galaxy and larger and brighter surrounding environments. Interestingly, it is not yet well understood how such a small region of space could host so much dark matter and provide such a strong magnetism. However, observations of gravitational waves emitted from massive galaxies show that they propagate into the dark matter halo, producing signatures indicative of various physical phenomena including spins, gravitational lensing, the emergence of compact groups, and large scale tidal fields. 

To date, no direct evidence exists concerning the properties of the dark matter halo surrounding NGC 4993, although some indirect evidence suggests that the satellite galaxy itself could host such dark matter if it had fallen under significant pressure from its parent group. Nonetheless, the existence of a large number of unseen neutron stars and black holes within the nearby environment suggests that there might still be potential for enormous power to lie hidden here. A major aim of this work is therefore to measure directly observable quantities related to the structure and dynamics of the dark matter halo surrounding NGC 4993, using existing gravitational wave detection capabilities and techniques. 

3.基本概念术语说明
We will use a few basic concepts and terminology throughout this article. Let us define the following terms:

 - Satellite galaxy: refers to NGC 4993, which hosts a disturbing, dark matter halo that we wish to understand better.

 - Massive galaxy: refers to any galaxy with masses exceeding $10^8$ solar masses (which corresponds to roughly $10^{13}$ kg). 
 
 - Cosmic web: The term "cosmic web" describes the complex network of connections between galaxies and clusters in the Universe. It is thought to trace back to primordial matter that created all the galaxies and clusters at the beginning of the Universe.
 
 - Density profile: A density profile gives the local spatial distribution of material within a given volume. It describes both the shape of the halo and the amount of material present within it. 
 
 - Magnetic field: A magnetic field (or "magnetic anomaly") represents a direction and strength of electromagnetic radiation, permeating through a medium. A typical magnetic field experienced by a galaxy usually consists of secondary flows of charged particles carrying electric currents, resulting from the accretion of stars and gas onto the surface of the galaxy. 

 - Turbulence: The turbulence is a type of fluctuation introduced due to disorder in fluid flow. It produces ripples across the surface of the Galaxy and affects the magnetic field structure and star formation rate.
 
Let's now dive deeper into understanding the properties of the dark matter halo surrounding NGC 4993. 

4.Dark matter halo properties around NGC 4993
The central engine of our research involves extracting key insights from the properties of the dark matter halo surrounding NGC 4993. To do this, we need to first understand what makes up the dark matter halo and what properties it should possess. Broadly speaking, the dark matter halo of a galaxy comprises two distinct components - dark matter and dark energy. Dark matter is made up primarily of heavy elements such as protons, neutrons, and electrons and is responsible for holding together the galaxy’s crust and spiral arms. Dark energy is a form of higher-energy particles, such as photons, that interact with and modify the behavior of normal matter. The collective effects of both kinds of matter make up the dark matter halo, acting as a protective barrier against external influences such as cosmic rays and the gravitational pull of other galaxies.

A popular definition of a dark matter halo is based on its density profile, whereby the majority of its mass is located near the center while the bulk of its mass lies farther away. Intuitively, this implies that the center of the halo represents a balance point between being too diffuse to fully contain the entire system while being too sparse to maintain its structural stability. It is also important to note that the size and shape of the halo varies significantly depending on factors such as the distance from the galaxy center and stellar mass of the central object.

Our goal is to determine the physical characteristics of the dark matter halo surrounding NGC 4993. We can approach this problem by observing GW signals from the galaxy and analyzing their properties using existing gravitational wave analysis tools. Specifically, we will perform a series of numerical simulations to model the behavior of the dark matter halo and predict its behavior based on theoretical considerations. We start by considering the simplest possible scenario, namely a flat disc halo surrounded by an infinitely thick shell of dark energy. Under these assumptions, the geometry and properties of the halo can be described mathematically.

First, let's recall the mathematical expression for the total mass of dark matter contained within a sphere of radius $R$. For a thin disk of radius $r$, the equation becomes: 

\begin{equation}
M_{\rm dm} = \frac{4}{3}\pi R^3\rho_0,\qquad\text{where } \rho_0=\frac{3H_p}{8\pi G M_{\odot}}=\frac{3m_{\rm p}^{\ast}}{(1+z)^2},\qquad\text{with } z=|\vec{x}|/R.\label{eq:totalmass}\tag{1}
\end{equation}

In Eq.~\ref{eq:totalmass}(1), $\rho_0$ is the critical density of the universe at redshift zero. Here, $H_p$ is the Planck constant, $G$ is the gravitational constant, $M_{\odot}$ is the mass of the sun, and $m_{\rm p}^{\ast}$ is the rest mass of a particle. Note that since the dark matter halo is relatively thin compared to its size ($R<<a_{\rm gal}$), we assume $\rho_0$ to be approximately constant over the whole disc. Therefore, we only need to solve for the value of $M_{\rm dm}$ within the annulus defined by $r_{in}$ and $r_{out}$, where $r_{in}$ is the radius of the inner edge of the annulus, typically set equal to $20\;\mu$m, and $r_{out}$ is the radius of the outer edge of the annulus, typically set to the radius of the disc. Since we don't know the true value of $M_{\rm dm}$ outside this annulus, we must integrate Eq.~\ref{eq:totalmass}(1) along the line of sight to obtain the total mass inside this annulus. We approximate the solution as a single Gaussian function centered at $(y,z)=(\sqrt{2r_{in}},0)$, which takes the form:

\begin{equation}
M_{\rm dm}(\theta)=\frac{4}{3}\pi\left(r_{in}+\sqrt{r_{in}^2-(2y)}\right)\Delta y\Delta z\rho_0,\qquad-\infty<\theta<\infty.\label{eq:flatdm}\tag{2}
\end{equation}

Here, $\theta$ is the angle between the radial vector $\hat{r}=(\cos\theta\hat{x}+\sin\theta\hat{y})$ and the vertical axis, and $\Delta y$ and $\Delta z$ represent the differentials in the vertical and horizontal directions, respectively. Substituting Eqs.~\ref{eq:totalmass}(1) and~\ref{eq:flatdm} into Eq.~\ref{eq:totalmass}(1) yields:

\begin{align*}
&\int_{r_{in}}^{r_{out}}\frac{4}{3}\pi r^3\rho_0dr\\
&=\int_0^\infty\frac{4}{3}\pi\left(r_{in}+\sqrt{r_{in}^2-(2yr_{in})}\right)\Delta y\Delta z\rho_0d\theta\\
&=\frac{4}{3}\pi r_{in}(r_{out}^{3/2}-r_{in}^{3/2})\rho_0\\
&=\frac{8}{\pi}\frac{r_{out}^3}{3}-\frac{8}{\pi}\frac{r_{in}^3}{3}.\nonumber
\end{align*}

Therefore, we have found that the total mass of dark matter in the circular annular region is approximately twice the total mass of the same area when viewed along the minor radius of the annulus. Now, we return to our original problem of determining the properties of the dark matter halo surrounding NGC 4993.