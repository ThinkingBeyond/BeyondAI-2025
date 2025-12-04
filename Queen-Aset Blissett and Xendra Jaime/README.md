![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# Bridging Theory and Practice in the Universal Approximation Theorem: A Combined Theoretical, Computational, and Visual Study

## Research Question

Building on the [elementary proof framework](https://drive.google.com/file/d/1OHn-AuPGZyvG5FMUpfPGxIURVVQsI4ED/view?usp=sharing) introduced by Shaana Amarawickrama and Karuna Prakash in the BeyondAI 2024 Proceedings, our research question **extends on this foundation while bridging theoretical insights and practical application to provide a comprehensive study of the universal approximation theorem.**

## Key contributions

We examine the BeyondAI 2024 elementary proof framework demonstrating that any continuous function on the real numbers can be approximated by a Multilayer Perceptron (MLP) with several hidden layers and a finite number of neurons using the sigmoid activation function. We extend this framework to include the **tanh activation function**. Additionally, we investigate their assumption that an MLP with ReLU activation can approximate any continuous function by constructing piecewise linear functions made up of **simplices**. Furthermore, we bridge theory and practice by **empirically evaluating the behaviour of each activation function in simple cases**, comparing their practical usability and performance.

We complement these evaluations with **visual aids that illustrate the theoretical proof concepts**, and present plots and graphs summarizing the empirical experiment results. Our aim is to close the theoretical gaps in the original elementary proof framework and work towards generalizing the framework, while also exploring the practical reasons for why developers may choose one activation function over another in neural network design.

## Motivation

The universal approximation theorem is a key concept in neural networks, stating that a Multilayer Perceptron with a finite number of neurons in a single hidden layer can approximate any continuous function to any desired degree of accuracy. The elementary proof framework introduced in the BeyondAI 2024 Proceedings built on Michael Nielson’s visual proof by using elementary mathematics and visualisations to provide an intuitive and accessible explanation. However, some theoretical gaps remain, particularly surrounding the assumption that ReLU-activated MLPs approximate continuous functions through piecewise linear components constructed from simplices. 

Moreover, beyond the theoretical guarantee of approximation, practical neural network design choices, such as the selection of activation functions and network width, directly impact the  performance and usability of a network. To gain deeper insight into this universality theorem as well as understand the popular preference of certain activations, it is necessary to combine constructive theoretical proofs with empirical results and observations. 

In our empirical work, we compare the sigmoid, tanh and ReLU activations in a single-hidden-layer network, examining their performance on varying numbers of neurons and different target functions, namely sinusoidals, and gaussian. This integrated approach assesses optimal neural network design and deepens insight into the approximation capabilities of different activations.

Hence, our research aims to rigorously extend and close the gaps in the existing elementary proof framework, empirically explore three different non-linear activations, and provide visual explanations to support our work.

## Acknowledgements
We sincerely thank our mentor, Mr. Spencer Goodfellow, for his invaluable advice, thoughtful feedback and steady guidance throughout this research project. This work was made possible through the dedication and efforts of the amazing volunteers at ThinkingBeyond, whose support we deeply appreciate. 
A special thanks goes out to Dr. Filip Bar, whose insightful lessons and teaching methodology provided us with the essential tools, frameworks and renewed mindset that led us to expand on our skillset and sparked a genuine commitment to life-long learning and building subject expertise. 
We’re also really grateful to Ms. Stephanie Atherton, whose reliable assistance and unwavering support has encouraged us throughout the entirety of the program.
And of course, our final thanks goes out to all the peers in our BeyondAI 2025 cohort! It has been an absolute pleasure working alongside so many bright minds. Thank you all!

# Methodology
## Proof Methodology

The full proof methodology for our Extension Paper builds upon a foundational three-part constructive framework established for Sigmoid activation and extends it rigorously to Tanh, whilst also examining the conditions of ReLU. The overall goal is to provide an accessible yet **rigorous, constructive proof** that a Multilayer Perceptron (MLP) can approximate any continuous function $$f: \mathbb{R}^n \to \mathbb{R}^m$$.

---

### Project Phase 1: Sigmoid and Tanh (Piecewise Constant Approximation)

This phase establishes how activation functions approximate step functions and build piecewise constant approximations.

| Step | Action Taken | Purpose/Significance |
| --- | --- | --- |
| **Part 1: Step Function Convergence** | Algebraically prove that scaled activation functions ($$\sigma(wx)$$ or $$\tanh(wx)$$) converge **pointwise** to an ideal step function $$s(x)$$ as the weight $$w \to \infty$$, including threshold weight $$W$$ for error $$\epsilon$$. | Proves a single neuron can form **sharp decision boundaries** using large but finite weights. |
| **Part 2: Bump and Tower Construction** | Combine shifted step functions to construct a 1D bump function $$c(x)$$ and extend to $$n$$ dimensions via tower function $$t(x)$$. | Builds **indicator functions** localizing input space regions, which are the building blocks of approximation. |
| **Part 3: Piecewise Constant Approximation** | Use uniform continuity to partition domain into cells $$\{D_i\}$$ and define piecewise constant $$h(x) = f(u_i)$$ within each cell. | Bounds geometric error $$\| f(x) - h(x) \| \leq \epsilon$$; final approximation is a finite sum of scaled tower functions. |

![Image of Piecewise Constant Construction](https://github.com/ThinkingBeyond/BeyondAI-2025/blob/main/Queen-Aset%20Blissett%20and%20Xendra%20Jaime/Images/Piecewise%20Constant%20Function%20Formation.png)
*An image visualising the steps taken in the Bounded activation case (Sigmoid/Tanh) to approximate continuous functions*

---

### Project Phase 2: ReLU Activation (Piecewise Linear Approximation)

ReLU naturally produces piecewise linear functions; this phase rigorously proves such constructions via simplices.

| Step | Action Taken | Purpose/Significance |
| --- | --- | --- |
| **1D Proof (Foundation)** | Show ReLU MLP generates piecewise linear $$h_n(x)$$ approximating continuous $$f$$ by linear interpolation. | Demonstrates ReLU’s capacity for uniform approximation via linear segments. |
| **Higher-D Simplicial Partitioning** | Partition domain into simplices; define piecewise affine function $$g(x)$$ on these. | Provides geometric blueprint for multi-dimensional approximation with error control. |
| **Network Realization (Gap Closure)** | Prove ReLU activations realize linear inequalities defining simplex boundaries. | Closes key assumption; confirms ReLU MLPs realize complex piecewise affine geometry. |

![Image of Simplices Construction](https://github.com/ThinkingBeyond/BeyondAI-2025/blob/main/Queen-Aset%20Blissett%20and%20Xendra%20Jaime/Images/Simplices%20Construction.png)
*An image visualising the formation of simplices in the ReLU activation case*

## Visualisations
For phase 4, we are incorporating the existing static visualizations for our poster from last cohort, and extending on new visualizations for a better understanding.

**Modified visualizations from 2024 Proceedings**:
+ Step, Bump, Tower, Piecewise constant animation to show the progression in Sigmoid/Tanh cases ([`2025_Visualisations_for_Poster_Final.ipynb`](https://github.com/ThinkingBeyond/BeyondAI-2025/blob/main/Queen-Aset%20Blissett%20and%20Xendra%20Jaime/Visualizations/2025_Visualisations_for_Poster_Final.ipynb))
+ Visuals of piecewise linear functions built by ReLU (the simplices being assembled) ([`2025_Simplex_Construction_For_Poster.ipynb`](https://github.com/ThinkingBeyond/BeyondAI-2025/blob/main/Queen-Aset%20Blissett%20and%20Xendra%20Jaime/Visualizations/2025_Simplex_Construction_For_Poster.ipynb))
+ Continuity $\epsilon -\delta$ definition (to be added)
  
**New visualizations**:
+ Explanation of bounded versus unbounded for the 3 presented activation functions (to be added)
+ Input passing through an MLP (video graphic for what happens in a neural network) (to be added)

## Empirical Framework
For phase 3, we examined the performance of the activations, sigmoid, tanh and ReLU, in a shallow network to precisely examine how the inherent mathematical structures of these activations translate to practice, and influence neural network design and deep learning.

The present computational experiment in [`Empirical_Validation_of_the_UAT.ipynb`](https://github.com/ThinkingBeyond/BeyondAI-2025/blob/main/Queen-Aset%20Blissett%20and%20Xendra%20Jaime/Empirical_Validation_of_the_UAT.ipynb) followed the sequence below:
1. **Capacity Analysis**: Can we find a neural network that approximates our target function to our fixed threshold, as the UAT states?
2. **Efficiency Analysis**: Now that we have found one network, what parameters offer the best computational efficiency?
3. **Generalization Analysis**: Now that we have our optimized neural network, can it approximate outside of the given domain training?

Our work fully completes Step 1 and partially addresses Step 2, using metrics such as the Minimum viable width, Marginal Efficiency, Convergence time and Approximation Accuracy (MSE, MAE, Maximum Error and $R^2$).

To produce the experiment, run [`Empirical_Validation_of_the_UAT.ipynb`](https://github.com/ThinkingBeyond/BeyondAI-2025/blob/main/Queen-Aset%20Blissett%20and%20Xendra%20Jaime/Empirical_Validation_of_the_UAT.ipynb) on Google Colab.

**Versions used**:
* torch= 2.9.0+cu126
* numpy= 2.0.2
* matplotlib==3.10.0

Moreover, the [`uat_code_manual`](https://github.com/ThinkingBeyond/BeyondAI-2025/blob/main/Queen-Aset%20Blissett%20and%20Xendra%20Jaime/uat_code_manual.md) provides an overview on the provided code.

# Conclusions
## Empirical findings
- **The models benefit from the number of neurons up to a variable point:** While there was a significant improvement when increasing from 10N → 50N across all activations, there were also diminishing returns beyond ~500-1000 neurons for all tested cases
- **The UAT's "sufficient neurons" statement implicitly requires sufficient optimization:** To achieve good approximations, it is needed a careful tuning of hyperparameters (LR, epochs) beyond just adding neurons.
- **Activation-specific efficiency for sin(2πx) experiment:**
    - **Tanh:** Most efficient (50N, 0.82s)
    - **ReLU:** Moderate (50N, 4.37s)
    - **Sigmoid:** Slower but achieves convergence (50N, 1.09s)

**Experimental conclusion:**
The "sufficient number of neurons" required to approximate a target function depends not only on network width, but critically on optimization dynamics, such as the learning rate, training epochs and activation function properties. The Universal Approximation Theorem guarantees the existence of suitable weights, but does not prescribe how to find them via gradient descent.
Some configurations never converged despite UAT guarantees, highlighting the gap between existence (theory) and discoverability (practice), concluding that, even if the desired neural networks exists, it may be not easy to find.

# Results and Future Plans
We created a paper referencing the established framework, detailing our extension to the tanh activation and formalised a proof for the assumption that ReLU-activated MLPs can construct piecewise linear functions out of simplices. The experiments validated these structures and offered a point of discussion: what are those parameters that find the "already existing" network? 
Additionally, we supported our project findings with visualisations located in our poster. An experimental report will further detail our empirical findings (yet to be added to the GitHub) and a user manual was detailed for those who would like to engage with our experimental set-up and colab code. Future work entails, 
+ Have comparison visuals showcasing / reflecting each activation function’s theoretical construction to their experimental approximations behaviours.
+ More rigorous experimentation to explore more how to get to the desired arbitrary approximation, starting from a theoretical approach and deriving in computational experiments

## References

List all your references here. Remember to put links into markdown. For example:

1. Shaana Amarawickrama and Karuna Prakash. *An Elementary Proof of the Universal Approximation Theorem for Multilayer Perceptrons*. 2024. [GitHub Tree](https://github.com/ThinkingBeyond/BeyondAI-2024/tree/main/shaana-karuna)
2. Michael A. Nielsen. *Neural Networks and Deep Learning*. 2019. [Chapter 4: Visual Proof](http://neuralnetworksanddeeplearning.com/chap4.html)

---

> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://thinkingbeyond.education/beyondai_proceedings_2025/).

