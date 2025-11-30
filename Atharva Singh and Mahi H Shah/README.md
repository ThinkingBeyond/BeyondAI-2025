![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# Mitigating Oversmoothing in GNNs on Heterophilic Graphs

# Research question 
Our research seeks to evaluate how effectively two prominent techniques, **GCNII** and **DropEdge**, mitigate the **oversmoothing** problem in graph neural networks. We evaluate their performance on **heterophilic graph datasets** to determine whether methods originally designed for homophilic settings retain their advantage when node similarity assumptions break down.

# Motivation

## An introduction to GNNs

Graph Neural Networks (GNNs) are a class of deep learning architectures designed to operate directly on graph-structured data. Formally, a GNN can be viewed as an optimizable transformation over all components of a graph (its node features, edge attributes, and global structural context) while respecting core graph symmetries.

Graphs provide an exceptionally versatile modelling framework, capable of representing a wide spectrum of real-world systems in which entities interact in non-Euclidean ways. Social networks, citation networks, molecular structures, transportation systems, and knowledge graphs all naturally encode their information as nodes connected by edges. This flexibility has made GNNs a foundational tool in modern machine learning.

Typical downstream tasks include **node-level classification** (e.g., detecting anomalies or classifying individuals in a network), **edge-level prediction** (e.g., predicting new friendships or molecular bonds), and **graph-level tasks** (e.g., predicting molecular properties or detecting fraudulent subgraphs). Each task requires the model to propagate and refine information across varying neighbourhood radii, which introduces both expressive opportunities and structural challenges, including, most notably, *the oversmoothing problem.*

## The oversmoothing problem in GNNs

A well-documented limitation of deep GNNs is **oversmoothing**, a phenomenon in which node representations become progressively indistinguishable as the number of layers increases. Because each layer aggregates information from local neighbours, deeper architectures repeatedly mix information across larger neighbourhoods. Past a certain depth, this leads to a homogenization of node embeddings such that the model loses its ability to discriminate between distinct structural roles or feature patterns. Ultimately, oversmoothing degrades task performance, especially on graphs with sparse or heterophilic connectivity. 

(To visualise the oversmoothing problem, we have also created a short animation, to be attached separately.) 


# Architectures Addressing Oversmoothing

## GCNII

GCNII introduces two crucial innovations: **initial residual connections** and **identity mapping.** These counteract oversmoothing by preserving information from earlier layers. Specifically, each layer blends the transformed node representation with the **original input features**, ensuring that deeper layers do not fully drift toward an over-mixed equilibrium.


## DropEdge 
DropEdge is a regularization technique that **randomly removes a subset of edges during training**. By injecting structural noise, DropEdge reduces feature correlation between neighboring nodes, effectively delaying the onset of oversmoothing and improving generalization.

# Methodology

## Datasets

We conducted our experiments on the **WebKB dataset**, which consists of multiple heterophilic graphs (Texas, Cornell, Wisconsin). Heterophilic graphs present a more challenging test environment because connected nodes often belong to *different* classes, making them particularly relevant for tasks involving structural anomalies or irregular connectivity patterns.

## Experimental setup

We evaluated GCNII and DropEdge across **2, 4, 8, 16, and 32 layers** (with computational limits preventing extensions beyond that). For comparison, we also implemented baseline **GCN** and **GraphSAGE** models to isolate the impact of architectural design on depth scalability.

To verify the reliability of our implementations, we additionally replicated results from the original papers on benchmark citation networks. Our reproduction closely matched the reported outcomes. Notably, the GCNII authors provide official WebKB implementations for their GCN baselines, lending further confidence to the validity of their comparative claims, whereas DropEdge’s authors primarily benchmarked on homophilic citation graphs.

# Results

**Table: Performance of GCNII and DropEdge**

| Database  | Backbone   | 2-layer |        | 4-layer |        | 8-layer |        | 16-layer |        | 32-layer |        |
|-----------|------------|---------|--------|---------|--------|---------|--------|----------|--------|----------|--------|
|           |            | GCNII   | DropEdge | GCNII | DropEdge | GCNII | DropEdge | GCNII | DropEdge | GCNII | DropEdge |
| Texas     | GCN        | 71.89 | 67.57 | 75.41 | 67.57 | 70.27 | 64.86 | 69.73 | 64.86 | 71.35 | 64.86 |
| Texas     | GraphSAGE  | 78.38 | 83.78 | 69.19 | 81.36 | 58.92 | 81.08 | 59.19 | 72.97 | 59.46 | 70.27 |
| Cornell   | GCN        | 62.16 | 54.05 | 65.95 | 48.65 | 72.16 | 48.65 | 75.14 | 40.54 | 75.95 | 40.54 |
| Cornell   | GraphSAGE  | 71.62 | 75.68 | 64.05 | 75.06 | 49.41 | 62.16 | 58.92 | 55.93 | 58.92 | 54.05 |
| Wisconsin | GCN        | 72.16 | 60.78 | 74.31 | 58.82 | 74.90 | 58.82 | 71.76 | 56.86 | 71.18 | 54.90 |
| Wisconsin | GraphSAGE  | 75.69 | 78.43 | 67.65 | 76.47 | 72.16 | 72.55 | 48.24 | 72.50 | 47.45 | 56.86 |


# Discussion

The experimental results consistently show that **GCNII outperforms DropEdge** on heterophilic datasets across deeper architectures. This could be attributed to the fact that GCNII’s identity mapping stabilises the flow of information throughout the network, preserving node-specific characteristics even at depth. Because heterophilic graphs require models to resist oversmoothing aggressively (since neighbors are often of different classes), GCNII’s feature-preserving design confers a strong advantage.

By contrast, **DropEdge performs well in the original citation networks** (which are typically homophilic) because removing edges in a redundant, densely connected environment does not meaningfully disrupt the underlying class-consistent neighborhoods. In heterophilic graphs like WebKB, however, random edge removal risks discarding structurally important, low-redundancy connections. This may inadvertently break the weakly informative relational paths that models rely on, leading to the degraded performance observed.

Overall, the experiments highlight that techniques designed to mitigate oversmoothing are **highly sensitive to the structural properties of the graph**. Methods that rely on structural noise may excel in homophilic regimes but struggle in heterophily.

# Conclusion 
Our project demonstrates that depth-robust GNN architectures behave fundamentally differently on heterophilic datasets compared to the homophilic benchmarks where they were originally validated. By testing GCNII and DropEdge across multiple WebKB subsets and increasing depths, we provide empirical evidence that **oversmoothing manifests earlier and more severely in heterophilic settings**, making conventional regularization techniques like DropEdge far less effective. In contrast, GCNII’s identity mapping and initial residual connections preserve discriminative node information even at substantial depth, highlighting its suitability for real-world networks where relational similarity cannot be assumed.

These findings are significant for two main reasons:

1. Most GNN research (and many industry applications) implicitly assumes homophily, which can lead to misleading conclusions about model reliability on heterogeneous or noisy graphs.
2. Many high-stakes domains such as cybersecurity, fraud detection, biological interaction networks, and irregular web-graph analysis are inherently heterophilic. Models that fail under oversmoothing in such environments risk obscuring critical anomalies or rare structural patterns.

By showing that architectural design plays a decisive role in depth scalability under heterophily, our results underscore the need for **context-sensitive GNN evaluation** rather than relying solely on results from citation networks.

# Future work 
Future extensions of this research could explore:
- Scaling experiments to **64 and 128 layers** to evaluate the true depth limits of GCNII and test the breaking point of oversmoothing.
- Investigating **DropGNN**, a different method that removes nodes rather than edges, potentially suppressing noise by pruning inactive or weakly informative nodes.
- Incorporating **adaptive node and edge pruning strategies** into DropEdge for application-specific sparsification.
- Applying the models to additional heterophilic datasets (e.g., Actor, Amazon, Chameleon) to generalize observations.


# References 

1. [Rong, Y., Huang, W., Xu, T., & Huang, J. *DropEdge: Towards Deep Graph Convolutional Networks on Node Classification.* arXiv:1907.10903.](https://arxiv.org/abs/1907.10903)
2. Chen, M., Wei, Z., Huang, Z., Ding, B., & Li, Y. *Simple and Deep Graph Convolutional Networks.* arXiv:2007.02133.
3. Kipf, T. N., & Welling, M. *Semi-Supervised Classification with Graph Convolutional Networks.* arXiv:1609.02907.
4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. *Graph Attention Networks.* arXiv:1710.10903.
5. Xu, B., Shen, H., Cao, Q., Qiu, Y., & Cheng, X. *Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View.* AAAI Conference on Artificial Intelligence.
6. Hou, Y., Zhang, J., Cheng, J., Ma, K., Chen, H., & Yang, M.-C. *Measuring and Improving the Use of Graph Information in Graph Neural Networks.* arXiv:2206.13170.
7. Luan, S., Hua, C., Lu, Q., Zhu, J., Ma, L., Wu, L., Wang, X., Xu, M., Chang, X. W., Precup, D., Ying, R., Li, S. Z., Wolf, G., & Jegelka, S. *The Heterophilic Graph Learning Handbook.* arXiv:2407.09618.
8. Papp, P. A., Martinkus, K., Faber, L., & Wattenhofer, R. *DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks.* arXiv:2111.06283.
9. Lampert, M., & Scholtes, I. *The Self-Loop Paradox: Investigating the Impact of Self-Loops on Graph Neural Networks.* arXiv:2312.01721.
10. Bronstein, M., Bruna, J., Cohen, T., & Velickovic, P. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* Distill (2021).
11. Weng, L. *A Gentle Introduction to Graph Neural Networks.* Distill (2021).
12. Wang, K., Zhang, G., Zhang, X., Fang, J., Wu, X., Li, G., Pan, S., Huang, W., & Liang, Y. *The Heterophilic Snowflake Hypothesis: Training and Empowering GNNs 13. for Heterophilic Graphs.* arXiv:2406.12539.

---

> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://thinkingbeyond.education/beyondai_proceedings_2025/).

