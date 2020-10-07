# CS692 Seminar: Systems for Machine Learning, Machine Learning for Systems
Course website: [https://guanh01.github.io/teaching/2020-fall-mlsys](https://guanh01.github.io/teaching/2020-fall-mlsys) 


This is the (evolving) reading list for the seminar. The papers are from top ML venues (ICML, ICLR, etc) and system venues (ASPLOS, PLDI, etc). The selection criteria is whether some keywords are in paper title.  


Topics of interest include, but are not limited to (copied from [MLSys website](https://mlsys.org/Conferences/2021/CallForPapers)):
* Efficient model training, inference, and serving 
* Distributed and parallel learning algorithms
* Privacy and security for ML applications
* Testing, debugging, and monitoring of ML applications
* Fairness, interpretability and explainability for ML applications
* Data preparation, feature selection, and feature extraction
* ML programming models and abstractions
* Programming languages for machine learning
* Visualization of data, models, and predictions
* Specialized hardware for machine learning
* Hardware-efficient ML methods
* Machine Learning for Systems



### Table of Contents 
* [Systems for Machine Learning](#sys4ml)
	* [Distributed and Parallel Learning](#distributed)
	* [Efficient Training](#training)
	* [Efficient Inference](#inference)
	* [Testing and debugging](#debugging)
	* [Robustness](#robustness)
	* [Other Metrics](#other-metrics )
	* [Data Preparation](#data)
	* [ML programming models](#pl-models)
* [Machine Learning for Systems](#ml4sys)
	* [ML Systems](#ml4ml)
	* [Compiler Optimization](#compiler)
	* [Programming Language](#pl)
	* [Memory Management](#mm)
* [General Reports](#reports)
* [Other Resources](#other)
 
## Systems for Machine Learning <a name="sys4ml"></a>

### Distributed and Parallel Learning <a name="distributed"></a>
* [ICLR'20][Decentralized Deep Learning with Arbitrary Communication Compression ](https://openreview.net/forum?id=SkgGCkrKvH)
* [ASPLOS'20][Prague: High-Performance Heterogeneity-Aware Asynchronous Decentralized Training.](https://dl.acm.org/doi/abs/10.1145/3373376.3378499) Prague is a high-performance heterogeneity-aware asynchronous decentralized training approach that improves the performance for high heterogeneity systems. It has two contributions.  First, it reduces synchronization cost via Partial All-Reduce that enables fast synchronization among a group of workers. Second, it reduces serialization cost via static group scheduling in homogeneous environment and simple techniques, i.e., Group Buffer and Group Division, to largely
eliminate conflicts with slightly reduced randomness.
* [EuroSys'20][Balancing Efficiency and Fairness in Heterogeneous GPU Clusters for Deep Learning](https://dl.acm.org/doi/abs/10.1145/3342195.3387555)
This paper talks about the scheduling of GPU clusters, which is not necessary related to deep learning.
* [EuroSys'19][Supporting Very Large Models using Automatic Dataflow Graph Partitioning](http://www.news.cs.nyu.edu/~jinyang/pub/tofu-eurosys19.pdf)
* [SOSP'19][A generic communication scheduler for distributed DNN training acceleration](https://dl.acm.org/doi/10.1145/3341301.3359642) Partition and rearrange the tensor transmissions without changing the code of underlying framework, such as TensorFlow, PyTorch, and MXNet, by reducing the communication overhead. 
* [SOSP'19][PipeDream: generalized pipeline parallelism for DNN training](https://dl.acm.org/doi/10.1145/3341301.3359646)
Proposed a inter-batch pipelining to improve parallel training throughput.
* [Survey'19] [Scalable Deep Learning on Distributed Infrastructures: Challenges, Techniques, and Tools](https://arxiv.org/pdf/1903.11314.pdf) by Mayer, Ruben, and Hans-Arno Jacobsen. ACM Computing Surveys (CSUR)

### Efficient Training <a name="training"></a>

#### DNN Training 
* [MLSys'20][MLPerf Training Benchmark](https://proceedings.mlsys.org/papers/2020/134)
* [ICLR'20(talk)][Reformer: The Efficient Transformer ](https://openreview.net/forum?id=rkgNKkHtvB)
* [ICLR'20(spotlight)][Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks ](https://openreview.net/forum?id=BJxsrgStvr)
* [ICLR'20][Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints](https://openreview.net/forum?id=HyxLRTVKPH)
* [ICML'20][DeltaGrad: Rapid retraining of machine learning models](https://icml.cc/virtual/2020/poster/5915) 
* [ASPLOS'20][Capuchin: Tensor-based GPU Memory Management for Deep Learning](https://dl.acm.org/doi/10.1145/3373376.3378505) 
* [ASPLOS'19][Split-CNN: Splitting Window-based Operations in Convolutional Neural Networks for Memory System Optimization](https://dl.acm.org/doi/abs/10.1145/3297858.3304038) The paper addresses the issue of non-sufficient memory issue of GPUs for CNN training. It proposes two approaches. First, it splits a CNN network to multiple smaller ones. Second, it proposes to utilize nv-link to implement memory management offloading. 



#### GNN Training 
* [ICLR'20(talk)][GraphZoom: A Multi-level Spectral Approach for Accurate and Scalable Graph Embedding ](https://openreview.net/forum?id=r1lGO0EKDH)
* [MLSys'20][Improving the Accuracy, Scalability, and Performance of Graph Neural Networks with Roc](https://proceedings.mlsys.org/papers/2020/83) The paper presents a distributed multi-GPU framework for fast GNN training and inference on graphs. ROC tackles two significant system challenges for distributed GNN computation: Graph partitioning and (2) memory management.  Graph partitioning is based on an online-trained linear regression model. The memory management decides in which device memory to store each intermediate tensor to minimize data transfter cost. ROC introduces a dynamic programming
algorithm to minimize data transfer cost. 
* [IPDPS'20][PCGCN: Partition-Centric Processing for Accelerating Graph Convolutional Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139807)
* [ArXiv'20][Reducing Communication in Graph Neural Network Training](https://arxiv.org/abs/2005.03300)
* [ICLR'20][GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/pdf/1907.04931.pdf)
* [NIPS'19][Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks](http://papers.nips.cc/paper/by-source-2019-6006)
* [KDD'19][Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf)
* [ATC'19][NeuGraph: Parallel Deep Neural Network Computation on Large Graphs](https://www.usenix.org/conference/atc19/presentation/ma) NeuGraph is a new framework that bridges the graph and dataflow models to support efficient and scalable parallel neural network computation on graphs. Basically, it  manages data partitioning, scheduling, and parallelism in dataflow-based deep learning frameworks in order to achieve better performance for GNN training. 
* [ICLR'19 Workshop][Deep graph library: Towards efficient and scalable deep learning on graphs](https://arxiv.org/abs/1909.01315)
* [VLDB'19][AliGraph: A Comprehensive Graph Neural Network Platform](http://www.vldb.org/pvldb/vol12/p2094-zhu.pdf)

#### Neural Architecture Search 

* [ICLR'20][Once-for-All: Train One Network and Specialize it for Efficient Deployment ](https://openreview.net/forum?id=HylxE1HKwS)
* [ICLR'20][Fast Neural Network Adaptation via Parameter Remapping and Architecture Search ](https://openreview.net/forum?id=rklTmyBKPH)
* [ICML'20][Breaking the Curse of Space Explosion: Towards Efficient NAS with Curriculum Search](https://icml.cc/virtual/2020/poster/5803) 


#### Continous Learning  

* [ICLR'20][Continual learning with hypernetworks ](https://openreview.net/forum?id=SJgwNerKvB)
* [ICLR'20][Continual Learning with Adaptive Weights (CLAW) ](https://openreview.net/forum?id=Hklso24Kwr) 
* [ICLR'20][Scalable and Order-robust Continual Learning with Additive Parameter Decomposition ](https://openreview.net/forum?id=r1gdj2EKPB)


### Efficient Inference <a name="inference"></a>

### Compiler 
* [SOSP'19][TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions](https://cs.stanford.edu/~matei/papers/2019/sosp_taso.pdf)
* [SOSP'18][TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/system/files/osdi18-chen.pdf)


#### Resource Management 
* [VLDB'21][Jointly Optimizing Preprocessing and Inference for DNN-based Visual Analytics](https://arxiv.org/pdf/2007.13005.pdf)
* [MLSys'20] [Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications](https://proceedings.mlsys.org/paper/2020/file/f7177163c833dff4b38fc8d2872f1ec6-Paper.pdf)
* [MLSys'20][Willump: A Statistically-Aware End-to-end Optimizer for Machine Learning Inference](https://arxiv.org/pdf/1906.01974.pdf)
* [MobiSys'20][Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization](https://dl.acm.org/doi/abs/10.1145/3386901.3388947)
* [RTSS'19][Pipelined Data-Parallel CPU/GPU Scheduling for Multi-DNN Real-Time Inference](https://intra.ece.ucr.edu/~hyoseung/pdf/rtss19-dart.pdf)
* [EuroSys'19][GRNN: Low-Latency and Scalable RNN Inference on GPUs](https://dl.acm.org/doi/10.1145/3302424.3303949) The paper improves the performance of RNN inference by providing a GPU-based RNN inference library, called GRNN, that provides low latency, high throughput, and efficient resource utilization.
* [EuroSys'19][μLayer: Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization](https://dl.acm.org/doi/10.1145/3302424.3303950) μLayer is a low latency on-device inference runtime that significantly improves the latency of NN-assisted services. μLayer accelerates each NN layer by simultaneously utilizing diverse heterogeneous processors on a mobile device and by performing computations using processor-friendly quantization. First, to accelerate an NN layer using both the CPU and the GPU at the same time, μLayer employs a layer distribution mechanism which completely removes redundant computations between the processors. Next, μLayer optimizes the per-processor performance by making the processors utilize different data types that maximize their utilization. In addition, to minimize potential latency increases due to overly aggressive workload distribution, μLayer selectively increases the distribution granularity to divergent layer paths.
* [sysml@nips'18][Dynamic Space-Time Scheduling for GPU Inference](http://learningsys.org/nips18/assets/papers/102CameraReadySubmissionGPU_Virtualization%20(8).pdf)
* [MobiSys'17][DeepEye: Resource Efficient Local Execution of Multiple Deep Vision Models using Wearable Commodity Hardware](https://dl.acm.org/doi/pdf/10.1145/3081333.3081359)


#### Compression 
* [ICML'20][Train Big, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](https://icml.cc/virtual/2020/poster/6828)
* [ICLR'20][BlockSwap: Fisher-guided Block Substitution for Network Compression on a Budget](https://openreview.net/forum?id=SklkDkSFPB)
* [ICLR'20][Probabilistic Connection Importance Inference and Lossless Compression of Deep Neural Networks ](https://openreview.net/forum?id=HJgCF0VFwr)
* [ICLR'20][Scalable Model Compression by Entropy Penalized Reparameterization ](https://openreview.net/forum?id=HkgxW0EYDS) 
* [ICLR'20][Compression based bound for non-compressed network: unified generalization error analysis of large compressible deep neural network ](https://openreview.net/forum?id=ByeGzlrKwH) 

#### Pruning 
* [ICML'20][PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination](https://icml.cc/virtual/2020/poster/6835) 
* [MLSys'20][What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/papers/2020/73) This paper provides an overview of approaches to pruning. The finding is that the community sufers from a lack of standardized benchmarks and metrics. It is hard to compare pruning techniques to one another or determine the progress the filed has made. The paper also introduce ShrinkBench, an open-source framework to faciliate standardized evaluations of pruning methods. 
* [ICML'20][PENNI: Pruned Kernel Sharing for Efficient CNN Inference](https://icml.cc/virtual/2020/poster/6232)
* [ICML'20][Operation-Aware Soft Channel Pruning using Differentiable Masks](https://icml.cc/virtual/2020/poster/5997) 
* [ICML'20][DropNet: Reducing Neural Network Complexity via Iterative Pruning](https://icml.cc/virtual/2020/poster/6092)
* [ICLR'20][A Signal Propagation Perspective for Pruning Neural Networks at Initialization ](https://openreview.net/forum?id=HJeTo2VFwH) 
* [ICML'20][Network Pruning by Greedy Subnetwork Selection](https://icml.cc/virtual/2020/poster/6053)
* [ICLR'20][Talk][Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://openreview.net/forum?id=S1gSj0NKvB) 
* [ICLR'20][Lookahead: A Far-sighted Alternative of Magnitude-based Pruning ](https://openreview.net/forum?id=ryl3ygHYDB)
* [ICLR'20][Provable Filter Pruning for Efficient Neural Networks ](https://openreview.net/forum?id=BJxkOlSYDH)
* [ICLR'20][Dynamic Model Pruning with Feedback ](https://openreview.net/forum?id=SJem8lSFwB)
* [ICLR'20][One-Shot Pruning of Recurrent Neural Networks by Jacobian Spectrum Evaluation ](https://openreview.net/forum?id=r1e9GCNKvH)


#### Quantization 
* [ICML'20][Boosting Deep Neural Network Efficiency with Dual-Module Inference](https://icml.cc/virtual/2020/poster/6670)
* [ICML'20][Towards Accurate Post-training Network Quantization via Bit-Split and Stitching](https://icml.cc/virtual/2020/poster/5787)
* [ICML'20][Differentiable Product Quantization for Learning Compact Embedding Layers](https://icml.cc/virtual/2020/poster/6429) 
* [ICML'20][Online Learned Continual Compression with Adaptive Quantization Modules](https://icml.cc/virtual/2020/poster/6338) 
* [ICML'20][Multi-Precision Policy Enforced Training (MuPPET) : A Precision-Switching Strategy for Quantised Fixed-Point Training of CNNs](https://icml.cc/virtual/2020/poster/6250)
* [ICML'20][Divide and Conquer: Leveraging Intermediate Feature Representations for Quantized Training of Neural Networks](https://icml.cc/virtual/2020/poster/6676) 
* [ICML'20][Up or Down? Adaptive Rounding for Post-Training Quantization](https://icml.cc/virtual/2020/poster/6482)
* [ICLR'20][And the Bit Goes Down: Revisiting the Quantization of Neural Networks ](https://openreview.net/forum?id=rJehVyrKwH)
* [ICML'20][Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware ](https://openreview.net/forum?id=H1lBj2VFPS) 
* [ICLR'20][AutoQ: Automated Kernel-Wise Neural Network Quantization ](https://openreview.net/forum?id=rygfnn4twS)
* [ICLR'20][Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks ](https://openreview.net/forum?id=BkgXT24tDS)
* [ICLR'20][Shifted and Squeezed 8-bit Floating Point format for Low-Precision Training of Deep Neural Networks ](https://openreview.net/forum?id=Bkxe2AVtPS)
* [ICLR'20][Mixed Precision DNNs: All you need is a good parametrization ](https://openreview.net/forum?id=Hyx0slrFvH)
* [MLSys'20][Riptide: Fast End-to-End Binarized Neural Networks](https://proceedings.mlsys.org/papers/2020/155)
This paper tries to speedup the implementation of binarized neural networks on Raspberry Pi 3B (RPi). 
* [MLSys'20][Memory-Driven Mixed Low Precision Quantization for Enabling Deep Network Inference on Microcontrollers](https://proceedings.mlsys.org/papers/2020/130) The paper studies mixed low-bitwidth compression, featuring 8, 4 or 2-bit uniform quantization to enable integer-only operations on microcontrollers. It determines the minimum bit precision of every activation and weight tensor given the memory constraints of a device. The authors evaluate their approach using MobileNetV1 on a STM32H7 microcontroller. 


#### Model Serving 
* [SOSP'19][Parity models: erasure-coded resilience for prediction serving systems](https://dl.acm.org/doi/pdf/10.1145/3341301.3359654)
Improving the performance of prediction serving system that could take in queries and return
predictions by performing inference on models. This paper means to mitigate tail latency inflation. 
* [ATC'19][MArk: Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving](https://www.usenix.org/conference/atc19/presentation/zhang-chengliang)
The work focuses on the improvement of performance for ML-as-a-Service:: developers train ML models and publish them in the cloud as online services to provide low-latency inference at scale.  
* [OSDI'18][PRETZEL: Opening the Black Box of Machine Learning Prediction Serving Systems](https://www.usenix.org/system/files/osdi18-lee.pdf): PRETZEL is  a  prediction  serving  system  introducing  anovel white box architecture enabling both end-to-endand multi-model optimizations. PRETZELis on average able to reduce 99th percentile la-tency by 5.5× while reducing memory footprint by 25×, and increasing throughput by 4.7×.


### Testing and Debugging <a name="debugging"></a>
* [MLSys'20][Model Assertions for Monitoring and Improving ML Models](https://proceedings.mlsys.org/papers/2020/189) The paper tries to monitor and improve ML models by using model assertions at all stages of ML system delopyment, including runtime monitoring and validating labels.  For runtime monitoring, model assertions can find high confidence errors. For training, they propose a bandit-based active learning algorithm that can sample from data flagged by assertion to reduce labeling cost. 




### Robustness <a name="robustness"></a>

* [PLDI'20][Proving Data-Poisoning Robustness in Decision Trees](https://dl.acm.org/doi/abs/10.1145/3385412.3385975) This paper studies the robustness of decision tree models to the data poinsoning. It proposes a system that could still guarantee the correctness of model, even if the data set has been tempered. 
* [ASPLOS'20][DNNGuard: An Elastic Heterogeneous DNN Accelerator Architecture against Adversarial Attacks](https://dl.acm.org/doi/abs/10.1145/3373376.3378532) The paper proposes an elastic heterogeneous DNN accelerator architec-ture that can efficiently orchestrate the simultaneous execu-tion of original (target) DNN networks and thedetectalgo-rithm or network that detects adversary sample attacks. 
* [ICML'20][Proper Network Interpretability Helps Adversarial Robustness in Classification](https://icml.cc/virtual/2020/poster/6031)
* [ICML'20][More Data Can Expand The Generalization Gap Between Adversarially Robust and Standard Models](https://icml.cc/virtual/2020/poster/5943)
* [ICML'20][Dual-Path Distillation: A Unified Framework to Improve Black-Box Attacks](https://icml.cc/virtual/2020/poster/6318)
* [ICML'20][Defense Through Diverse Directions](https://icml.cc/virtual/2020/poster/6557)
* [ICML'20][Adversarial Robustness Against the Union of Multiple Threat Models](https://icml.cc/virtual/2020/poster/6411)
* [ICML'20][Second-Order Provable Defenses against Adversarial Attacks](https://icml.cc/virtual/2020/poster/6256)
* [ICML'20][Understanding and Mitigating the Tradeoff between Robustness and Accuracy](https://icml.cc/virtual/2020/poster/6801) 
* [ICML'20][Adversarial Robustness via Runtime Masking and Cleansing](https://icml.cc/virtual/2020/poster/5817)
* [ICLR'20][Talk][Adversarial Training and Provable Defenses: Bridging the Gap ](https://openreview.net/forum?id=SJxSDxrKDr)
* [ICLR'20][Defending Against Physically Realizable Attacks on Image Classification ](https://openreview.net/forum?id=H1xscnEKDr) 
* [ICLR'20][Enhancing Adversarial Defense by k-Winners-Take-All ](https://openreview.net/forum?id=Skgvy64tvr)
* [ICLR'20][Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks ](https://openreview.net/forum?id=ByxtC2VtPB)
* [ICLR'20][Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness ](https://openreview.net/forum?id=Byg9A24tvB)
* [ICLR'20][Robust Local Features for Improving the Generalization of Adversarial Training](https://openreview.net/forum?id=H1lZJpVFvr) 
* [ICLR'20][GAT: Generative Adversarial Training for Adversarial Example Detection and Robust Classification ](https://openreview.net/forum?id=SJeQEp4YDH)
* [ICLR'20][Robust training with ensemble consensus ](https://openreview.net/forum?id=ryxOUTVYDH)
* [ICLR'20][Fast is better than free: Revisiting adversarial training ](https://openreview.net/forum?id=BJx040EFvH)
* [ICLR'20][EMPIR: Ensembles of Mixed Precision Deep Networks for Increased Robustness Against Adversarial Attacks ](https://openreview.net/forum?id=HJem3yHKwH)
* [ICLR'20][Triple Wins: Boosting Accuracy, Robustness and Efficiency Together by Enabling Input-Adaptive Inference ](https://openreview.net/forum?id=rJgzzJHtDB)

### Other Metrics (Interpretability, Privacy, etc.) <a name="other-metrics"></a>

* [MLSys'20][Privacy-Preserving Bandits](https://proceedings.mlsys.org/papers/2020/136)  The paper tries to enable privacy in personalized recommendation. This paper proposes a technique Privacy-Preserving Bandits (P2B); a system that updates local agents by collecting feedback from other local agents in a differentially-private manner. 
* [ASPLOS'20][Shredder: Learning Noise Distributions to Protect Inference Privacy](https://dl.acm.org/doi/pdf/10.1145/3373376.3378522)
The work focuses on the cloud-based inference. It introduces the noise to the input data, but without sacrificing the accuracy of inference. 
* [ASPLOS'20][DeepSniffer: A DNN Model Extraction Framework Based on Learning Architectural Hints. 385-399](https://dl.acm.org/doi/abs/10.1145/3373376.3378460) DeepSniffer extracts the model architecture information by learning the relation between extracted architectural hints (e.g., volumes of memory reads/writes obtained by side-channel or bus snooping attacks) and model internal architectures.



### Data Preparation <a name="data"></a>

* [MLSys'20][Attention-based Learning for Missing Data Imputation](https://proceedings.mlsys.org/papers/2020/123) The paper focuses on mixed (discrete and continuous) data and proposes AimNet, an attention-based learning network for missing data imputation in HoloClean, a state-of-the-art ML-based data cleaning framework. The paper argues that  attention should be a central component in deep learning architectures for data imputation.  
* [ICLR'20][Can gradient clipping mitigate label noise? ](https://openreview.net/forum?id=rklB76EKPr)
* [ICLR'20][SELF: Learning to Filter Noisy Labels with Self-Ensembling ](https://openreview.net/forum?id=HkgsPhNYPS)

### ML programming models <a name="pl-models"></a>

* [MLSys'20][Sense & Sensitivities: The Path to General-Purpose Algorithmic Differentiation](https://proceedings.mlsys.org/papers/2020/16) present Zygote, an algorithmic differentiation (AD) system for the Julia language. 


## Machine Learning for Systems <a name="ml4sys"></a>

### ML for ml system <a name='ml4ml'></a>
* [asplos'20][FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://dl.acm.org/doi/abs/10.1145/3373376.3378508?casa_token=YWcNPnp03fsAAAAA:Xo3RLkiykJSN8H70bsiQre-0hI20U5Sgu3_LYbsIqOSCsi8aBay18752gyZSFvYVlG34pTjrYyHm)

* [Learning to Optimize Tensor Programs](https://papers.nips.cc/paper/7599-learning-to-optimize-tensor-programs.pdf)

### ML for compiler <a name='compiler'></a>
* [MLsys'20][AutoPhase: Juggling HLS Phase Orderings in Random Forests with Deep Reinforcement Learning](https://proceedings.mlsys.org/paper/2020/file/4e732ced3463d06de0ca9a15b6153677-Paper.pdf)

### ML for programming languages <a name="pl"></a>

* [PLDI'20][Learning Nonlinear Loop Invariants with Gated Continuous Logic Networks](https://dl.acm.org/doi/abs/10.1145/3385412.3385986) The paper proposes a new neuro architecture (Gated Continuous Logic Network(G-CLN)) to learn nonlinear loop invariants. Utilizing DNN to solve and understand the system issue. 
* [PLDI'20][Blended, Precise Semantic Program Embeddings](https://dl.acm.org/doi/abs/10.1145/3385412.3385999) This paper is utilizing ML for systems. Basically, it utilizes DNN to learn program embeddings, vector representations of pro-gram semantics. Existing approaches predominately learn to embed programs from their source code, and, as a result, they do not capture deep, precise program semantics. On the other hand, models learned from runtime information critically depend on the quality of program executions, thus leading to trainedmodels with highly variant quality. LiGer learns programrepresentations from a mixture of symbolic and concrete exe-cution traces.
* [ICLR'18][Learning to Represent Programs with Graphs](https://arxiv.org/abs/1711.00740)

### ML for memory management <a name="mm"></a>

* [ICML'20][An Imitation Learning Approach for Cache Replacement](https://icml.cc/virtual/2020/poster/6044)
* [(ASPLOS'20][Learning-based Memory Allocation for C++ Server Workloads](https://dl.acm.org/doi/pdf/10.1145/3373376.3378525)


## General Reports <a name="reports"></a>
- [MLSys Whitepaper'18]: [SysML: The New Frontier of Machine Learning Systems](https://arxiv.org/abs/1904.03257)[must-read]
- [NeurIPS'15][Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)


## Other Resources <a name="other"></a>

- [https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning)
- [https://wencongxiao.github.io/articles/mlsyspaperlist/](https://wencongxiao.github.io/articles/mlsyspaperlist/)
- [https://github.com/thunlp/GNNPapers](https://github.com/thunlp/GNNPapers)
- [http://mlforsystems.org/](http://mlforsystems.org/)
- [https://paperswithcode.com/](https://paperswithcode.com/)
- [https://emeryberger.com/systems-lunch/](https://emeryberger.com/systems-lunch/)
- [https://deeplearn.org/](https://deeplearn.org/)
- [https://www.connectedpapers.com/](https://www.connectedpapers.com/)
- [Software Engineering for AI/ML -- An Annotated Bibliography](https://github.com/ckaestne/seaibib)
