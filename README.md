# CS692 Seminar: Systems for Machine Learning, Machine Learning for Systems
This is the (evolving) reading list for the seminar. The papers are selected from top venues (ICML, ICLR, PLDI, MLSys, etc) and are mostly published 2020. Topics of interest include, but are not limited to (copied from [MLSys website](https://mlsys.org/Conferences/2021/CallForPapers)):
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
	* [Programming Language](#pl)
	* [Memory Management](#mm)
 
## Systems for Machine Learning <a name="sys4ml"></a>

### Distributed and Parallel Learning <a name="distributed"></a>

* [Decentralized Deep Learning with Arbitrary Communication Compression ](https://openreview.net/forum?id=SkgGCkrKvH)
* [A generic communication scheduler for distributed DNN training acceleration](https://dl.acm.org/doi/10.1145/3341301.3359642) Partition and rearrange the tensor transmissions without changing the code of underlying framework, such as TensorFlow, PyTorch, and MXNet, by reducing the communication overhead. 
* [PipeDream: generalized pipeline parallelism for DNN training](https://dl.acm.org/doi/10.1145/3341301.3359646)
Proposed a inter-batch pipelining to improve parallel training throughput.
* [Prague: High-Performance Heterogeneity-Aware Asynchronous Decentralized Training.](https://dl.acm.org/doi/abs/10.1145/3373376.3378499) Prague is a high-performance heterogeneity-aware asynchronous decentralized training approach that improves the performance for high heterogeneity systems. It has two contributions.  First, it reduces synchronization cost via Partial All-Reduce that enables fast synchronization among a group of workers. Second, it reduces serialization cost via static group scheduling in homogeneous environment and simple techniques, i.e., Group Buffer and Group Division, to largely
eliminate conflicts with slightly reduced randomness.
* [Balancing Efficiency and Fairness in Heterogeneous GPU Clusters for Deep Learning](https://dl.acm.org/doi/abs/10.1145/3342195.3387555)
This paper talks about the scheduling of GPU clusters, which is not necessary related to deep learning.
* [Supporting Very Large Models using Automatic Dataflow Graph Partitioning](http://www.news.cs.nyu.edu/~jinyang/pub/tofu-eurosys19.pdf)

### Efficient Training <a name="training"></a>

#### DNN Training 
* [MLPerf Training Benchmark](https://proceedings.mlsys.org/papers/2020/134)
* [Reformer: The Efficient Transformer ](https://openreview.net/forum?id=rkgNKkHtvB)
* [Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks ](https://openreview.net/forum?id=BJxsrgStvr)
* [Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints](https://openreview.net/forum?id=HyxLRTVKPH)
* [DeltaGrad: Rapid retraining of machine learning models](https://icml.cc/virtual/2020/poster/5915) 
* [Split-CNN: Splitting Window-based Operations in Convolutional Neural Networks for Memory System Optimization](https://dl.acm.org/doi/abs/10.1145/3297858.3304038) The paper addresses the issue of non-sufficient memory issue of GPUs for CNN training. It proposes two approaches. First, it splits a CNN network to multiple smaller ones. Second, it proposes to utilize nv-link to implement memory management offloading. 
* [Capuchin: Tensor-based GPU Memory Management for Deep Learning](https://dl.acm.org/doi/10.1145/3373376.3378505) 



#### GNN Training 
* [GraphZoom: A Multi-level Spectral Approach for Accurate and Scalable Graph Embedding ](https://openreview.net/forum?id=r1lGO0EKDH)
* [Improving the Accuracy, Scalability, and Performance of Graph Neural Networks with Roc](https://proceedings.mlsys.org/papers/2020/83) The paper presents a distributed multi-GPU framework for fast GNN training and inference on graphs. ROC tackles two significant system challenges for distributed GNN computation: Graph partitioning and (2) memory management.  Graph partitioning is based on an online-trained linear regression model. The memory management decides in which device memory to store each intermediate tensor to minimize data transfter cost. ROC introduces a dynamic programming
algorithm to minimize data transfer cost. 
* [NeuGraph: Parallel Deep Neural Network Computation on Large Graphs](https://www.usenix.org/conference/atc19/presentation/ma) NeuGraph is a new framework that bridges the graph and dataflow models to support efficient and scalable parallel neural network computation on graphs. Basically, it  manages data partitioning, scheduling, and parallelism in dataflow-based deep learning frameworks in order to achieve better performance for GNN training. 
* [Deep graph library: Towards efficient and scalable deep learning on graphs](https://arxiv.org/abs/1909.01315)
* [PCGCN: Partition-Centric Processing for Accelerating Graph Convolutional Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139807)
* [Reducing Communication in Graph Neural Network Training](https://arxiv.org/abs/2005.03300)

#### Neural Architecture Search 

* [Once-for-All: Train One Network and Specialize it for Efficient Deployment ](https://openreview.net/forum?id=HylxE1HKwS)
* [Fast Neural Network Adaptation via Parameter Remapping and Architecture Search ](https://openreview.net/forum?id=rklTmyBKPH)
* [Breaking the Curse of Space Explosion: Towards Efficient NAS with Curriculum Search](https://icml.cc/virtual/2020/poster/5803) 


#### Continous Learning  

* [Continual learning with hypernetworks ](https://openreview.net/forum?id=SJgwNerKvB)
* [Continual Learning with Adaptive Weights (CLAW) ](https://openreview.net/forum?id=Hklso24Kwr) 
* [Scalable and Order-robust Continual Learning with Additive Parameter Decomposition ](https://openreview.net/forum?id=r1gdj2EKPB)


### Efficient Inference <a name="inference"></a>

#### Resource Management 
* [GRNN: Low-Latency and Scalable RNN Inference on GPUs](https://dl.acm.org/doi/10.1145/3302424.3303949) The paper improves the performance of RNN inference by providing a GPU-based RNN inference library, called GRNN, that provides low latency, high throughput, and efficient resource utilization.
* [μLayer: Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization](https://dl.acm.org/doi/10.1145/3302424.3303950) μLayer is a low latency on-device inference runtime that significantly improves the latency of NN-assisted services. μLayer accelerates each NN layer by simultaneously utilizing diverse heterogeneous processors on a mobile device and by performing computations using processor-friendly quantization. First, to accelerate an NN layer using both the CPU and the GPU at the same time, μLayer employs a layer distribution mechanism which completely removes redundant computations between the processors. Next, μLayer optimizes the per-processor performance by making the processors utilize different data types that maximize their utilization. In addition, to minimize potential latency increases due to overly aggressive workload distribution, μLayer selectively increases the distribution granularity to divergent layer paths.
* [DeepEye: Resource Efficient Local Execution of Multiple Deep Vision Models using Wearable Commodity Hardware](https://dl.acm.org/doi/pdf/10.1145/3081333.3081359)
* [Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization](https://dl.acm.org/doi/abs/10.1145/3386901.3388947)

#### Model Design 
* [PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination](https://icml.cc/virtual/2020/poster/6835) 
* [Train Big, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](https://icml.cc/virtual/2020/poster/6828) 
* [Boosting Deep Neural Network Efficiency with Dual-Module Inference](https://icml.cc/virtual/2020/poster/6670)


#### Compression 

* [BlockSwap: Fisher-guided Block Substitution for Network Compression on a Budget](https://openreview.net/forum?id=SklkDkSFPB)
* [Probabilistic Connection Importance Inference and Lossless Compression of Deep Neural Networks ](https://openreview.net/forum?id=HJgCF0VFwr)
* [Scalable Model Compression by Entropy Penalized Reparameterization ](https://openreview.net/forum?id=HkgxW0EYDS) 
* [Compression based bound for non-compressed network: unified generalization error analysis of large compressible deep neural network ](https://openreview.net/forum?id=ByeGzlrKwH) 

#### Pruning 

* [What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/papers/2020/73) This paper provides an overview of approaches to pruning. The finding is that the community sufers from a lack of standardized benchmarks and metrics. It is hard to compare pruning techniques to one another or determine the progress the filed has made. The paper also introduce ShrinkBench, an open-source framework to faciliate standardized evaluations of pruning methods. 
* [A Signal Propagation Perspective for Pruning Neural Networks at Initialization ](https://openreview.net/forum?id=HJeTo2VFwH) 
* [DropNet: Reducing Neural Network Complexity via Iterative Pruning](https://icml.cc/virtual/2020/poster/6092)
* [Network Pruning by Greedy Subnetwork Selection](https://icml.cc/virtual/2020/poster/6053)
* [Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://openreview.net/forum?id=S1gSj0NKvB) 
* [Lookahead: A Far-sighted Alternative of Magnitude-based Pruning ](https://openreview.net/forum?id=ryl3ygHYDB)
* [Provable Filter Pruning for Efficient Neural Networks ](https://openreview.net/forum?id=BJxkOlSYDH)
* [Dynamic Model Pruning with Feedback ](https://openreview.net/forum?id=SJem8lSFwB)
* [One-Shot Pruning of Recurrent Neural Networks by Jacobian Spectrum Evaluation ](https://openreview.net/forum?id=r1e9GCNKvH)
* [PENNI: Pruned Kernel Sharing for Efficient CNN Inference](https://icml.cc/virtual/2020/poster/6232)
* [Operation-Aware Soft Channel Pruning using Differentiable Masks](https://icml.cc/virtual/2020/poster/5997) 


#### Quantization 
* [Towards Accurate Post-training Network Quantization via Bit-Split and Stitching](https://icml.cc/virtual/2020/poster/5787)
* [Differentiable Product Quantization for Learning Compact Embedding Layers](https://icml.cc/virtual/2020/poster/6429) 
* [Online Learned Continual Compression with Adaptive Quantization Modules](https://icml.cc/virtual/2020/poster/6338) 
* [Multi-Precision Policy Enforced Training (MuPPET) : A Precision-Switching Strategy for Quantised Fixed-Point Training of CNNs](https://icml.cc/virtual/2020/poster/6250)
* [Divide and Conquer: Leveraging Intermediate Feature Representations for Quantized Training of Neural Networks](https://icml.cc/virtual/2020/poster/6676) 
* [Up or Down? Adaptive Rounding for Post-Training Quantization](https://icml.cc/virtual/2020/poster/6482)
* [And the Bit Goes Down: Revisiting the Quantization of Neural Networks ](https://openreview.net/forum?id=rJehVyrKwH)
* [Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware ](https://openreview.net/forum?id=H1lBj2VFPS) 
* [AutoQ: Automated Kernel-Wise Neural Network Quantization ](https://openreview.net/forum?id=rygfnn4twS)
* [Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks ](https://openreview.net/forum?id=BkgXT24tDS)
* [Shifted and Squeezed 8-bit Floating Point format for Low-Precision Training of Deep Neural Networks ](https://openreview.net/forum?id=Bkxe2AVtPS)
* [Mixed Precision DNNs: All you need is a good parametrization ](https://openreview.net/forum?id=Hyx0slrFvH)
* [Riptide: Fast End-to-End Binarized Neural Networks](https://proceedings.mlsys.org/papers/2020/155)
This paper tries to speedup the implementation of binarized neural networks on Raspberry Pi 3B (RPi). 
* [Memory-Driven Mixed Low Precision Quantization for Enabling Deep Network Inference on Microcontrollers](https://proceedings.mlsys.org/papers/2020/130) The paper studies mixed low-bitwidth compression, featuring 8, 4 or 2-bit uniform quantization to enable integer-only operations on microcontrollers. It determines the minimum bit precision of every activation and weight tensor given the memory constraints of a device. The authors evaluate their approach using MobileNetV1 on a STM32H7 microcontroller. 


#### Model Serving 

* [PRETZEL: Opening the Black Box of Machine Learning Prediction Serving Systems](https://www.usenix.org/system/files/osdi18-lee.pdf): PRETZEL is  a  prediction  serving  system  introducing  anovel white box architecture enabling both end-to-endand multi-model optimizations. PRETZELis on average able to reduce 99th percentile la-tency by 5.5× while reducing memory footprint by 25×, and increasing throughput by 4.7×.
* [Parity models: erasure-coded resilience for prediction serving systems](https://dl.acm.org/doi/pdf/10.1145/3341301.3359654)
Improving the performance of prediction serving system that could take in queries and return
predictions by performing inference on models. This paper means to mitigate tail latency inflation. 
* [MArk: Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving](https://www.usenix.org/conference/atc19/presentation/zhang-chengliang)
The work focuses on the improvement of performance for ML-as-a-Service:: developers train ML models and publish them in the cloud as online services to provide low-latency inference at scale.  



### Testing and Debugging <a name="debugging"></a>
* [Model Assertions for Monitoring and Improving ML Models](https://proceedings.mlsys.org/papers/2020/189) The paper tries to monitor and improve ML models by using model assertions at all stages of ML system delopyment, including runtime monitoring and validating labels.  For runtime monitoring, model assertions can find high confidence errors. For training, they propose a bandit-based active learning algorithm that can sample from data flagged by assertion to reduce labeling cost. 




### Robustness <a name="robustness"></a>

* [Proving Data-Poisoning Robustness in Decision Trees](https://dl.acm.org/doi/abs/10.1145/3385412.3385975) This paper studies the robustness of decision tree models to the data poinsoning. It proposes a system that could still guarantee the correctness of model, even if the data set has been tempered. 
* [DNNGuard: An Elastic Heterogeneous DNN Accelerator Architecture against Adversarial Attacks](https://dl.acm.org/doi/abs/10.1145/3373376.3378532) The paper proposes an elastic heterogeneous DNN accelerator architec-ture that can efficiently orchestrate the simultaneous execu-tion of original (target) DNN networks and thedetectalgo-rithm or network that detects adversary sample attacks. 
* [Proper Network Interpretability Helps Adversarial Robustness in Classification](https://icml.cc/virtual/2020/poster/6031)
* [More Data Can Expand The Generalization Gap Between Adversarially Robust and Standard Models](https://icml.cc/virtual/2020/poster/5943)
* [Dual-Path Distillation: A Unified Framework to Improve Black-Box Attacks](https://icml.cc/virtual/2020/poster/6318)
* [Defense Through Diverse Directions](https://icml.cc/virtual/2020/poster/6557)
* [Adversarial Robustness Against the Union of Multiple Threat Models](https://icml.cc/virtual/2020/poster/6411)
* [Second-Order Provable Defenses against Adversarial Attacks](https://icml.cc/virtual/2020/poster/6256)
* [Understanding and Mitigating the Tradeoff between Robustness and Accuracy](https://icml.cc/virtual/2020/poster/6801) 
* [Adversarial Robustness via Runtime Masking and Cleansing](https://icml.cc/virtual/2020/poster/5817)
* [Adversarial Training and Provable Defenses: Bridging the Gap ](https://openreview.net/forum?id=SJxSDxrKDr)
* [Defending Against Physically Realizable Attacks on Image Classification ](https://openreview.net/forum?id=H1xscnEKDr) 
* [Enhancing Adversarial Defense by k-Winners-Take-All ](https://openreview.net/forum?id=Skgvy64tvr)
* [Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks ](https://openreview.net/forum?id=ByxtC2VtPB)
* [Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness ](https://openreview.net/forum?id=Byg9A24tvB)
* [Robust Local Features for Improving the Generalization of Adversarial Training](https://openreview.net/forum?id=H1lZJpVFvr) 
* [GAT: Generative Adversarial Training for Adversarial Example Detection and Robust Classification ](https://openreview.net/forum?id=SJeQEp4YDH)
* [Robust training with ensemble consensus ](https://openreview.net/forum?id=ryxOUTVYDH)
* [Fast is better than free: Revisiting adversarial training ](https://openreview.net/forum?id=BJx040EFvH)
* [EMPIR: Ensembles of Mixed Precision Deep Networks for Increased Robustness Against Adversarial Attacks ](https://openreview.net/forum?id=HJem3yHKwH)
* [Triple Wins: Boosting Accuracy, Robustness and Efficiency Together by Enabling Input-Adaptive Inference ](https://openreview.net/forum?id=rJgzzJHtDB)

### Other Metrics (Interpretability, Privacy, etc.) <a name="other-metrics"></a>

* [Privacy-Preserving Bandits](https://proceedings.mlsys.org/papers/2020/136)  The paper tries to enable privacy in personalized recommendation. This paper proposes a technique Privacy-Preserving Bandits (P2B); a system that updates local agents by collecting feedback from other local agents in a differentially-private manner. 
* [Shredder: Learning Noise Distributions to Protect Inference Privacy](https://dl.acm.org/doi/pdf/10.1145/3373376.3378522)
The work focuses on the cloud-based inference. It introduces the noise to the input data, but without sacrificing the accuracy of inference. 
* [DeepSniffer: A DNN Model Extraction Framework Based on Learning Architectural Hints. 385-399](https://dl.acm.org/doi/abs/10.1145/3373376.3378460) DeepSniffer extracts the model architecture information by learning the relation between extracted architectural hints (e.g., volumes of memory reads/writes obtained by side-channel or bus snooping attacks) and model internal architectures.



### Data Preparation <a name="data"></a>

* [Attention-based Learning for Missing Data Imputation](https://proceedings.mlsys.org/papers/2020/123) The paper focuses on mixed (discrete and continuous) data and proposes AimNet, an attention-based learning network for missing data imputation in HoloClean, a state-of-the-art ML-based data cleaning framework. The paper argues that  attention should be a central component in deep learning architectures for data imputation.  
* [Can gradient clipping mitigate label noise? ](https://openreview.net/forum?id=rklB76EKPr)
* [SELF: Learning to Filter Noisy Labels with Self-Ensembling ](https://openreview.net/forum?id=HkgsPhNYPS)

### ML programming models <a name="pl-models"></a>

* [Sense & Sensitivities: The Path to General-Purpose Algorithmic Differentiation](https://proceedings.mlsys.org/papers/2020/16) present Zygote, an algorithmic differentiation (AD) system for the Julia language. 


## Machine Learning for Systems <a name="ml4sys"></a>

### ML for programming languages <a name="pl"></a>

* [Learning Nonlinear Loop Invariants with Gated Continuous Logic Networks](https://dl.acm.org/doi/abs/10.1145/3385412.3385986) The paper proposes a new neuro architecture (Gated Continuous Logic Network(G-CLN)) to learn nonlinear loop invariants. Utilizing DNN to solve and understand the system issue. 

* [Blended, Precise Semantic Program Embeddings](https://dl.acm.org/doi/abs/10.1145/3385412.3385999) This paper is utilizing ML for systems. Basically, it utilizes DNN to learn program embeddings, vector representations of pro-gram semantics. Existing approaches predominately learn to embed programs from their source code, and, as a result, they do not capture deep, precise program semantics. On the other hand, models learned from runtime information critically depend on the quality of program executions, thus leading to trainedmodels with highly variant quality. LiGer learns programrepresentations from a mixture of symbolic and concrete exe-cution traces.


### ML for memory management <a name="mm"></a>

* [An Imitation Learning Approach for Cache Replacement](https://icml.cc/virtual/2020/poster/6044)
* [Learning-based Memory Allocation for C++ Server Workloads](https://dl.acm.org/doi/pdf/10.1145/3373376.3378525)
