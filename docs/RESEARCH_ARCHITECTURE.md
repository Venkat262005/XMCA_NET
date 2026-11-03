# XMCA-Net: Cross-Modal Consistency Attribution Network
## Research Architecture Specification

### 1. Overview & Theoretical Framework

Multimodal Large Language Models (MLLMs / VLMs) frequently exhibit cross-modal hallucinations—generating claims that contradict image evidence, fabricated external factual assertions, or ungrounded prior hallucinations. Existing mitigation and detection methods typically reduce hallucination detection to a scalar confidence score, failing to disentangle the distinct failure modes underlying generation.

**XMCA-Net (Cross-Modal Consistency Attribution Network)** addresses this fundamental limitation by decomposing cross-modal consistency into three orthogonal diagnostic signals:
1. **Perception Consistency ($S_P$)**: Does the generated token sequence faithfully ground in the visual evidence?
2. **Retrieval Consistency ($S_R$)**: Does the model retrieve accurate factual representations from internal parametric memory and external knowledge sources?
3. **Utilization Consistency ($S_U$)**: Does the language decoder faithfully utilize the grounded perception and retrieved knowledge, rather than being dominated by language prior bias?

```
                     +---------------------------------------+
                     |         Visual Input + Prompt         |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |      VLM Backbone (LLaVA/Qwen/Intern)  |
                     +---------------------------------------+
                                         |
            +----------------------------+---------------------------+
            |                            |                           |
            v                            v                           v
+-----------------------+    +-----------------------+   +-----------------------+
| Perception Diagnostic |    | Retrieval Diagnostic  |   | Utilization Diagnostic|
|         (S_P)         |    |         (S_R)         |   |         (S_U)         |
+-----------------------+    +-----------------------+   +-----------------------+
            |                            |                           |
            +----------------------------+---------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |    Calibrated XMCA Detection Combiner |
                     +---------------------------------------+
                                         |
                        +----------------+---------------+
                        |                                |
                        v                                v
         [Hallucination Probability P(H)]   [Cause-Attribution Head]
                                            (Supervised ONLY where valid
                                             ground-truth labels exist)
```

---

### 2. Tri-Component Consistency Diagnostics

#### 2.1 Perception Consistency ($S_P$)
- **Scope**: Evaluates cross-attention and visual token alignment across the visual encoder and generative decoder layers.
- **Formulation**: $S_P \in [0, 1]$, where values approaching $1.0$ indicate strong visual grounding and $0.0$ indicates visual disconnect (perceptual hallucination).
- **Diagnostic Target**: Identifies object hallucination, attribute misbinding, and spatial inaccuracies.

#### 2.2 Retrieval Consistency ($S_R$)
- **Scope**: Assesses consistency between retrieved factual knowledge (internal entity representations or external corpus retrieval) and generated factual propositions.
- **Formulation**: $S_R \in [0, 1]$, measuring knowledge alignment and parametric coherence.
- **Diagnostic Target**: Identifies parametric memory corruption, ungrounded entity generation, and factual drift on knowledge-intensive benchmarks (InfoSeek, A-OKVQA).

#### 2.3 Utilization Consistency ($S_U$)
- **Scope**: Quantifies context fidelity against language prior bias. It measures whether the output conditioned on both multimodal context and retrieved tokens diverges from unconditioned language-model priors.
- **Formulation**: $S_U \in [0, 1]$, where high scores indicate faithful context utilization and low scores signify prior-dominated token generation.
- **Diagnostic Target**: Distinguishes between perceptual failures and language-model prior bias hallucinations.

---

### 3. Calibrated XMCA Detection Combiner

The detection combiner maps the consistency triplet $(S_P, S_R, S_U)$ to a calibrated hallucination probability:
$$\hat{y} = \sigma\left(\mathbf{W} \cdot [S_P, S_R, S_U]^\top + b\right)$$

- **Strict Calibration Requirement**: Evaluated using Expected Calibration Error (ECE) and Brier Score alongside standard discrimination metrics (AUROC, Precision-Recall AUC).
- **Prohibition of Hardcoded Proxies**: No arbitrary weighting, proxy heuristics, or synthetic probability assignments are permitted. Calibration parameters must be learned strictly on validation splits.

---

### 4. Independently Supervised Cause-Attribution Head

The cause-attribution head assigns fine-grained failure modes to detected hallucinations:
- **Class Labels**: `{Perception Failure, Retrieval Failure, Utilization Failure}`.
- **Supervision Constraint**: The attribution head is trained and evaluated **strictly on datasets with verified ground-truth attribution annotations**:
  - POPE: Evaluates perception-level object presence/absence.
  - FaithScore: Evaluates fine-grained atomic factual hallucination versus perceptual consistency.
- **Unlabeled Datasets**: On datasets without independent attribution ground truth (such as general VQA benchmarks), attribution outputs are uncoupled and never trained on synthetic or proxy pseudo-labels.

---

### 5. Strict Execution Integrity Rules

1. **Zero Dummy Values**: Every diagnostic module raises `NotImplementedError` until authentic feature extraction and computation pipelines are authorized.
2. **Deterministic Reproducibility**: Fixed random seeds, logged environment manifests, and pinned model revisions ensure 100% bit-exact experimental replicability.
3. **Dual T4 Kaggle Isolation**: All model executions and heavy tensor diagnostics run exclusively on dual NVIDIA T4 GPUs in Kaggle.
