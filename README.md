# Hybrid Source Code Representation Learning Framework
### Based on: *"A systematic mapping study of source code representation for deep learning in software engineering"* by Samoaa et al. (2022)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📖 Overview & Theoretical Grounding

Deep learning models for Software Engineering (SE) tasks rely critically on how source code is abstracted and represented. In their systematic mapping study (*IET Software*, 2022), **Hazem Peter Samoaa, Firas Bayram, Pasquale Salza, and Philipp Leitner** synthesized over a decade of research into source code representation. 

This repository provides a **complete, self-contained, and Colab-safe implementation** of a **Multi-View Hybrid Representation Learning Framework** integrating:

| Representation View | Paper Section | Modality Characteristics | Deep Learning Architecture |
| :--- | :--- | :--- | :--- |
| **Token-Based** | Section 2.1 & 5.1 | Lexical streams, sub-tokens (camelCase/snake_case), keywords, operators | Bidirectional LSTM with Attention Pooling |
| **Tree-Based (AST)** | Section 2.2 & 5.2 | Grammatical hierarchy, syntax nesting without punctuation clutter via Structure-Based Traversal (SBT) | AST-BiLSTM with Multi-Scale 1D Convolutions |
| **Graph-Based (CFG/DFG)** | Section 2.3 & 5.3 | Non-sequential Control Flow (branches/loops) and Data Flow (variable def-use chains) | Graph Convolutional Network (GCN) with global readout |
| **Multi-View Fusion** | Section 8 | Attention-weighted integration over representations | Cross-Modality Attention & Projection Layer |
| **Downstream SE Task** | Section 4 & 6 | Code Clone Detection (Type 1–4 Clones vs. Non-Clones) | Siamese Neural Network with comparison interaction vectors |

---

## 🚀 Key Features

1. **Colab-Safe Memory Architecture**:
   - Explicit garbage collection (`gc.collect()`) and CUDA cache eviction (`torch.cuda.empty_cache()`) per epoch.
   - Bounded sequences, batch size controls, and vectorized PyTorch GCN layers that eliminate binary wheel installation issues in Colab.
2. **Four-Clone Benchmark Support**:
   - Evaluates **Type-1** (identical syntax), **Type-2** (renamed variables/identifiers), **Type-3** (modified/reordered statements), and **Type-4** (semantic clones with completely different syntax, e.g. recursion vs. iteration).
3. **Cross-Modality Attention Inspection**:
   - The model dynamically outputs attention weights $\alpha = [\alpha_{tok}, \alpha_{tree}, \alpha_{graph}]$ showing how much each view influenced the clone decision for any input pair.
4. **Zero-Dependency Vectorized GCN**:
   - Graph Convolution ($H' = \tilde{A}_{norm} H W$) is implemented directly in native PyTorch tensors, eliminating PyTorch Geometric compilation errors while remaining 100% faithful to Kipf & Welling's GCN formulation.

---

## 📁 Repository Structure

```
.
├── hybrid_code_representation.ipynb   # Complete, ready-to-run Google Colab Jupyter Notebook
├── hybrid_code_representation.py      # Modular, self-contained Python script for local execution
├── generate_notebook.py               # Automated script to regenerate the .ipynb notebook
├── training_metrics.png               # Loss & Accuracy/Precision/Recall/F1 performance curves
└── README.md                          # Documentation and paper mapping guide
```

---

## 💻 How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `hybrid_code_representation.ipynb`.
3. In Colab, navigate to **Runtime > Change runtime type** and ensure **T4 GPU** (or CPU) is selected.
4. Run all cells (`Runtime > Run all` or `Ctrl + F9`).
5. All dependencies (`torch`, `numpy`, `scikit-learn`, `matplotlib`) are automatically installed in the first code cell.

---

## 🛠️ How to Run Locally

### Using `uv` (Recommended - fast & isolated)
```bash
uv run --python 3.11 --with torch --with scikit-learn --with matplotlib python hybrid_code_representation.py
```

### Using standard `pip`
```bash
pip install torch numpy scikit-learn matplotlib
python hybrid_code_representation.py
```

---

## 🔬 Multi-View Feature Extraction Pipeline

### 1. Lexical Tokenizer (`CodeTokenizer`)
- Strips code comments and normalizes formatting.
- Decomposes compound identifiers into constituent sub-tokens:
  $$\text{split\_identifier}(\text{"binarySearch"}) \rightarrow [\text{"binary"}, \text{"search"}]$$
- Normalizes integer and float literals to `<NUM>` to prevent vocabulary explosion.
- Encodes token sequences bounded by `<BOS>` and `<EOS>` with `<PAD>` zero-padding.

### 2. AST Structure-Based Traversal (`ASTParser`)
- Parses code into an Abstract Syntax Tree using Python's `ast` parser.
- Linearizes the tree using Structure-Based Traversal (SBT) as formulated by Hu et al. (2018):
  $$\text{SBT}(N) = \text{‘(’}N, \text{SBT}(C_1), \text{SBT}(C_2), \dots, \text{‘)’}N$$
- Uniquely represents parent-child tree hierarchy without depth loss, preserving syntactic relationships for sequential Bi-LSTM encoders.

### 3. Program Graph Construction (`ProgramGraphBuilder`)
- **Nodes**: Statements and control anchors (`Entry`, `FunctionDef`, `Assign`, `For`, `While`, `If`, `Return`, `Exit`).
- **Control Flow Edges (CFG)**: Sequential transitions, loop headers, loop back-edges, and conditional branches.
- **Data Flow Edges (DFG)**: Variable Def-Use associations (connecting statement $i$ where variable $x$ is assigned to statement $j$ where variable $x$ is read).
- **Adjacency Normalization**:
  $$\tilde{A} = A + I_N, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}, \quad A_{norm} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

---

## 🧠 Neural Architecture & Multi-View Fusion

```
Input Snippet A                               Input Snippet B
      │                                             │
      ├───────────┬───────────┐                     ├───────────┬───────────┐
      ▼           ▼           ▼                     ▼           ▼           ▼
  [Tokens]      [AST]     [CFG/DFG]             [Tokens]      [AST]     [CFG/DFG]
      │           │           │                     │           │           │
      ▼           ▼           ▼                     ▼           ▼           ▼
  Bi-LSTM     AST-LSTM       GCN                Bi-LSTM     AST-LSTM       GCN
   h_tok       h_tree       h_graph              h_tok       h_tree       h_graph
      │           │           │                     │           │           │
      └───────────┼───────────┘                     └───────────┼───────────┘
                  ▼                                             ▼
        [Attention Fusion]                            [Attention Fusion]
               h_A                                           h_B
                  │                                             │
                  └──────────────────────┬──────────────────────┘
                                         ▼
                             Comparison Features:
                 f = [h_A, h_B, |h_A - h_B|, h_A ⊙ h_B]
                                         │
                                         ▼
                            Classification Head (MLP)
                                         │
                                         ▼
                         P(is_clone | c_A, c_B) ∈ [0, 1]
```

### Siamese Interaction Features
For two code embeddings $h_A, h_B \in \mathbb{R}^d$, the interaction vector is:
$$f = [h_A \,\|\, h_B \,\|\, |h_A - h_B| \,\|\, h_A \odot h_B] \in \mathbb{R}^{4d}$$
- $|h_A - h_B|$ captures element-wise distance in the latent semantic space.
- $h_A \odot h_B$ captures directional alignment (Hadamard product).

---

## 🧪 5 Real-World Prediction Test Cases & Outputs

The framework evaluates multiple clone types and outputs the clone probability alongside the **Modality Attention Breakdown** (Token vs. Tree vs. Graph):

### Test Case 1: Type-2 Clone (Renamed Variables in Bubble Sort)
- **Snippet A**: Standard `bubble_sort(arr)` with variables `i, j, n`.
- **Snippet B**: Renamed `sort_items(lst)` with variables `p, q, size`.
```
>>> Prediction:  CLONE (Matching Pair)
    Probability: 0.5964
    Modality Weights:
      • Token View (Names & Keywords): 46.7%
      • Tree View (AST Structure):     36.7%
      • Graph View (Execution Flow):   16.6%
```

### Test Case 2: Type-4 Semantic Clone (Factorial: Loop vs. Recursion)
- **Snippet A**: Iterative `factorial` with `for i in range(1, n + 1)`.
- **Snippet B**: Recursive `factorial` with `n * factorial(n - 1)`.
```
>>> Prediction:  CLONE (Matching Pair)
    Probability: 0.6394
    Modality Weights:
      • Token View (Names & Keywords): 39.5%
      • Tree View (AST Structure):     28.1%
      • Graph View (Execution Flow):   32.4%
```

### Test Case 3: Type-4 Semantic Clone (Fibonacci: Iterative vs. Recursive)
- **Snippet A**: Iterative `fibonacci` using accumulator loop.
- **Snippet B**: Recursive `fibonacci` using divide-and-conquer calls.
```
>>> Prediction:  CLONE (Matching Pair)
    Probability: 0.6385
    Modality Weights:
      • Token View (Names & Keywords): 41.5%
      • Tree View (AST Structure):     27.6%
      • Graph View (Execution Flow):   30.9%
```

### Test Case 4: Non-Clone (Factorial vs. Binary Search)
- **Snippet A**: Factorial computation.
- **Snippet B**: Binary search on a sorted array with `while l <= r`.
```
>>> Prediction:  NON-CLONE (Distinct Code)
    Probability: 0.1338
    Modality Weights:
      • Token View (Names & Keywords): 45.7%
      • Tree View (AST Structure):     29.4%
      • Graph View (Execution Flow):   24.9%
```

### Test Case 5: Non-Clone (Bubble Sort vs. Prime Number Checker)
- **Snippet A**: Quadratic Bubble Sort.
- **Snippet B**: Primality testing via trial division up to $\sqrt{n}$.
```
>>> Prediction:  NON-CLONE (Distinct Code)
    Probability: 0.0913
    Modality Weights:
      • Token View (Names & Keywords): 44.4%
      • Tree View (AST Structure):     30.4%
      • Graph View (Execution Flow):   25.2%
```

---

## 📚 Citation

If you use this framework in your research, please cite the survey paper:

```bibtex
@article{samoaa2022systematic,
  author    = {Samoaa, Hazem Peter and Bayram, Firas and Salza, Pasquale and Leitner, Philipp},
  title     = {A systematic mapping study of source code representation for deep learning in software engineering},
  journal   = {IET Software},
  volume    = {16},
  number    = {4},
  pages     = {351--385},
  year      = {2022},
  publisher = {Wiley Online Library},
  doi       = {10.1049/sfw2.12068}
}
```
