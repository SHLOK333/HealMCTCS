# LLM-Verified with Monte Carlo Tree Search

## Overview

This project combines the power of **Large Language Models (LLMs)** with **Monte Carlo Tree Search (MCTS)** to provide verifiable, optimized, and adaptive responses across various reasoning tasks. Verification backends such as **Dafny** and **Coq** are integrated to ensure correctness of logic, while containerization and GPU acceleration power its performance at scale.


Main Goal of this project was to implement an AI model locally at you desk while it turned out to be why not code it health realted wise locally at your pc 
get personal recommendations and everything locally 
---

## Features

* **MCTS-Driven Reasoning**: Evaluates multiple LLM-generated answers, selects optimal verified response paths.
* **Custom LLM Dataset**: Over 6,800 curated samples built from healthline.com and augmented via scripting.
* **Formal Verification**:

  * **Dafny** for automated logic validation.
  * **Coq** for proof-backed correctness.
* **Multi-GPU Enabled**: Optimized for dual NVIDIA A100s with Modal Labs sandbox orchestration.
* **Docker/Singularity Ready**: Containerized setup for scalable, reproducible environments.
* **Rust Integration**: Planned for performance-critical components like tree simulation and concurrency.

---

## System Architecture
![Untitled](https://github.com/user-attachments/assets/f10b7ecf-e217-4fac-9e7e-45ba348d99ac)

Practical Guidline to setup a pipline between all LLms 

![image](https://github.com/user-attachments/assets/88cf7a95-6238-42e6-86dd-4e5803a3a9c8)


```plaintext
            +-----------------------+
            |    User Interface     |
            +----------+------------+
                       |
                       v
            +-----------------------+
            |     LLM Response      |
            |     Generation        |
            +----------+------------+
                       |
                       v
            +-----------------------+
            |  Monte Carlo Tree     |
            |  Search (MCTS) Engine |
            +----------+------------+
                       |
             +---------+--------+
             |                  |
             v                  v
    +----------------+   +----------------+
    | Dafny Verifier |   |   Coq Verifier |
    +----------------+   +----------------+
             |
             v
     +---------------------+
     | Verified Response   |
     +---------------------+
```

---

## Monte Carlo Tree Search (MCTS)

MCTS is a decision-making algorithm often used in reinforcement learning and game theory. It builds a search tree incrementally using four steps:

1. **Selection**: Select optimal child node based on a policy (e.g., UCT).
2. **Expansion**: Generate one or more new nodes.
3. **Simulation**: Simulate a rollout from the node.
4. **Backpropagation**: Update the value estimates of nodes in the path.

We apply MCTS to explore different LLM-generated outputs and rank them based on formal verification and semantic alignment.

![image](https://github.com/user-attachments/assets/3b729854-ce2e-4219-9819-85f3c198e9b9)

---

## Corpus and LLM Utility

* **6,800+ Custom Samples**
* **Source**: healthline.com + scripted augmentation
* **Augmentation**: Diversified question structure and context generation using Python
* **Benefit**: Enables LLM to handle different user querying styles effectively

---

## Infrastructure Setup

### Modal Sandbox Execution

```python
import modal

app = modal.App.lookup('my-sandbox-app', create_if_missing=True)
sb = modal.Sandbox.create(app=app)

p = sb.exec('echo', 'hello world')
print(p.stdout.read())

sb.terminate()
```

### Singularity (for HPC)

```bash
mkdir -p ~/singularity
cd ~/singularity
singularity pull docker://namin/llm-verified
```

### Docker (Alternative)

```bash
docker pull namin/llm-verified
# Run with: --docker_sandbox
```

---

## Formal Verification Tools

### Dafny

* Install from: [https://github.com/dafny-lang/dafny](https://github.com/dafny-lang/dafny)
* Z3 must be on `PATH`

```bash
python3 okdafny.py  # Run test verifications
```

### Coq

```bash
opam init
opam install coq
opam install "coq-serapi>=8.10.0+0.7.0"
opam repo add coq-released https://coq.inria.fr/opam/released
opam install coq-hammer
```

---

## Rust Integration (Planned)

Rust components will:

* Optimize memory usage for tree simulations
* Improve concurrency and parallel rollout processing
* Ensure safety and performance at scale using Rust’s guarantees

---

## Performance Metrics (Simulated)

| Component             | Metric                        | Value (A100 x2)   |
| --------------------- | ----------------------------- | ----------------- |
| LLM Inference Time    | Avg. response latency         | \~90 ms           |
| MCTS Tree Depth       | Avg. search depth             | 12–20 nodes       |
| Dafny Verification    | Time per proof                | \~120 ms          |
| Coq Verification      | Time per proof                | \~300 ms          |
| Modal Sandbox Runtime | Cold start                    | \~1.2s            |
| Docker Execution      | Total inference + verify loop | \~3.8s end-to-end |

> Note: Metrics will be updated with real-time benchmarks after full pipeline integration.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

