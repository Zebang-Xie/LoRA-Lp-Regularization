# Automatic Rank Determination for Low-Rank Adaptation via Structured L_p Regularization


Just done the initial experiment which verify the algorithm can induce sparsity to the LoRA BA matrix.
The initial experiment were conducted using a small-scale GPT-2 model on the E2E dataset, with parameters set to $M=10$, $\lambda \in [0.1, 1.0]$, and $q=0.5$. The procedure involved an initial 1,000 steps using the original model, followed by a switch to the custom_optimiser for an additional 10 steps.

<img width="490" height="280" alt="image" src="https://github.com/user-attachments/assets/ce93436b-b5ff-45d3-a022-8a990ccd2d17" />

<img width="846" height="177" alt="image" src="https://github.com/user-attachments/assets/42315a92-b4de-4a18-a768-322927893ed4" />








Project Data Flow Diagram: The brown sections represent the modifications, while the remaining parts belong to the original LoRA-GPT2 project (sourced from https://github.com/microsoft/LoRA).
<img width="2593" height="2461" alt="LoRA GPT2-FT" src="https://github.com/user-attachments/assets/d689fb80-db01-4e25-a7b9-66f952a8b91b" />
