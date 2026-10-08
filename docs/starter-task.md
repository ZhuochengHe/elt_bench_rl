# Student Task: Build an End-to-End ELT-Bench RL Environment

Build a complete [Tinker](https://tinker-docs.thinkingmachines.ai/tinker/) [reinforcement learning with verifiable rewards (RLVR)](https://arxiv.org/abs/2402.03300) recipe for [ELT-Bench](https://arxiv.org/abs/2504.04808). It is recommended that you reuse [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) code as much as possible.

## Objective

Train a language model to solve ELT-Bench tasks end to end. In each rollout, the model must inspect the task specification, configure and run extraction/loading, implement the requested transformations, use execution feedback, and submit a final pipeline. The environment must grade the resulting warehouse state through execution and return a useful RL reward. The environment should support at least one data warehouse, while being destination-specific so that other data warehouses can be added without rewriting the environment.

Grading covers:

1. A credential-free local integration test
2. A credentialed rollout on at least one official ELT-Bench task
3. at least one Tinker RL optimization step using execution-derived rewards

## Suggested workflow and hints

**Step 1.** Understand the ELT task:

- What are the components of an ELT task?
- What are the prompts, tools, and terminal conditions for an LLM to complete an ELT task?
- How does the state of an ELT environment change? Is the environment stateless?

**Step 2\.** Define the reward function:

- What is the definition of a correct ELT pipeline?
- How can we verify the correctness of a submitted pipeline?
- In RLVR, the model is optimized using outcome rewards and/or process rewards. Which types of reward function can we build for ELT tasks? Why?

**Step 3\.** Generate a full implementation plan and build it.

**Step 4\.** Think about reward hacking: does the environment design and implementation have any loopholes that the model can leverage to achieve a high reward without completing a task?

**Step 5\.** Think about training efficiency: can we further optimize the environment to reduce latency during environment transition and/or reward computing?

## Submission

1. A Google Doc with a concise implementation design
2. A GitHub repository with a self-contained implementation and clear documentation
