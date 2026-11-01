# Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning

**MCA III Semester | Session 2026–2027**

This folder contains two runnable mini-projects:

1. **Handwritten Digit Recognition** — a Deep Neural Network (DNN) trained on MNIST using TensorFlow/Keras.
2. **Q-Learning Maze Solver** — a tabular Q-Learning agent that learns a route through a maze using Gymnasium.

The programs save trained models, evaluation results, and learning-curve plots in the `outputs/` folder.

## 1. Requirements

- Python 3.11 or newer
- Internet connection for the first MNIST download
- TensorFlow/Keras, NumPy, Pandas, Matplotlib, Gymnasium

## 2. Setup on Windows (VS Code)

Open this folder in VS Code. Open **Terminal → New Terminal**, then run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not available, use `python -m venv .venv` after installing Python 3.11+.

> TensorFlow installation can take some time. The first run of Project 1 also downloads the MNIST dataset.

## 3. Run Project 1 — MNIST Digit Recognition

```powershell
python project1_mnist/train_mnist.py
```

The script:
- Loads MNIST (60,000 training and 10,000 test images).
- Normalizes pixel values from 0–255 to 0–1.
- Builds a fully connected DNN with ReLU hidden layers and a Softmax output layer.
- Trains and evaluates the model.
- Saves the trained Keras model and evaluation files.

Expected output files:
- `outputs/mnist_dnn.keras` — trained model
- `outputs/mnist_learning_curves.png` — accuracy and loss curves
- `outputs/mnist_metrics.csv` — test loss and accuracy
- `outputs/mnist_classification_report.csv` — precision, recall and F1-score by digit

Optional settings can be changed at the top of `train_mnist.py` (epochs, batch size).

## 4. Run Project 2 — Q-Learning Maze Solver

```powershell
python project2_maze/q_learning_maze.py
```

The agent starts at the top-left cell and must reach the goal at the bottom-right cell. It learns using rewards:
- Reaching the goal: +100
- Hitting a wall or blocked cell: -10
- Each normal move: -1

The Q-table is updated using the Q-Learning update rule:

`Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]`

The script uses epsilon-greedy action selection (exploration vs. exploitation), evaluates the learned policy, and saves:
- `outputs/maze_q_table.npy` — trained Q-table
- `outputs/maze_model.json` — model settings and evaluation summary
- `outputs/maze_learning_curve.png` — episode reward curve
- `outputs/maze_learned_path.png` — route followed by the trained agent
- `outputs/maze_metrics.csv` — evaluation success rate and average steps

## 5. Compare the Projects

After running both projects, execute:

```powershell
python compare_results.py
```

This creates `outputs/project_comparison.csv`. The two projects solve different tasks, so their accuracy/success-rate values should not be treated as directly equivalent. The comparison file records each project's evaluation measure and explains what it means.

## 6. Suggested execution order

```powershell
python project1_mnist/train_mnist.py
python project2_maze/q_learning_maze.py
python compare_results.py
```

## 7. Important notes

- The ZIP contains the complete source code and setup instructions. Trained model files are generated on your computer when you run the training scripts; they are not pre-trained inside this ZIP.
- MNIST is downloaded automatically by Keras on the first run.
- For a quick test, reduce `EPOCHS` in `project1_mnist/train_mnist.py` or `EPISODES` in `project2_maze/q_learning_maze.py`. For final submission, use the default settings and keep the generated output files.
- Keep the `outputs/` folder when submitting your source code and trained model.

## 8. Viva preparation

1. **What is a neural network?** A model made of connected artificial neurons that learns patterns from data.
2. **What is a DNN?** A neural network containing more than one hidden layer.
3. **What is MNIST?** A dataset of 70,000 grayscale handwritten digit images (0–9), each 28×28 pixels.
4. **Why normalize image pixels?** Smaller input values help make neural-network training more stable.
5. **What is ReLU?** An activation function: `f(x) = max(0, x)`.
6. **Why use Softmax in the output layer?** It converts output scores into class probabilities.
7. **What is an epoch?** One complete pass through the training dataset.
8. **What is reinforcement learning?** A method where an agent learns actions by interacting with an environment and receiving rewards.
9. **What is Q-Learning?** A model-free reinforcement-learning algorithm that learns the value of taking an action in a state.
10. **What is the epsilon-greedy strategy?** It chooses a random action with probability epsilon; otherwise it chooses the action with the highest Q-value.
11. **What do alpha and gamma mean?** Alpha is the learning rate; gamma is the discount factor for future rewards.
12. **What is the difference between supervised and reinforcement learning?** Supervised learning learns from labelled examples, while reinforcement learning learns from rewards received after actions.
