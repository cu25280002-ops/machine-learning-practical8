"""Create a simple comparison table after both training scripts have run."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

mnist_file = OUT / "mnist_metrics.csv"
maze_file = OUT / "maze_metrics.csv"

missing = [str(p.name) for p in (mnist_file, maze_file) if not p.exists()]
if missing:
    raise SystemExit(
        "Run both training scripts first. Missing: " + ", ".join(missing)
    )

mnist = pd.read_csv(mnist_file).iloc[0]
maze = pd.read_csv(maze_file).iloc[0]

comparison = pd.DataFrame([
    {
        "project": "MNIST Digit Recognition",
        "learning_method": "Supervised Deep Learning (DNN)",
        "evaluation_measure": "Test accuracy",
        "value": float(mnist["test_accuracy"]),
        "interpretation": "Fraction of test images classified correctly"
    },
    {
        "project": "Maze Solver",
        "learning_method": "Reinforcement Learning (Q-Learning)",
        "evaluation_measure": "Evaluation success rate",
        "value": float(maze["success_rate"]),
        "interpretation": "Fraction of evaluation episodes reaching the goal"
    }
])
comparison.to_csv(OUT / "project_comparison.csv", index=False)
print(comparison.to_string(index=False))
print("\\nSaved outputs/project_comparison.csv")
print("Note: These values measure different tasks and are not directly comparable.")
