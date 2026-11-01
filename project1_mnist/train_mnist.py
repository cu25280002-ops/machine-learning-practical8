"""Project 1: Handwritten digit recognition using a TensorFlow/Keras DNN."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EPOCHS = 8
BATCH_SIZE = 128
RANDOM_SEED = 42

def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("Loading the MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixels and flatten each 28x28 image into 784 features.
    x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
    x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(10, activation="softmax")
    ], name="MNIST_DNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    model_path = OUTPUT_DIR / "mnist_dnn.keras"
    model.save(model_path)

    pd.DataFrame([{
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    }]).to_csv(OUTPUT_DIR / "mnist_metrics.csv", index=False)

    report = classification_report(
        y_test, predictions, labels=list(range(10)),
        output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "mnist_classification_report.csv", index=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("MNIST Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Training loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_title("MNIST Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mnist_learning_curves.png", dpi=150)
    plt.close(fig)

    print("\\nProject 1 completed.")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Saved model: {model_path}")

if __name__ == "__main__":
    main()
