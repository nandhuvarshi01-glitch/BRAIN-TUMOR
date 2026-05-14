import matplotlib.pyplot as plt

def plot_history(history_resnet=None, history_mobilenet=None):

    plt.figure(figsize=(10,5))

    # ---------------- RESNET ----------------
    if history_resnet is not None:
        if 'accuracy' in history_resnet.history:
            plt.plot(history_resnet.history['accuracy'], label='ResNet Train')
        if 'val_accuracy' in history_resnet.history:
            plt.plot(history_resnet.history['val_accuracy'], label='ResNet Val')

    # ---------------- MOBILENET ----------------
    if history_mobilenet is not None:
        if 'accuracy' in history_mobilenet.history:
            plt.plot(history_mobilenet.history['accuracy'], label='MobileNet Train')
        if 'val_accuracy' in history_mobilenet.history:
            plt.plot(history_mobilenet.history['val_accuracy'], label='MobileNet Val')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# 🔥 CALL FUNCTION AFTER TRAINING
plot_history(history_resnet, history_mobilenet)