import tensorflow as tf
import numpy as np
import cv2

def grad_cam_safe(model, img_array, layer_name=None):
    """
    Safe Grad-CAM for any CNN model (ResNet, MobileNet, etc.)
    img_array must be shape (1, H, W, 3)
    """

    # 🔍 Auto-select last Conv2D layer if not given
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

        if layer_name is None:
            raise ValueError("❌ No Conv2D layer found in model")

    print(f"✅ Using layer: {layer_name}")

    # Create grad model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array, training=False)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Compute gradients
    grads = tape.gradient(loss, conv_output)

    # Global Average Pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1)

    # Apply ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()