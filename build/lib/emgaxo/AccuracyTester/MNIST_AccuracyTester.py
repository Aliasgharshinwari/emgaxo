import time
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from onnx import OperatorSetIdProto
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef, make_onnx_model, 
    get_library_path as _get_library_path)
import os

def check_accuracy(model, use_custom_ops=False, custom_domain='test.customop', results_file_name = 'model_results'):
    """
    Function to check the accuracy of an ONNX model.

    Parameters:
        model: ONNX model to be evaluated.
        use_custom_ops: Boolean indicating whether to register custom operators.
        custom_domain: String representing the custom operator domain.

    Returns:
        accuracy: Accuracy of the model on the test dataset.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize the input data to [0, 1] range
    x_test = x_test.astype(np.float32) / 255.0

    # Register custom opset if required
    if use_custom_ops:
        custom_opset = OperatorSetIdProto()
        custom_opset.domain = custom_domain  # Custom operator domain
        custom_opset.version = 1  # Custom opset version

        # Check if custom domain opset is already present
        domain_exists = False
        for opset in model.opset_import:
            if opset.domain == custom_domain:
                domain_exists = True
                opset.version = custom_opset.version  # Ensure version consistency

        # Add custom domain opset if not present
        if not domain_exists:
            model.opset_import.append(custom_opset)

    # Set up ONNX Runtime session options
    so = ort.SessionOptions()
    if use_custom_ops and custom_domain == 'ai.onnx.contrib':
        so.register_custom_ops_library(_get_library_path())  # Provide the shared library path
    elif use_custom_ops and custom_domain == 'test.customop':
        so.register_custom_ops_library("D:/GitHUb/onnxruntime/build/Windows/Release/Release/custom_op_library.dll")  # Path to your shared library

    # Serialize the ONNX model
    ser = model.SerializeToString()

    # Create an inference session
    session = ort.InferenceSession(ser, so, providers=['CPUExecutionProvider'])

    # Get the first input from the graph
    first_input = model.graph.input[0]
    # Extract the input name
    input_name = first_input.name
    # Extract the input shape
    input_shape = [dim.dim_value if dim.dim_value != 0 else None for dim in first_input.type.tensor_type.shape.dim]

    # Print the input shape
    print(f"Input Shape: {input_shape}")

    # Reshape x_test based on input_shape
    if len(input_shape) == 2 and input_shape[1] == 784:  # Model expects flattened input
        x_test = x_test.reshape(x_test.shape[0], -1)  # Reshape to (N, 784)

    # Run the ONNX model on the test set
    predicted_labels = []

    # Define batch size and initialize total inference time
    batch_size = 1
    total_time = 0

    # Iterate through the test dataset in batches
    # Ensure the Results directory exists
    os.makedirs("./Results", exist_ok=True)
    with open(f"./Results/{results_file_name}.txt", 'w') as file:
        for i in range(0, len(x_test), batch_size):
            batch = x_test[i:i + batch_size]

            # Measure inference time
            start_time = time.time()
            output = session.run([session.get_outputs()[0].name], {input_name: batch})
            end_time = time.time()

            # Calculate batch inference time
            batch_time = end_time - start_time
            #file.write(f"Inference time for iteration {i // batch_size + 1} = {batch_time * 1000:.6f} ms\n")

            total_time += batch_time

            # Extract predictions
            predicted_batch = np.argmax(output[0], axis=1)  # Adjust if output format differs
            predicted_labels.extend(predicted_batch)

        # Convert predictions to a numpy array
        predicted_labels = np.array(predicted_labels)

        # Calculate accuracy
        accuracy = np.mean(predicted_labels == y_test)
        
        # Calculate average inference time per sample
        avg_inference_time = total_time / len(predicted_labels)
        
        # Print total execution time
        file.write(f"Total execution time for inference: {total_time*1000:.4f} ms\n")
        file.write(f"Average inference time: {avg_inference_time*1000:.4f} ms\n")
        file.write(f"Accuracy: {accuracy*100:.4f}%\n")

    return accuracy
