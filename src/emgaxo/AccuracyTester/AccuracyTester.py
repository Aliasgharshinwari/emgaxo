import os
import time
import platform
import numpy as np
import onnxruntime as ort
from onnx import mapping
from onnx.onnx_pb import OperatorSetIdProto
#from onnxscript import opset11 as op
from sklearn.metrics import f1_score, precision_score, recall_score

def quantize_linear_np(x_float, scale, zero_point, dtype=np.int8):
    q = np.round(x_float / scale).astype(np.int64)
    q = q + zero_point
    info = np.iinfo(dtype)
    q = np.clip(q, info.min, info.max)
    return q.astype(dtype)

def check_accuracy(model, use_custom_ops=False, custom_domain='test.customop', results_file_name='model_results', x_test=[], y_test=[], batch_size=1):
    """
    Function to evaluate an ONNX model's classification performance.

    Parameters:
        model: Loaded ONNX model object to be evaluated.
        use_custom_ops (bool): Whether to register and use custom ONNX operators.
        custom_domain (str): Custom operator domain name (used if use_custom_ops=True).
        results_file_name (str): Name of the output file to save inference results.
        x_test (list or np.ndarray): Input test samples.
        y_test (list or np.ndarray): Ground truth labels for test samples.
        batch_size (int): Number of samples to process in a single inference batch.

    Returns:
        accuracy (float): Ratio of correct predictions to total samples.
        precision (float): Ratio of true positives to predicted positives (TP / (TP + FP)).
        recall (float): Ratio of true positives to actual positives (TP / (TP + FN)).
        f1_score (float): Harmonic mean of precision and recall.
    """

    # Register custom opset if required
    if use_custom_ops:
        custom_opset = OperatorSetIdProto()
        custom_opset.domain = custom_domain
        custom_opset.version = 1

        domain_exists = False
        for opset in model.opset_import:
            if opset.domain == custom_domain:
                domain_exists = True
                opset.version = custom_opset.version

        if not domain_exists:
            model.opset_import.append(custom_opset)

    # Set up ONNX Runtime session options
    so = ort.SessionOptions()
    if use_custom_ops:
        custom_domain == 'test.customop'
        #custom_op_library_path = "../CustomOpLib/libcustom_op_library.so"
        custom_op_library_path = "libcustom_op_library.so
        so.register_custom_ops_library(custom_op_library_path)

    # Serialize the ONNX model
    ser = model.SerializeToString()

        # Choose execution providers
    available_providers = ort.get_available_providers()
    #print("Available providers:", available_providers)

    # Choose execution providers
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']

    # Create inference session
    session = ort.InferenceSession(ser, so, providers=providers)

    # Extract input info
    first_input = model.graph.input[0]
    input_name = first_input.name
    input_shape = [dim.dim_value if dim.dim_value != 0 else None for dim in first_input.type.tensor_type.shape.dim]
    input_tensor_type = first_input.type.tensor_type.elem_type
    input_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[input_tensor_type]
    #print(f"Model expects input type: {input_dtype}, shape: {input_shape}")

    # Normalize or cast input based on expected dtype
   
    if np.issubdtype(input_dtype, np.floating):
        x_test = x_test.astype(np.float32) 
    elif np.issubdtype(input_dtype, np.int8):
        x_test = x_test.astype(np.int8) 

    else:
        x_test = x_test.astype(np.uint8)

    
    # Reshape input if necessary
    if len(input_shape) == 2 and input_shape[1] == 784:
        x_test = x_test.reshape(x_test.shape[0], -1)

    # Run inference
    predicted_labels = []
    batch_size = batch_size
    total_time = 0.0

    os.makedirs("./Results", exist_ok=True)
    with open(f"./Results/{results_file_name}.txt", 'w') as file:
        for i in range(0, len(x_test), batch_size):
            batch = x_test[i:i + batch_size]

            start_time = time.time()
            output = session.run([session.get_outputs()[0].name], {input_name: batch})
            end_time = time.time()

            logits = output[0]
            output_dtype = logits.dtype
            num_classes = logits.shape[1] if logits.ndim > 1 else 1

            if num_classes == 1:
                # Binary classification
                if np.issubdtype(output_dtype, np.floating):
                    threshold = 0.5
                elif np.issubdtype(output_dtype, np.integer):
                    threshold = 127
                else:
                    raise ValueError(f"Unsupported output dtype: {output_dtype}")

                predicted_batch = (logits > threshold).astype(int).flatten()

            else:
                # Multi-class classification
                predicted_batch = np.argmax(logits, axis=1)

            predicted_labels.extend(predicted_batch)
            total_time += (end_time - start_time)
            #print("predicted_batch", predicted_batch)
            #print("predicted_labels", predicted_labels)
            # Optional logging
            # print(logits)
            # file.write(f"{logits} Label:{y_test[i]} Predicted:{predicted_batch}\n")



    predicted_labels = np.array(predicted_labels)
    accuracy = np.mean(predicted_labels == y_test)
 
    # New metrics
    is_binary = len(np.unique(y_test)) == 2
    average_method = 'binary' if is_binary else 'weighted'

    precision = precision_score(y_test, predicted_labels, average=average_method, zero_division=0)
    recall = recall_score(y_test, predicted_labels, average=average_method)
    f1 = f1_score(y_test, predicted_labels, average=average_method)

    #print(f"Accuracy: {accuracy*100:.2f}%")
    #print(f"Precision: {precision:.4f}")
    #print(f"Recall: {recall:.4f}")
    #print(f"F1 Score: {f1:.4f}")

    avg_inference_time = total_time / len(predicted_labels)
    #print(f"Average inference time: {avg_inference_time*1000:.4f} ms")
    #print(f"Total execution time for inference: {total_time*1000:.4f} ms")
    #print(f"Accuracy: {accuracy*100:.2f}%")

    return accuracy, precision, recall, f1
