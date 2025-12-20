import os
import time
import platform
import numpy as np
import onnxruntime as ort
from onnx import mapping
from onnx.onnx_pb import OperatorSetIdProto
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef, make_onnx_model, 
    get_library_path as _get_library_path)

import numpy as np

def _get_confusion_counts(y_true, y_pred, cls):
    """
    Helper to calculate TP, FP, FN for a specific class.
    """
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    return tp, fp, fn

def calculate_precision(y_true, y_pred):
    """
    Calculates Precision: TP / (TP + FP)
    Supports Binary and Weighted Multi-class.
    """
    classes = np.unique(y_true)
    
    # Binary Case
    if len(classes) == 2 and set(classes) <= {0, 1}:
        tp, fp, _ = _get_confusion_counts(y_true, y_pred, cls=1)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Weighted Multi-class Case
    precisions = []
    weights = []
    
    for cls in classes:
        tp, fp, _ = _get_confusion_counts(y_true, y_pred, cls)
        support = np.sum(y_true == cls)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(p)
        weights.append(support)
        
    total_support = np.sum(weights)
    return np.sum(np.array(precisions) * np.array(weights)) / total_support if total_support > 0 else 0.0

def calculate_recall(y_true, y_pred):
    """
    Calculates Recall: TP / (TP + FN)
    Supports Binary and Weighted Multi-class.
    """
    classes = np.unique(y_true)
    
    # Binary Case
    if len(classes) == 2 and set(classes) <= {0, 1}:
        tp, _, fn = _get_confusion_counts(y_true, y_pred, cls=1)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Weighted Multi-class Case
    recalls = []
    weights = []
    
    for cls in classes:
        tp, _, fn = _get_confusion_counts(y_true, y_pred, cls)
        support = np.sum(y_true == cls)
        
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(r)
        weights.append(support)
        
    total_support = np.sum(weights)
    return np.sum(np.array(recalls) * np.array(weights)) / total_support if total_support > 0 else 0.0

def calculate_f1_score(y_true, y_pred):
    """
    Calculates F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    Supports Binary and Weighted Multi-class.
    """
    classes = np.unique(y_true)
    
    # Binary Case
    if len(classes) == 2 and set(classes) <= {0, 1}:
        tp, fp, fn = _get_confusion_counts(y_true, y_pred, cls=1)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    # Weighted Multi-class Case
    f1s = []
    weights = []
    
    for cls in classes:
        tp, fp, fn = _get_confusion_counts(y_true, y_pred, cls)
        support = np.sum(y_true == cls)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        f1s.append(f)
        weights.append(support)
        
    total_support = np.sum(weights)
    return np.sum(np.array(f1s) * np.array(weights)) / total_support if total_support > 0 else 0.0

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
        system = platform.system()
        if custom_domain == 'ai.onnx.contrib':
            so.register_custom_ops_library(_get_library_path())  # Placeholder for contrib lib
        elif custom_domain == 'test.customop':
            if system == 'Windows':
                custom_op_library_path = "C:/Users/Engineer/Desktop/GitHUb/onnxruntime/build/Windows/RelWithDebInfo/Debug/custom_op_library.dll"
            elif system == 'Linux':
                custom_op_library_path = "/home/ali/Desktop/onnxruntime/cmake/cmake-build-release/libcustom_op_library.so"
            elif system == 'Darwin':
                custom_op_library_path = ""  # Add macOS path if needed
            else:
                raise RuntimeError(f"Unsupported OS: {system}")
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
        # FC MNIST: [N, 784]
        x_test = x_test.reshape(x_test.shape[0], -1)

    elif len(input_shape) == 4:
        # Handle Conv2D MNIST or similar
        N = x_test.shape[0]
        # Current x_test from tf.keras.datasets.mnist is [N, 28, 28]
        if x_test.ndim == 3:
            H, W = x_test.shape[1], x_test.shape[2]
            exp_N, exp_D1, exp_D2, exp_D3 = input_shape  # could be [None,28,28,1] or [None,1,28,28]

            # Case A: NHWC [N, 28, 28, 1]
            if (exp_D1 in (None, H) and exp_D2 in (None, W) and exp_D3 in (None, 1)):
                x_test = x_test.reshape(N, H, W, 1)

            # Case B: NCHW [N, 1, 28, 28]
            elif (exp_D1 in (None, 1) and exp_D2 in (None, H) and exp_D3 in (None, W)):
                x_test = x_test.reshape(N, 1, H, W)

            else:
                raise ValueError(
                    f"Model expects {input_shape} (4D) but x_test has shape {x_test.shape}. "
                    "Check whether model is NHWC or NCHW."
                )

        # If x_test already 4D but layout mismatches, try NHWC<->NCHW swap
        elif x_test.ndim == 4:
            # Model NHWC: [None, 28, 28, 1], data NCHW: [N, 1, 28, 28]
            if input_shape[1] == 28 and input_shape[2] == 28 and input_shape[3] in (None, 1) \
               and x_test.shape[1] == 1:
                x_test = np.transpose(x_test, (0, 2, 3, 1))
            # Model NCHW: [None, 1, 28, 28], data NHWC: [N, 28, 28, 1]
            elif input_shape[1] in (None, 1) and input_shape[2] == 28 and input_shape[3] == 28 \
                 and x_test.shape[-1] == 1:
                x_test = np.transpose(x_test, (0, 3, 1, 2))

    #print("Run Inference...")
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

    precision = calculate_precision(y_test, predicted_labels, average=average_method, zero_division=0)
    recall = calculate_recall(y_test, predicted_labels, average=average_method)
    f1 = calculate_f1_score(y_test, predicted_labels, average=average_method)

    #print(f"Accuracy: {accuracy*100:.2f}%")
    #print(f"Precision: {precision:.4f}")
    #print(f"Recall: {recall:.4f}")
    #print(f"F1 Score: {f1:.4f}")

    avg_inference_time = total_time / len(predicted_labels)
    print(f"Average inference time: {avg_inference_time*1000:.4f} ms")
    #print(f"Total execution time for inference: {total_time*1000:.4f} ms")
    #print(f"Accuracy: {accuracy*100:.2f}%")
    
    return accuracy, precision, recall, f1

