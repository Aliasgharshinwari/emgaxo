import os
# print("📁 Current working directory:", os.getcwd())
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

def check_accuracy(model, use_custom_ops=False, custom_domain='test.customop',
                   results_file_name='model_results', x_test=[], y_test=[], batch_size=1):
    """
    Evaluate an ONNX model (MLP or CNN) on classification data.
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

    # ────────────────────────────────
    # Session setup
    # ────────────────────────────────
    so = ort.SessionOptions()
    if use_custom_ops:
        import importlib.util
        import emgaxo

        # Dynamically locate the installed emgaxo package path
        emgaxo_path = os.path.dirname(importlib.util.find_spec("emgaxo").origin)
        custom_op_library_path = os.path.join(emgaxo_path, "CustomOpLib", "libcustom_op_library.so")

        if not os.path.exists(custom_op_library_path):
            raise FileNotFoundError(f"❌ Custom op library not found: {custom_op_library_path}")
        else:
            print(f"✅ Loading custom op library from: {custom_op_library_path}")

        so.register_custom_ops_library(custom_op_library_path)

    ser = model.SerializeToString()

    available_providers = ort.get_available_providers()
    print("Available providers:", available_providers)
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
    session = ort.InferenceSession(ser, so, providers=providers)

    # ────────────────────────────────
    # Input info
    # ────────────────────────────────
    first_input = model.graph.input[0]
    input_name = first_input.name
    input_shape = [dim.dim_value if dim.dim_value != 0 else None
                   for dim in first_input.type.tensor_type.shape.dim]
    input_tensor_type = first_input.type.tensor_type.elem_type
    input_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[input_tensor_type]

    print(f"Model expects input type: {input_dtype}, shape: {input_shape}")

    # ────────────────────────────────
    # Normalize or cast
    # ────────────────────────────────
    if np.issubdtype(input_dtype, np.floating):
        x_test = x_test.astype(np.float32)
    elif np.issubdtype(input_dtype, np.int8):
        x_test = x_test.astype(np.int8)
    else:
        x_test = x_test.astype(np.uint8)

    # ────────────────────────────────
    # Auto shape correction
    # ────────────────────────────────
    expected_rank = len(input_shape)
    got_rank = x_test.ndim

    if got_rank < expected_rank:
        # If model expects (N, 784) and we have (N, 28, 28)
        if expected_rank == 2 and (x_test.ndim == 3 or x_test.shape[1:] == (28, 28)):
            x_test = x_test.reshape(x_test.shape[0], -1)

        # If model expects (N, 28, 28, 1) and we have (N, 28, 28)
        elif expected_rank == 4 and (x_test.ndim == 3 or x_test.shape[1:] == (28, 28)):
            x_test = np.expand_dims(x_test, axis=-1)

        else:
            # Generic fallback
            while x_test.ndim < expected_rank:
                x_test = np.expand_dims(x_test, axis=-1)

    # ────────────────────────────────
    # Inference loop
    # ────────────────────────────────
    predicted_labels = []
    total_time = 0.0
    os.makedirs("./Results", exist_ok=True)

    with open(f"./Results/{results_file_name}.txt", 'w') as file:
        for i in range(0, len(x_test), batch_size):
            batch = x_test[i:i + batch_size]

            start_time = time.time()
            output = session.run([session.get_outputs()[0].name], {input_name: batch})
            end_time = time.time()

            logits = output[0]
            num_classes = logits.shape[1] if logits.ndim > 1 else 1

            if num_classes == 1:
                threshold = 0.5 if np.issubdtype(logits.dtype, np.floating) else 127
                predicted_batch = (logits > threshold).astype(int).flatten()
            else:
                predicted_batch = np.argmax(logits, axis=1)

            predicted_labels.extend(predicted_batch)
            total_time += (end_time - start_time)

    # ────────────────────────────────
    # Metrics
    # ────────────────────────────────
    predicted_labels = np.array(predicted_labels)
    accuracy = np.mean(predicted_labels == y_test)

    is_binary = len(np.unique(y_test)) == 2
    average_method = 'binary' if is_binary else 'weighted'

    precision = precision_score(y_test, predicted_labels, average=average_method, zero_division=0)
    recall = recall_score(y_test, predicted_labels, average=average_method)
    f1 = f1_score(y_test, predicted_labels, average=average_method)

    avg_inference_time = total_time / len(predicted_labels)
    print(f"✅ Accuracy: {accuracy*100:.2f}% | F1: {f1:.4f} | Inference Time: {avg_inference_time*1000:.4f} ms")

    return accuracy, precision, recall, f1
