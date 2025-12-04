import onnx
from onnx.onnx_ml_pb2 import OperatorSetIdProto
from onnx import helper
from onnx.helper import make_opsetid
import onnx
from onnx import helper
from onnx import TensorProto
def get_init_tensor(graph, name):
    """Return the numpy value of an initializer tensor by name."""
    for init in graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    raise KeyError(f"initializer '{name}' not found")

def modify_model(source_path, destination_path, ops_to_replace, nodes_to_replace=None, custom_domain='test.customop', use_approximate_ops = False, INIT_Value=-1):
    """
    Generalized function to replace specific operator types in an ONNX model with custom nodes.

    Parameters:
        source_path (str): Path to the source ONNX model.
        destination_path (str): Path to save the modified ONNX model.
        ops_to_replace (list of str): List of operator types to replace with custom nodes.
        nodes_to_replace (list of str): List of node names to replace.
        custom_domain (str): Domain of the Custom Operators for inference

    Returns:
        None
    """
    # Load the pre-trained model
    pre_trained_model = onnx.load(source_path)
    graph = pre_trained_model.graph

    # Define the custom opset import
    custom_opset = helper.make_operatorsetid(domain=custom_domain, version=1)

    # Add the opset import to the model (if it's not already present)
    if not any(opset.domain == custom_domain for opset in pre_trained_model.opset_import):
        pre_trained_model.opset_import.append(custom_opset)

    # Ensure the default ONNX domain is present
    if not any(opset.domain == '' for opset in pre_trained_model.opset_import):
        pre_trained_model.opset_import.append(OperatorSetIdProto(domain='', version=15))

    # Map inputs to outputs
    input_to_output = {}
    for node in graph.node:
        for i, output in enumerate(node.output):
            input_to_output[output] = (node, i)

    # Replace specified nodes with custom nodes
    for i, node in enumerate(graph.node):
        if node.op_type in ops_to_replace:
            if node.name in nodes_to_replace or not nodes_to_replace:

                input_names = list(node.input)
                output_names = list(node.output)

                # Preserve existing attributes
                preserved_attrs = [attr for attr in node.attribute]

                # Add new custom attributes
                preserved_attrs.append(helper.make_attribute("custom_optimized", True))


                if not use_approximate_ops:
                    custom_node = helper.make_node(
                        op_type=f"Approx{node.op_type}",  # Custom operator type
                        inputs=input_names,
                        outputs=output_names,
                        name=node.name,
                        domain=custom_domain
                    )

                else:


                    custom_initializer = helper.make_tensor(
                        name=f"INIT_Value{i}",
                        data_type=onnx.TensorProto.INT64,  # Use the appropriate data type
                        dims=[],  # Scalar value
                        vals=[INIT_Value]  # Replace INIT_Value with actual value
                    )
                        # Add the new input to the list of inputs
                    input_names.append(f"INIT_Value{i}")

                    # Create the custom node
                    custom_node = helper.make_node(
                        op_type=f"Approx{node.op_type}",
                        inputs=input_names,
                        outputs=output_names,
                        name=node.name,
                        domain=custom_domain
                    )


                     # Add the initializer to the graph
                    graph.initializer.append(custom_initializer)


                # Remove the current node and insert the custom node
                graph.node.remove(node)
                graph.node.insert(i, custom_node)
                print(f"{i} Successfully replaced {node.name} ({node.op_type}) with {custom_node.name} ({custom_node.op_type})")

                # Update connections if necessary
                for output_name in output_names:
                    if output_name in input_to_output:
                        connected_node, output_index = input_to_output[output_name]
                        connected_node.input[output_index] = custom_node.output[0]

    #pre_trained_model = helper.make_model(graph,
    #                      opset_imports=[make_opsetid("", 19), make_opsetid(custom_domain, 1)])

    # Save the modified model
    onnx.save_model(pre_trained_model, destination_path)
    print(f"Model successfully saved to {destination_path}")


def modify_model(source_path, destination_path, ops_to_replace, nodes_to_replace=None, custom_domain='test.customop', use_approximate_ops = False, INIT_Value=-1, save_model=True, verbose=True):
    """
    Generalized function to replace specific operator types in an ONNX model with custom nodes.

    Parameters:
        source_path (str): Path to the source ONNX model.
        destination_path (str): Path to save the modified ONNX model.
        ops_to_replace (list of str): List of operator types to replace with custom nodes.
        nodes_to_replace (list of str): List of node names to replace.
        custom_domain (str): Domain of the Custom Operators for inference

    Returns:
        None
    """
    # Load the pre-trained model
    pre_trained_model = onnx.load(source_path)
    graph = pre_trained_model.graph
    if verbose:
        print(f"[DEBUG modify_model] INIT_Value = {INIT_Value}")
    # Define the custom opset import
    custom_opset = helper.make_operatorsetid(domain=custom_domain, version=1)

    # Add the opset import to the model (if it's not already present)
    if not any(opset.domain == custom_domain for opset in pre_trained_model.opset_import):
        pre_trained_model.opset_import.append(custom_opset)

    # Ensure the default ONNX domain is present
    if not any(opset.domain == '' for opset in pre_trained_model.opset_import):
        pre_trained_model.opset_import.append(OperatorSetIdProto(domain='', version=15))

    # Map inputs to outputs
    input_to_output = {}
    for node in graph.node:
        for i, output in enumerate(node.output):
            input_to_output[output] = (node, i)

    # Replace specified nodes with custom nodes
    for i, node in enumerate(graph.node):
        if node.op_type in ops_to_replace:
            if node.name in nodes_to_replace or not nodes_to_replace:

                input_names = list(node.input)
                output_names = list(node.output)

                # Preserve existing attributes
                preserved_attrs = [attr for attr in node.attribute]

                # Add new custom attributes
                preserved_attrs.append(helper.make_attribute("custom_optimized", True))


                if not use_approximate_ops:
                    custom_node = helper.make_node(
                        op_type=f"Custom{node.op_type}",  # Custom operator type
                        inputs=input_names,
                        outputs=output_names,
                        name=node.name,
                        domain=custom_domain
                    )

                else:

                    try:
                        A_scale = float(get_init_tensor(graph, input_names[1]))   # A_scale
                        B_scale = float(get_init_tensor(graph, input_names[4]))   # B_scale
                        C_scale = float(get_init_tensor(graph, input_names[7]))   # C_scale
                    except Exception as e:      # fall back gracefully
                        raise RuntimeError(f"could not fetch scales for {node.name}: {e}")

#                    output_scale_val = A_scale * B_scale / C_scale

                    custom_initializer1 = helper.make_tensor(
                        name=f"INIT_Value{i}",
                        data_type=onnx.TensorProto.INT64,  # Use the appropriate data type
                        dims=[],  # Scalar value
                        vals=[INIT_Value]  # Replace INIT_Value with actual value
                    )

                    # custom_initializer2 = helper.make_tensor(
                    #     name=f"output_scale{i}",
                    #     data_type=onnx.TensorProto.FLOAT,  # Use the appropriate data type
                    #     dims=[],  # Scalar value
                    #     vals=[output_scale_val]  # Replace INIT_Value with actual value
                    # )
                        # Add the new input to the list of inputs
                    input_names.append(f"INIT_Value{i}")
                    #input_names.append(f"output_scale{i}")

                    # Create the custom node
                    custom_node = helper.make_node(
                        op_type=f"Approx{node.op_type}",
                        inputs=input_names,
                        outputs=output_names,
                        name=node.name,
                        domain=custom_domain
                    )

                     # Add the initializer to the graph
                    graph.initializer.append(custom_initializer1)
#                    graph.initializer.append(custom_initializer2)


                # Remove the current node and insert the custom node
                graph.node.remove(node)
                graph.node.insert(i, custom_node)
                if verbose:
                    print(f"{i} Successfully replaced {node.name} ({node.op_type}) with {custom_node.name} ({custom_node.op_type})")
                    print(INIT_Value)
                # Update connections if necessary
                for output_name in output_names:
                    if output_name in input_to_output:
                        connected_node, output_index = input_to_output[output_name]
                        connected_node.input[output_index] = custom_node.output[0]

    #pre_trained_model = helper.make_model(graph,
    #                      opset_imports=[make_opsetid("", 19), make_opsetid(custom_domain, 1)])

    if save_model:
        # Save the modified model
        onnx.save_model(pre_trained_model, destination_path)
        print(f"Model successfully saved to {destination_path}")

    return pre_trained_model

import onnx
from onnx import helper

def OptimizeQGraph(model, io_datatype):
    modified_model = remove_node(model, "QuantizeLinear")
    modified_model = remove_node(modified_model, "DequantizeLinear")
    modified_model = remove_node(modified_model, "QLinearSoftmax")
    #modified_model = remove_node(modified_model, "Reshape")
    #modified_model = set_tensor_dtype(modified_model, "args_0", "uint8", True)
    #modified_model = set_tensor_dtype(modified_model, "sequential/dense_2/BiasAdd:0_quantized", "uint8", False)
     # Set dtype for all inputs
    for input_tensor in modified_model.graph.input:
        modified_model = set_tensor_dtype(modified_model, input_tensor.name, io_datatype, True)

    # Set dtype for all outputs
    for output_tensor in modified_model.graph.output:
        modified_model = set_tensor_dtype(modified_model, output_tensor.name, io_datatype, False)

    return modified_model

def remove_node(model, target_node_op_type):
    graph = model.graph
    target_node = None

    # Locate target node by op_type
    for node in graph.node:
        if node.op_type == target_node_op_type:
            target_node = node
            break

    if target_node is None:
        print(f"Node '{target_node_op_type}' not found.")
        return model  # Node not found, return original model

    print(f"Found node '{target_node_op_type}', removing...")

    # Find all successors (nodes that consume the output of the target node)
    successors = []
    for node in graph.node:
        for input_idx, input_name in enumerate(node.input):
            if input_name in target_node.output:
                successors.append((node, input_idx))

    # Check if any graph output directly references this node's output
    output_rewired = False
    for graph_output in graph.output:
        if graph_output.name in target_node.output:
            graph_output.name = target_node.input[0]  # Bypass to the input of removed node
            output_rewired = True

    if output_rewired:
        print(f"Rewired graph.output to bypass '{target_node_op_type}'")

    # Bypass the node itself (pass input directly to successors)
    input_to_pass = target_node.input[0]  # This works for QuantizeLinear/DequantizeLinear

    for successor, input_idx in successors:
        successor.input[input_idx] = input_to_pass  # Rewire to bypass target node

    # Rebuild graph without the target node
    new_nodes = [node for node in graph.node if node != target_node]

    # Create new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=graph.output,  # outputs were updated if needed
        initializer=graph.initializer,
        value_info=graph.value_info,
    )

    # Create new model
    new_model = helper.make_model(new_graph, producer_name=model.producer_name)
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)
    new_model.opset_import[0].version = 21  # Set opset to 21 before saving
    print(f"Node '{target_node_op_type}' removed and graph rewired.")
    return new_model


def set_tensor_dtype(model, tensor_name, new_dtype, is_input=True):
    """
    Set the data type of an input or output tensor in the ONNX model.

    Args:
        model (onnx.ModelProto): The ONNX model.
        tensor_name (str): The name of the input/output tensor.
        new_dtype (str): The new data type ("float32", "uint8", "int8", etc.).
        is_input (bool): Set True to modify an input, False for an output.

    Returns:
        onnx.ModelProto: Modified ONNX model.
    """
    # Map dtype string to ONNX TensorProto type
    dtype_map = {
        "float32": TensorProto.FLOAT,
        "float": TensorProto.FLOAT,
        "uint8": TensorProto.UINT8,
        "int8": TensorProto.INT8,
        "int16": TensorProto.INT16,
        "int32": TensorProto.INT32,
        "int64": TensorProto.INT64,
        "bool": TensorProto.BOOL
    }

    if new_dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {new_dtype}")

    new_dtype_enum = dtype_map[new_dtype]

    # Target either graph.input or graph.output
    target_list = model.graph.input if is_input else model.graph.output

    found = False
    for tensor in target_list:
        if tensor.name == tensor_name:
            tensor.type.tensor_type.elem_type = new_dtype_enum
            found = True
            print(f"{'Input' if is_input else 'Output'} '{tensor_name}' data type changed to {new_dtype}")
            break

    if not found:
        raise ValueError(f"{'Input' if is_input else 'Output'} '{tensor_name}' not found in the model")

    return model