import onnx
from onnx.onnx_ml_pb2 import OperatorSetIdProto
from onnx import helper

def modify_model(source_path, destination_path, ops_to_replace, nodes_to_replace=None, custom_domain='test.customop', use_approximate_ops = False):
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

                if not use_approximate_ops:
                    # Create the custom node
                    custom_node = helper.make_node(
                        op_type=f"Custom{node.op_type}",  # Custom operator type
                        inputs=input_names,
                        outputs=output_names,
                        name=node.name,
                        domain=custom_domain
                    )
                else:
                                  # Create the custom node
                    custom_node = helper.make_node(
                        op_type=f"Approximate{node.op_type}",  # Custom operator type
                        inputs=input_names,
                        outputs=output_names,
                        name=node.name,
                        domain=custom_domain
                    )

                # Remove the current node and insert the custom node
                graph.node.remove(node)
                graph.node.insert(i, custom_node)
                print(f"{i} Successfully replaced {node.name} ({node.op_type}) with {custom_node.name} ({custom_node.op_type})")

                # Update connections if necessary
                for output_name in output_names:
                    if output_name in input_to_output:
                        connected_node, output_index = input_to_output[output_name]
                        connected_node.input[output_index] = custom_node.output[0]

    # Save the modified model
    onnx.save_model(pre_trained_model, destination_path)
    print(f"Model successfully saved to {destination_path}")
