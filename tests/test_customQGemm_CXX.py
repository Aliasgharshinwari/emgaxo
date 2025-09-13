import onnx
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as ort
import time

#[[105  70 125  82 100 137 201  51 110  88]]
def _create_test_model_test():
    nodes = [
        helper.make_node(
            'CustomQGemm',  # Use the correct operator name for your custom op
            ['A', 'A_scale', 'A_zeroPoint', 'B', 'B_scale', 'B_zeroPoint', 'C', 'C_scale', 'C_zeroPoint'],
            ['Y'],
            #domain='com.microsoft'
            domain='test.customop'
        )
    ]

    # Define input tensors as float32 and uint8
    A = helper.make_tensor_value_info('A', onnx_proto.TensorProto.UINT8, [1, 64])
    A_scale = helper.make_tensor_value_info('A_scale', onnx_proto.TensorProto.FLOAT, [])
    A_zeroPoint = helper.make_tensor_value_info('A_zeroPoint', onnx_proto.TensorProto.UINT8, [])
    B = helper.make_tensor_value_info('B', onnx_proto.TensorProto.UINT8, [64, 10])
    B_scale = helper.make_tensor_value_info('B_scale', onnx_proto.TensorProto.FLOAT, [])
    B_zeroPoint = helper.make_tensor_value_info('B_zeroPoint', onnx_proto.TensorProto.UINT8, [])
    C = helper.make_tensor_value_info('C', onnx_proto.TensorProto.INT32, [1, 10])
    C_scale = helper.make_tensor_value_info('C_scale', onnx_proto.TensorProto.FLOAT, [])
    C_zeroPoint = helper.make_tensor_value_info('C_zeroPoint', onnx_proto.TensorProto.UINT8, [])
    Y = helper.make_tensor_value_info('Y', onnx_proto.TensorProto.UINT8, [1, 10])

    # Define the graph
    graph = helper.make_graph(nodes, 'test0', [A, A_scale, A_zeroPoint, B, B_scale, B_zeroPoint, C, C_scale, C_zeroPoint], [Y])
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('ai.onnx.contrib', 1)], ir_version=7)
    return model

# Create and save the ONNX model
model = _create_test_model_test()
onnx.save(model, 'test_model.onnx')

# Define the matrices as Python lists
A = [[20, 0, 87, 27, 0, 27, 0, 57, 0, 0, 66, 0, 105, 33, 0, 0, 50, 0, 16, 20,
      41, 23, 0, 76, 6, 90, 0, 53, 13, 0, 127, 0, 11, 36, 25, 52, 0, 31, 0, 0, 
      0, 125, 0, 81, 43, 0, 0, 0, 90, 85, 0, 75, 0, 120, 98, 0, 98, 54, 0, 0, 0, 0, 0, 8]]
A_scale = np.array([0.07515275478363037], dtype=np.float32)
A_zeroPoint = np.array([0], dtype=np.uint8)

B = [[14, 225, 168, 77, 163, 192, 171, 127, 151, 104],
     [61, 169, 59, 77, 211, 70, 108, 150, 101, 136],
     [161, 129, 120, 133, 236, 69, 199, 81, 89, 123],
     [180, 252, 77, 159, 142, 217, 184, 61, 78, 149],
     [161, 55, 129, 125, 169, 136, 75, 141, 185, 102],
     [210, 101, 255, 88, 195, 145, 117, 131, 40, 82],
     [172, 161, 112, 189, 66, 119, 130, 203, 71, 100],
     [63, 48, 208, 206, 54, 85, 135, 53, 188, 221],
     [103, 94, 113, 135, 149, 140, 160, 103, 99, 183],
     [195, 97, 108, 144, 115, 127, 68, 152, 148, 185],
     [133, 41, 54, 84, 125, 195, 204, 81, 167, 162],
     [152, 182, 194, 48, 134, 149, 133, 187, 167, 162],
     [103, 122, 65, 62, 170, 192, 207, 114, 202, 184],
     [204, 69, 205, 201, 211, 99, 53, 102, 99, 214],
     [166, 149, 218, 175, 67, 68, 70, 105, 146, 155],
     [170, 210, 245, 104, 144, 186, 190, 219, 61, 51],
     [187, 221, 214, 69, 195, 113, 168, 143, 206, 96],
     [158, 79, 91, 97, 51, 231, 76, 172, 170, 110],
     [178, 157, 62, 120, 199, 59, 175, 110, 161, 196],
     [160, 59, 167, 96, 149, 210, 101, 155, 104, 176],
     [135, 150, 69, 79, 176, 179, 163, 154, 131, 171],
     [224, 110, 93, 146, 135, 115, 155, 111, 203, 150],
     [113, 162, 111, 218, 180, 180, 60, 161, 206, 212],
     [234, 133, 129, 58, 96, 94, 213, 188, 112, 178],
     [83, 119, 115, 171, 112, 147, 0, 190, 136, 183],
     [183, 148, 174, 145, 109, 197, 193, 114, 182, 73],
     [55, 92, 170, 217, 203, 161, 94, 165, 216, 210],
     [173, 175, 220, 74, 203, 98, 161, 197, 204, 192],
     [217, 163, 144, 171, 156, 131, 213, 218, 118, 104],
     [118, 111, 101, 150, 165, 87, 102, 132, 160, 114],
     [145, 79, 206, 206, 59, 200, 188, 58, 184, 75],
     [72, 148, 203, 197, 78, 49, 52, 200, 180, 175],
     [20, 219, 163, 175, 87, 203, 37, 107, 119, 123],
     [92, 104, 201, 168, 130, 189, 104, 95, 79, 184],
     [133, 64, 84, 190, 141, 208, 114, 194, 180, 150],
     [200, 169, 117, 143, 200, 133, 187, 92, 182, 211],
     [167, 117, 189, 98, 114, 179, 80, 65, 103, 224],
     [56, 202, 209, 190, 228, 150, 90, 77, 96, 130],
     [223, 59, 109, 38, 205, 135, 116, 161, 103, 161],
     [64, 199, 60, 191, 188, 138, 116, 194, 198, 102],
     [159, 218, 31, 155, 143, 190, 165, 145, 102, 74],
     [88, 162, 182, 104, 123, 130, 215, 174, 150, 34],
     [149, 202, 151, 180, 90, 48, 186, 121, 142, 178],
     [68, 180, 101, 172, 125, 194, 205, 111, 145, 221],
     [218, 50, 147, 181, 131, 71, 197, 165, 131, 108],
     [176, 124, 169, 110, 104, 145, 105, 101, 149, 148],
     [102, 192, 115, 120, 107, 195, 53, 186, 83, 225],
     [237, 156, 205, 124, 60, 116, 98, 150, 49, 83],
     [60, 94, 192, 40, 107, 215, 218, 160, 165, 61],
     [148, 107, 82, 94, 211, 154, 198, 213, 79, 190],
     [111, 143, 140, 143, 189, 171, 169, 206, 215, 132],
     [180, 221, 129, 87, 203, 92, 179, 93, 40, 216],
     [189, 230, 118, 78, 178, 72, 45, 190, 138, 179],
     [140, 84, 203, 181, 89, 217, 203, 30, 199, 164],
     [192, 205, 125, 171, 97, 185, 208, 53, 176, 93],
     [120, 172, 147, 119, 97, 96, 189, 103, 225, 41],
     [103, 75, 135, 214, 175, 109, 146, 191, 61, 53],
     [194, 91, 215, 121, 77, 152, 154, 160, 116, 117],
     [111, 109, 197, 125, 82, 98, 135, 178, 90, 194],
     [81, 73, 121, 220, 24, 119, 101, 198, 105, 65],
     [180, 137, 164, 160, 111, 196, 192, 234, 77, 83],
     [45, 121, 32, 140, 194, 118, 132, 151, 115, 130],
     [91, 114, 153, 169, 223, 135, 90, 115, 197, 105],
     [207, 153, 139, 208, 65, 216, 170, 128, 161, 208]]

B_scale = np.array([0.004783816169947386], dtype=np.float32)
B_zeroPoint = np.array([146], dtype=np.uint8)

C = [[-263, -169, 22, -164, 30, 67, -186, -97, 413, 97]]
C_scale = np.array([0.2838478982448578], dtype=np.float32)
C_zeroPoint = np.array([115], dtype=np.uint8)

# Create ONNX Runtime session with custom ops
so = ort.SessionOptions()
so.register_custom_ops_library("D:/GitHub/onnxruntime/build/Windows/Release/Release/custom_op_library.dll")  # Path to your shared library
so.intra_op_num_threads = 3
start_time = time.time()

# Create inference session
sess = ort.InferenceSession("test_model.onnx", so)

# Run inference
outputs = sess.run(None, {
    'A': A,
    'A_scale': A_scale,
    'A_zeroPoint': A_zeroPoint,
    'B': B,
    'B_scale': B_scale,
    'B_zeroPoint': B_zeroPoint,
    'C': C,
    'C_scale': C_scale,
    'C_zeroPoint': C_zeroPoint
})
# End the timer
total_time = time.time() - start_time

# Print output
print(f"Model Output: {outputs[0]}")
print(f"Total Time Taken {total_time}s")
#[[ -7302 -35129   7579 -26060 -11582  17064  68302 -50684  -4448 -21285]]