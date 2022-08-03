import onnx_graphsurgeon as gs
import numpy as np
import onnx


# gs load a graph
graph = gs.import_onnx(onnx.load("yolov7-w6-pose-sim.onnx"))

# Since we already know the names of the tensors we're interested in, we can
# grab them directly from the tensor map.
#
# NOTE: If you do not know the tensor names you want, you can view the graph in
# Netron to determine them, or use ONNX GraphSurgeon in an interactive shell
# to print the graph.
tensors = graph.tensors()

# If you want to embed shape information, but cannot use ONNX shape inference,
# you can manually modify the tensors at this point:
#
# IMPORTANT: You must include type information for input and output tensors if it is not already
# present in the graph.
#
# NOTE: ONNX GraphSurgeon will also accept dynamic shapes - simply set the corresponding
# dimension(s) to `gs.Tensor.DYNAMIC`, e.g. `shape=(gs.Tensor.DYNAMIC, 3, 224, 224)`
inputs = [tensors["745"].to_variable(dtype=np.float32), 
    tensors["783"].to_variable(dtype=np.float32), 
    tensors["821"].to_variable(dtype=np.float32),
    tensors["859"].to_variable(dtype=np.float32)]

# Add a output tensor of new graph
modified_output = gs.Variable(name="output0", dtype=np.float32, shape=(57001, 1, 1))

# Add a new node that you want
new_node = gs.Node(op="YoloLayer_TRT", name="YoloLayer_TRT_0", inputs=inputs, outputs=[modified_output])

# append into graph
graph.nodes.append(new_node)
graph.outputs = [modified_output]

graph.cleanup().toposort()

# gs save a graph
onnx.save(gs.export_onnx(graph), "yolov7-w6-pose-sim-yolo.onnx")
