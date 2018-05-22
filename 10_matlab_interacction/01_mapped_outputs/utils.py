# --------------------------------------------------- #
# LOG HELPER FUNCTION
# --------------------------------------------------- #
def shape_log(tensor):
  print("{} has shape: {}".format(
    tensor.op.name,
    tensor.get_shape().as_list()))


# --------------------------------------------------- #
# WEIGHT BIAS EXTRACTION
# --------------------------------------------------- #
def get_layer_vars(op_name) :
  w = tf.get_default_graph().get_tensor_by_name(
      "{}/kernel:0".format(op_name))
  b = tf.get_default_graph().get_tensor_by_name(
      "{}/bias:0".format(op_name))
  return w, b