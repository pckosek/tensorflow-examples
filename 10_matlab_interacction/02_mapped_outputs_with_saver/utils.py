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


# --------------------------------------------------- #
# TRAINING STEP LOGGER
# --------------------------------------------------- #
def display_update(indx, interval):
    # post an update
    if ((indx+1)%interval == 0):
        print("[step: {}]".format(indx))

# --------------------------------------------------- #
# SAVER CLASS
# --------------------------------------------------- #
class saver_ops():

  def __init__(self, max_to_keep, global_step, sess):
    self.saver_dir   = os.getcwd()
    self.ckpt_path   = os.path.join(self.saver_dir, 'model.ckpt')
    self.saver_path  = os.path.join(self.ckpt_path, 'saver')
    self.global_step = global_step
    self.sess        = sess

    self.Saver = tf.train.Saver(max_to_keep=max_to_keep)

    self.try_to_restore()

  def try_to_restore(self) :
    # possibly restore model
    ckpt = tf.train.get_checkpoint_state(self.saver_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.Saver.restore(self.sess, ckpt.model_checkpoint_path)
      print('saver restored')
    else:
      print('saver NOT restored')


  def save(self) :
    # save the saver
    self.Saver.save(self.sess, self.ckpt_path, global_step=self.global_step)
