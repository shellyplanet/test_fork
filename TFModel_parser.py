from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
    
def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        
FLAGS = None

def main(_):

  with tf.Graph().as_default():
    sess = tf.Session()
    print('load graph:', FLAGS.pb)
    #g = load_pb(FLAGS.pb)
    load_model(FLAGS.pb)
    g = sess.graph
    print('length', len(g.get_operations()))

    if FLAGS.tb_path:
        writer = tf.summary.FileWriter(FLAGS.tb_path, graph = g)

    from tensorflow.python.framework import graph_util
    import numpy as np
    operations = g.get_operations()
    strOpNames = ""
    i = 1
    for op in operations:
        strOpNames += "Operation:" + op.name + "\n"

    with open(FLAGS.dump_nodes_path, 'w') as file:
        file.write(strOpNames)
    
    lstNode = [n.name for n in g.as_graph_def().node]
    strNodeNames = ""
    for node in lstNode:
        strNodeNames += node + "\n"
    with open(FLAGS.dump_ops_path, 'w') as file:
        file.write(strNodeNames)

    strFlopsInfo = "Layer, Filter Num, Filter H, Filter W, Params\n"
    lstConv2D = [n for n in g.as_graph_def().node if n.op=='Conv2D']
    for node in lstConv2D:
        print('[_calc_conv_flops]node.name', node.name)
        strFlopsInfo += node.name + ","
        input_shape = graph_util.tensor_shape_from_node_def_name(g, node.input[0])
        print('[_calc_conv_flops]input_shape.as_list()', input_shape.as_list())
        print('[_calc_conv_flops]node.input[0]', input_shape)
        filter_shape = graph_util.tensor_shape_from_node_def_name(g, node.input[1])
        print('[_calc_conv_flops]node.input[1]', filter_shape)
        output_shape = graph_util.tensor_shape_from_node_def_name(g, node.name)
        print('[_calc_conv_flops]output_shape', output_shape)
        filter_height = int(filter_shape[0])
        filter_width = int(filter_shape[1])
        filter_in_depth = int(filter_shape[2])
        filter_num = int(filter_shape[3])
        params = filter_in_depth * filter_height * filter_width * filter_num
        print('[_calc_conv_flops]h:%d w:%d d:%d n:%d'% (filter_height, filter_width, filter_in_depth, filter_num))
        strFlopsInfo += "," + str(filter_height) + "," + str(filter_width) + "," + str(params) + "\n"
        print('[_calc_conv_flops]params:%d'% params)
        print('[_calc_conv_flops]output_shape.as_list()', output_shape.as_list())
        output_count = np.prod(output_shape.as_list()[1:], dtype=np.int64)
        output_dim = output_shape.as_list()[1:2]
        #output_count = np.prod(output_shape.as_list(), dtype=np.int64)
        print('[_calc_conv_flops]output_count', output_shape.as_list()[1:])    
    with open(FLAGS.dump_flops_path, 'w') as file:
        file.write(strFlopsInfo)
                
    # parse weights
    graph_nodes=[n for n in g.as_graph_def().node]
    #wts = [n for n in graph_nodes if n.op=='Const']
    #wts = [n for n in graph_nodes if n.name=='squeezenet/conv1/Conv2D_eightbit_min_squeezenet/conv1/weights/read']
    wts = [n for n in graph_nodes if n.name=='squeezenet/conv1/Conv2D_eightbit_reshape_squeezenet/conv1/weights/read']
    #wts = [n for n in graph_nodes if n.name=='squeezenet/conv1/weights']
#    t = g.get_tensor_by_name('squeezenet/conv1/Conv2D_eightbit_min_input:0')
#    print(t)

    strName = ""
    for n in wts:
        #p = tf.Print(n, [n], message="Test=========>")
        #print(p)
        print("node:", n.attr['T'])
        print("node type:", type(n.attr['T']))

    from tensorflow.python.framework import tensor_util
    strWts = ""
    for n in wts:
        strWts += "Name of the node - %s\n" % n.name
#        strWts += "Value - " + str(tensor_util.MakeNdarray(n.attr['value'].tensor)) + "\n"
#    with open(FLAGS.dump_weights_path, 'w') as file:
#        file.write(strWts)
'''
  from tensorflow.python.framework import ops as ops
  #g=tf.get_default_graph()
  i = 1
  for op in g.get_operations():
#    print('[%d]op.name:%s [%s]' % (i, op.name, op.type))
    if op.type in ['Conv2D']:
#      print('[%d]op.name:%s [%s]' % (i, op.name, op.type))
      flops = ops.get_stats_for_node_def(g, op.node_def, 'flops').value
#      print('flops:', flops)
      if flops is not None:
        print('[%d]op.name:%s [%s] \t %d' % (i, op.name, op.type, flops))
#        print('[%s] %d' % (op.type, flops))
#        print('op.vars:', op)
      i = i + 1

#  with g2.as_default():
#    flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
#    print('FLOP after freezing', flops.total_float_ops)
'''
'''
# ***** (1) Create Graph *****
g = tf.Graph()
sess = tf.Session(graph=g)
with g.as_default():
    A = tf.Variable(initial_value=tf.random_normal([25, 16]))
    B = tf.Variable(initial_value=tf.random_normal([16, 9]))
    C = tf.matmul(A, B, name='output')
    sess.run(tf.global_variables_initializer())
    flops = tf.profiler.profile(g, options = tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOP before freezing', flops.total_float_ops)
# *****************************        

# ***** (2) freeze graph *****
output_graph_def = graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ['output'])

with tf.gfile.GFile('graph.pb', "wb") as f:
    f.write(output_graph_def.SerializeToString())
# *****************************
# ***** (3) Load frozen graph *****
g2 = load_pb('./graph.pb')
with g2.as_default():
    flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOP after freezing', flops.total_float_ops)
'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--pb',
      type=str,
#      default='/home/shellyh/Data/Code/python/squeezenet/quantized_sqnet_0105.pb',
      default='/home/shellyh/Data/Code/python/squeezenet/saved_model.pb',
      help='path of .pb file')
  parser.add_argument(
      '--dump_weights_path',
      type=str,
      default='/tmp/graph/weights_all.txt',
      help='path to dump weights as .csv file')
  parser.add_argument(
      '--tb_path',
      type=str,
      default='/tmp/graph',
      help='path to dump logs for tensorboard')
  parser.add_argument(
      '--dump_nodes_path',
      type=str,
      default='/tmp/graph/nodes.txt',
      help='path to dump nodes')
  parser.add_argument(
      '--dump_ops_path',
      type=str,
      default='/tmp/graph/ops.txt',
      help='path to dump operation names')
  parser.add_argument(
      '--dump_flops_path',
      type=str,
      default='/tmp/graph/flops.csv',
      help='path to dump weights as .csv file')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
