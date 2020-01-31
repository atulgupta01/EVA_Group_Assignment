import numpy as np
import time, math
import tensorflow as tf

def byte_to_tf_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_to_tf_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_to_tf_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
  
def save_tf_records(imgList, labelList, out_path):
    writer = tf.python_io.TFRecordWriter(out_path)
 
    for i in range(labelList.shape[0]):
   
        example = tf.train.Example(features=tf.train.Features(
            feature={'image': byte_to_tf_feature(imgList[i].tostring()),
                     'labels': int64_to_tf_feature(
                         labelList[i])
                     }))
 
        writer.write(example.SerializeToString())
 
    writer.close()  
    
def load_tf_records_as_dataset(path,imgShape):
    dataset = tf.data.TFRecordDataset(path)
 
    def parser(record):
        featDefn = {
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'labels': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        }
 
        example = tf.parse_single_example(record, featDefn)
        img = tf.decode_raw(example['image'], tf.float32)
        img = tf.reshape(img, ( imgShape[0], imgShape[1], imgShape[2]))
        label = tf.cast(example['labels'], tf.int64)
        return img, label
 
    dataset = dataset.map(parser)
    return dataset
