from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
import torch
from model.NLDF import NLDF

checkpoint_path = 'nldf/model.ckpt-20'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
total_parameters = 0
name_list = ['conv1_1/filter', 'conv1_1/biases', 'conv1_2/filter', 'conv1_2/biases',
             'conv2_1/filter', 'conv2_1/biases', 'conv2_2/filter', 'conv2_2/biases',
             'conv3_1/filter', 'conv3_1/biases', 'conv3_2/filter', 'conv3_2/biases',
             'conv3_3/filter', 'conv3_3/biases', 'conv4_1/filter', 'conv4_1/biases',
             'conv4_2/filter', 'conv4_2/biases', 'conv4_3/filter', 'conv4_3/biases',
             'conv5_1/filter', 'conv5_1/biases', 'conv5_2/filter', 'conv5_2/biases',
             'conv5_3/filter', 'conv5_3/biases', 'Fea_P1/W', 'Fea_P1/b',
             'Fea_P2/W', 'Fea_P2/b', 'Fea_P3/W', 'Fea_P3/b',
             'Fea_P4/W', 'Fea_P4/b', 'Fea_P5/W', 'Fea_P5/b',
             'Fea_P5_Deconv/W', 'Fea_P5_Deconv/b', 'Fea_P4_Deconv/W', 'Fea_P4_Deconv/b',
             'Fea_P3_Deconv/W', 'Fea_P3_Deconv/b', 'Fea_P2_Deconv/W', 'Fea_P2_Deconv/b',
             'Local_Fea/W', 'Local_Fea/b', 'Local_Score/W', 'Local_Score/b',
             'Fea_Global_1/W', 'Fea_Global_1/b', 'Fea_Global_2/W', 'Fea_Global_2/b',
             'Fea_Global/W', 'Fea_Global/b', 'Global_Score/W', 'Global_Score/b']

# name_list = ['conv1_1/filter/Adam_1', 'conv1_1/biases/Adam_1', 'conv1_2/filter/Adam_1', 'conv1_2/biases/Adam_1',
#              'conv2_1/filter/Adam_1', 'conv2_1/biases/Adam_1', 'conv2_2/filter/Adam_1', 'conv2_2/biases/Adam_1',
#              'conv3_1/filter/Adam_1', 'conv3_1/biases/Adam_1', 'conv3_2/filter/Adam_1', 'conv3_2/biases/Adam_1',
#              'conv3_3/filter/Adam_1', 'conv3_3/biases/Adam_1', 'conv4_1/filter/Adam_1', 'conv4_1/biases/Adam_1',
#              'conv4_2/filter/Adam_1', 'conv4_2/biases/Adam_1', 'conv4_3/filter/Adam_1', 'conv4_3/biases/Adam_1',
#              'conv5_1/filter/Adam_1', 'conv5_1/biases/Adam_1', 'conv5_2/filter/Adam_1', 'conv5_2/biases/Adam_1',
#              'conv5_3/filter/Adam_1', 'conv5_3/biases/Adam_1', 'Fea_P1/W/Adam_1', 'Fea_P1/b/Adam_1',
#              'Fea_P2/W/Adam_1', 'Fea_P2/b/Adam_1', 'Fea_P3/W/Adam_1', 'Fea_P3/b/Adam_1',
#              'Fea_P4/W/Adam_1', 'Fea_P4/b/Adam_1', 'Fea_P5/W/Adam_1', 'Fea_P5/b/Adam_1',
#              'Fea_P5_Deconv/W/Adam_1', 'Fea_P5_Deconv/b/Adam_1', 'Fea_P4_Deconv/W/Adam_1', 'Fea_P4_Deconv/b/Adam_1',
#              'Fea_P3_Deconv/W/Adam_1', 'Fea_P3_Deconv/b/Adam_1', 'Fea_P2_Deconv/W/Adam_1', 'Fea_P2_Deconv/b/Adam_1',
#              'Local_Fea/W/Adam_1', 'Local_Fea/b/Adam_1', 'Local_Score/W/Adam_1', 'Local_Score/b/Adam_1',
#              'Fea_Global_1/W/Adam_1', 'Fea_Global_1/b/Adam_1', 'Fea_Global_2/W/Adam_1', 'Fea_Global_2/b/Adam_1',
#              'Fea_Global/W/Adam_1', 'Fea_Global/b/Adam_1', 'Global_Score/W/Adam_1', 'Global_Score/b/Adam_1']


tf_list = []

for name in name_list:
    # print name
    temp = reader.get_tensor(name)
    # print temp.shape
    if len(temp.shape) > 1:
        temp = np.transpose(temp,(3,2,0,1))
    # print temp.shape
    tf_list.append(temp)

net = NLDF()
key_list = net.state_dict().keys()

# for key in key_list:
#     print key
#     print net.state_dict()[key].shape

for i in range(len(tf_list)):
    key = key_list[i]
    if net.state_dict()[key].shape == tf_list[i].shape:
        net.state_dict()[key].copy_(torch.Tensor(tf_list[i]))

torch.save(net.state_dict(), 'tf_param.pth')