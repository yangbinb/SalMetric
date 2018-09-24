def load_pretrain_model(net, weights):
 net_keys = net.state_dict().keys()
 weights_keys = weights.keys()
 assert(len(net_keys) <= len(weights_keys))
 i = 0
 j = 0
 while i < len(net_keys):
  name_i = net_keys[i]
  name_j = weights_keys[j]
  if net.state_dict()[name_i].shape == weights[name_j].shape:
   net.state_dict()[name_i].copy_(weights[name_j].cpu())
   i += 1
   j += 1
  else:
   i += 1
 return net