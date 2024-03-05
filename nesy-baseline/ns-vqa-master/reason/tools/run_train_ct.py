import os, sys
import json
from typing import Counter
import numpy as np

sys.path.append("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason/options") # Adds higher directory to python modules path.
sys.path.append("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason")

import torch
import warnings
import utils.utils as utils
from train_options import TrainOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
from models.model_CC import ImageConstraintTypeClassification
#from trainer import Trainer

if torch.cuda.is_available():
  print("cuda available..")  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)
warnings.filterwarnings("ignore")


opt = TrainOptions().parse()
train_loader = get_dataloader(opt, 'train')
val_loader = get_dataloader(opt, 'val')
executor = get_executor(opt)
quest_parser = Seq2seqParser(opt).to(device)
num_constraint_types = 30
const_pred = ImageConstraintTypeClassification(device, num_constraint_types, opt)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, const_pred.parameters()),
                                          lr=opt.learning_rate)

if (opt.load_checkpoint_path_ct is not None):
  checkpoint = torch.load(opt.load_checkpoint_path_ct)
  const_pred.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


reward_decay = opt.reward_decay
entropy_factor = opt.entropy_factor
num_iters = opt.num_iters
run_dir = opt.run_dir
display_every = opt.display_every
checkpoint_every = opt.checkpoint_every
visualize_training = opt.visualize_training
        
vocab = utils.load_vocab(opt.clevr_vocab_path)

stats = {
            'train_losses': [],
            'train_batch_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0,
            'train_count_reason': [],
            'train_count_error': [],
            'train_count_re':[],
            'train_dir':[],
            'train_dirp':[],
            'train_rp':[],
            'train_cons':[],
            'val_cons':[],
            'val_count_reason': [],
            'val_count_error': [],
            'val_count_re':[],
            'val_dir' : [],
            'val_dirp':[],
            'val_rp':[]
        }
if opt.visualize_training:
    from reason.utils.logger import Logger
    logger = Logger('%s/logs' % opt.run_dir)


def log_stats( tag, value, t):
    if visualize_training and logger is not None:
        logger.scalar_summary(tag, value, t)

def log_params( t):
    if visualize_training and logger is not None:
        for tag, value in const_pred.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, _to_numpy(value), t)
            if value.grad is not None:
               logger.histo_summary('%s/grad' % tag, _to_numpy(value.grad), t)

def _to_numpy( x):
    return x.data.cpu().numpy()

def getToken(seq_ids, idx_to_token):
  tokens = ""
  for i in seq_ids:
    tokens= tokens+" "+idx_to_token[i.item()]
  return tokens

def check_val_accuracy( val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_idx, val_batches_list_ciidx, val_batches_list_ct, val_batches_list_images):
    reward = 0
    count_reason = 0
    count_error = 0
    count_re = 0
    count_dir = 0
    count_dirp = 0
    count_rp = 0
    count_cons_val = 0
    t = 0
    print_res = []
    #for x, y, ans, idx, complete_image_idx, constraint_type in self.val_loader:
    for x, y, ans, idx, complete_image_idx, constraint_type, image in zip(val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_idx, val_batches_list_ciidx, val_batches_list_ct, val_batches_list_images):
        """
        x = x.to(device = self.device)
        y.to(device = self.device)
        ans = ans.to(device = self.device)
        idx  = idx.to(device = self.device)
        complete_image_idx = complete_image_idx.to(device = self.device)
        constraint_type =  constraint_type.to(device = self.device)
        """
        ct_pred = const_pred.forward(image)
        quest_parser.set_input(x, y)
        pred = quest_parser.parse()
        #print("Pred_val:", pred)
        reward1, r, e, re, d, dp, rp, count_cons, print_res_batch = get_batch_reward(x,y, pred, ans, idx, complete_image_idx, ct_pred, constraint_type, 'val')
        reward += reward1
        count_cons_val += count_cons
        count_reason += r
        count_error += e
        count_re += re
        count_dir += d
        count_dirp += dp 
        count_rp += rp 
        print_res.extend(print_res_batch)
        ##reward += self.get_batch_reward(pred, ans, idx, 'val')
        t += 1
    reward = reward / t if t is not 0 else 0
    return reward, count_reason, count_error, count_re, count_dir, count_dirp, count_rp, count_cons_val, print_res 

def get_batch_reward(quests, gt, programs, answers, image_idxs, complete_image_idxs, constraint_type, constraint_g, split):
##def get_batch_reward(self, programs, answers, image_idxs, split):
    pg_np = programs.cpu().detach().numpy()
    ans_np = answers.cpu().detach().numpy()
    idx_np = image_idxs.cpu().detach().numpy()
    ct_np = constraint_type.cpu().detach().numpy()
    cs_np = complete_image_idxs.cpu().detach().numpy()
    ct_g = constraint_g.cpu().detach().numpy()
    """
    pg_np = programs._to_numpy()
    ans_np = answers._to_numpy()
    idx_np = image_idxs._to_numpy()
    ct_np = constraint_type._to_numpy()
    cs_np = complete_image_idxs._to_numpy()
    """
    reward = 0
    #print("idx_np:",idx_np)
    count_reason = 0
    count_error = 0
    count_re = 0
    count_dir = 0
    count_dirp = 0
    count_rp = 0
    print_res = []
    count_cons = 0
    #for i in range(pg_np.shape[0]):
    for i in range(len(pg_np)):
        
        ans = vocab['answer_idx_to_token'][ans_np[i]]
        pred, r, e, re, d, g = executor.run(pg_np[i], idx_np[i], cs_np[i], ct_np[i], split)
        
        count_reason += r
        count_error += e
        count_re += re 
        count_dir += d
        ##pred = self.executor.run(pg_np[i], idx_np[i], split)
        if ct_g[i] == ct_np[i]:
          count_cons = count_cons+1
        if pred == ans:
            reward += 1.0
            if d == 1:
              count_dirp += 1
            elif r == 1:
              count_rp += 1
        if split=='val':
          quest_token = getToken(quests[i], vocab['question_idx_to_token'])
          gt_token = getToken(gt[i], vocab['program_idx_to_token'])
          pred_pgm = getToken(pg_np[i], vocab['program_idx_to_token'])
          print_res.append("Question:"+quest_token+"\n GT:"+gt_token+"\n Pred pg:"+pred_pgm+"\n Ans:"+ans+"\n Pred_ans:"+pred+"\n Cons_pred:"+str(ct_np[i])+"\n Cons_gt:"+str(ct_g[i]))
    reward /= pg_np.shape[0]
    #reward /= len(pg_np)
    #print(split,'::Count-reason', count_reason)
    #print(split,'::Count-error', count_error)
    #print(split,'::Count-re', count_re)
    #print('---------------------------------------')
    return reward, count_reason, count_error, count_re, count_dir, count_dirp, count_rp, count_cons, print_res






#Train
t = 0
epoch = 0
baseline = 0
batches_list_x = []
batches_list_y = []
batches_list_ans = []
batches_list_idx = []
batches_list_ciidx = []
batches_list_ct = []
batches_list_images = []
for x, y, ans, idx, complete_image_idx, constraint_type, image in train_loader:
    image = image.to(device=device)
    x = x.to(device = device)
    y = y.to(device = device)
    ans = ans.to(device = device)
    idx = idx.to(device = device)
    complete_image_idx = complete_image_idx.to(device = device)
    constraint_type =  constraint_type.to(device = device)
    image = image.to(device = device)
    batches_list_x.append(x)
    batches_list_y.append(y)
    batches_list_ans.append(ans)
    batches_list_idx.append(idx)
    batches_list_ciidx.append(complete_image_idx)
    batches_list_ct.append(constraint_type)
    batches_list_images.append(image)
        
        
val_batches_list_x = []
val_batches_list_y = []
val_batches_list_ans = []
val_batches_list_idx = []
val_batches_list_ciidx = []
val_batches_list_ct = []
val_batches_list_images = []
for x, y, ans, idx, complete_image_idx, constraint_type, image in val_loader:
    image = image.to(device=device)
    x = x.to(device = device)
    y = y.to(device = device)
    ans = ans.to(device = device)
    idx = idx.to(device = device)
    complete_image_idx = complete_image_idx.to(device = device)
    constraint_type =  constraint_type.to(device = device)
    image = image.to(device = device)
    val_batches_list_x.append(x)
    val_batches_list_y.append(y)
    val_batches_list_ans.append(ans)
    val_batches_list_idx.append(idx)
    val_batches_list_ciidx.append(complete_image_idx)
    val_batches_list_ct.append(constraint_type)
    val_batches_list_images.append(image)

while t < num_iters:
  epoch += 1
  for x, y, ans, idx, complete_image_idx, constraint_type, image in zip(batches_list_x, batches_list_y, batches_list_ans, batches_list_idx, batches_list_ciidx, batches_list_ct, batches_list_images):
               
    t += 1
    loss, reward = None, None
    optimizer.zero_grad()
    ct_pred = const_pred.forward(image)
    quest_parser.set_input(x, y)
    pred = quest_parser.parse()
    reward, count_reason, count_error, count_re, count_dir, count_dirp, count_rp, count_cons_train, print_batch_res = get_batch_reward(x, y, pred, ans, idx, complete_image_idx, ct_pred, constraint_type, 'train')
    print("Reward ="+str(reward)+" at epoch:", epoch)
    baseline = reward * (1 - reward_decay) + baseline * reward_decay
    advantage = reward - baseline
    #print("Advantage:", advantage, type(advantage))
    loss = const_pred.backward(advantage, entropy_factor)
    optimizer.step()
    stats['train_batch_accs'].append(reward)
    stats['train_count_reason'].append(count_reason)
    stats['train_count_error'].append(count_error)
    stats['train_count_re'].append(count_re)
    stats['train_dir'].append(count_dir)
    stats['train_dirp'].append(count_dirp)
    stats['train_rp'].append(count_rp)
    stats['train_losses'].append(loss.item())
    stats['train_accs_ts'].append(t)
    stats['train_cons'].append(count_cons_train)
    if t % display_every == 0:
      
      log_stats('training batch reward', reward, t)
      print('| iteration %d / %d, epoch %d, reward %f' % (t, num_iters, epoch, reward))
      

    if t % checkpoint_every == 0 or t >= num_iters:
      print('| checking validation accuracy')
      val_acc, count_reason, count_error, count_re, count_dir, count_dirp, count_rp, count_cons_val, print_res = check_val_accuracy(val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_idx, val_batches_list_ciidx, val_batches_list_ct, val_batches_list_images)
      print('| validation accuracy %f' % val_acc)
      if val_acc >= stats['best_val_acc']:
          print('| best model')
          stats['best_val_acc'] = val_acc
          stats['model_t'] = t
          
          #Save model
          torch.save({
            'model_state_dict': const_pred.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, '%s/checkpoint_best.pt' % run_dir)
        
        
          if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            const_pred.cuda(opt.gpu_ids[0])

          with open('/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/data/reason/outputs_3000/val_rein.txt', 'w') as f1:
              for line in print_res:
                  f1.write(f"{line}\n")
          stats['val_cons'].append(count_cons_val)
          stats['val_accs'].append(val_acc)
          stats['val_count_reason'].append(count_reason)
          stats['val_count_error'].append(count_error)
          stats['val_count_re'].append(count_re)
          stats['val_dir'].append(count_dir)
          stats['val_dirp'].append(count_dirp)
          stats['val_rp'].append(count_rp)
          log_stats('val accuracy', val_acc, t)
          stats['val_accs_ts'].append(t)
          with open('%s/stats.json' % run_dir, 'w') as fout:
            json.dump(stats, fout)
            log_params(t)
                    
    if t >= num_iters:
      break
           
    

