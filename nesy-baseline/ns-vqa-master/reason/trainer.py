import json 
import torch
import time
import gc
import sys
import os
from pathlib import Path
import pickle

sys.path.append(os.path.join(Path(__file__).parents[0]))

from executors.aspsolver import solve, getToken, getToken_program 
import utils.utils as utils
import numpy
class Trainer():
    """Trainer"""

    def __init__(self, opt, train_loader, val_loader, model, device):
        self.opt = opt
        self.device = device
        self.data_folder_name = opt.data_folder_name
        self.ns_vqa_root = opt.ns_vqa_root
        self.reinforce = opt.reinforce
        self.reward_decay = opt.reward_decay
        self.entropy_factor = opt.entropy_factor
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.visualize_training = opt.visualize_training
        self.env_folder = opt.clevr_constraint_scene_path
        if opt.dataset == 'clevr':
            self.vocab = utils.load_vocab(opt.clevr_vocab_path)
        elif opt.dataset == 'clevr-humans':
            self.vocab = utils.load_vocab(opt.human_vocab_path)
        else:
            raise ValueError('Invalid dataset')

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.train_scene_path = opt.clevr_train_scene_path
        self.val_scene_path = opt.clevr_val_scene_path
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.seq2seq.parameters()),
                                          lr=opt.learning_rate)

        self.stats = {
            'train_losses': [],
            'train_batch_accs': [],
            'train_accs_ts': [],
            'val_losses': [],
            'val_accs': [],
            'val_accs_ts': [],
            'best_val_acc': -1,
            'model_t': 0,
                    }
        if opt.visualize_training:
            from reason.utils.logger import Logger
            self.logger = Logger('%s/logs' % opt.run_dir)

    def train(self):
        training_mode = 'reinforce' if self.reinforce else 'seq2seq'
        print('| start training %s, running in directory %s' % (training_mode, self.run_dir))
        t = 0
        epoch = 0
        baseline = 0
        batches_list_x = []
        batches_list_y = []
        batches_list_ans = []
        batches_list_scenes = []
        batches_list_ct = []
        for x, y, ans, idx, constraint_type in self.train_loader:
            x = x.to(device = self.device)
            y = y.to(device = self.device)
            ans = ans.to(device = self.device)
            constraint_type =  constraint_type.to(device = self.device)
            batches_list_x.append(x)
            batches_list_y.append(y)
            batches_list_ans.append(ans)
            
            batches_list_scenes.append(idx)
            batches_list_ct.append(constraint_type)
        val_batches_list_x = []
        val_batches_list_y = []
        val_batches_list_ans = []
        val_batches_list_scenes = []
        val_batches_list_ct = []
        for x, y, ans, idx,  constraint_type in self.val_loader:
            x = x.to(device = self.device)
            y = y.to(device = self.device)
            ans = ans.to(device = self.device)
            idx = idx.to(device = self.device)
            constraint_type =  constraint_type.to(device = self.device)
            val_batches_list_x.append(x)
            val_batches_list_y.append(y)
            val_batches_list_ans.append(ans)
            
            val_batches_list_scenes.append(idx)
            
            val_batches_list_ct.append(constraint_type)
         
        trainPlot_loss = []
        trainPlot_reward = []
        validationPlot_acc = []
        valPlot_loss = []
        while t < self.num_iters:
            ts = time.time()
            epoch += 1
            for x, y, ans, scene, constraint_type in zip(batches_list_x, batches_list_y, batches_list_ans, batches_list_scenes,  batches_list_ct):
                
                t += 1
                loss, reward = None, None
                self.model.set_input(x, y)
                self.optimizer.zero_grad()
                if self.reinforce:
                    
                    pred = self.model.reinforce_forward()
                    reward,val_pgm_acc, print_batch_res = self.get_batch_reward(x, y, pred, ans, scene, constraint_type, 'train')
                    #print("Reward ="+str(reward)+" at epoch:", epoch)
                    baseline = reward * (1 - self.reward_decay) + baseline * self.reward_decay
                    advantage = reward - baseline
                    self.model.set_reward(advantage)
                    loss = self.model.reinforce_backward(self.entropy_factor)
                                        
                else:
                    loss = self.model.supervised_forward()
                    print("Loss ="+str(loss)+" at iteration:", t)
                    self.model.supervised_backward()
                self.optimizer.step()

                if t >= self.num_iters:
                    break
            
            
            trainPlot_loss.append(loss)
            #input('VAL_ACC:::')
            val_acc,val_pgm_acc, print_res = self.check_val_accuracy(val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_scenes, val_batches_list_ct)
            print('|Epoch| ', epoch)
            print('|Loss %f|' % loss)
            print('| Validation Accuracy %f' % val_acc)
            print('| Validation Pgm Accuracy %f' % val_pgm_acc)
            #input('EPOCH over')
            validationPlot_acc.append(val_acc)
            if val_acc >= self.stats['best_val_acc']:
                print('| best model')
                self.stats['best_val_acc'] = val_acc
                self.stats['model_t'] = t
                self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir)
            if self.reinforce:
                trainPlot_reward.append(reward)
                print('|Reward %f|' % reward)
            else:
                val_loss= self.check_val_loss()
                valPlot_loss.append(val_loss)
                
                
        if self.reinforce:
            path_trainLoss = os.path.join(self.ns_vqa_root, 'data/reason', self.data_folder_name, 'train_loss_R.pickle')
            path_trainReward = os.path.join(self.ns_vqa_root, 'data/reason', self.data_folder_name, 'train_reward.pickle')
            path_valAcc = os.path.join(self.ns_vqa_root, 'data/reason', self.data_folder_name, 'val_acc_R.pickle')
            path_valRes = os.path.join(self.ns_vqa_root, 'data/reason', self.data_folder_name, 'val_res.txt')
            
            with open(path_trainLoss, 'wb') as f:
                pickle.dump(trainPlot_loss, f)
            with open(path_trainReward, 'wb') as f:
                pickle.dump(trainPlot_reward, f) 
            with open(path_valAcc, 'wb') as f:
                pickle.dump(validationPlot_acc, f) 
            with open(path_valRes, 'w') as f1:
                for line in print_res:
                    f1.write(f"{line}\n")
            f1.close()
            f.close()
        else:
            path_trainLoss = os.path.join(self.ns_vqa_root, 'data/reason', self.data_folder_name, 'train_loss_P.pickle')
            path_valLoss = os.path.join(self.ns_vqa_root, 'data/reason', self.data_folder_name, 'val_loss.pickle')
            path_valAcc = os.path.join(self.ns_vqa_root, 'data/reason', self.data_folder_name, 'val_acc_P.pickle')
            path_valRes = os.path.join(self.ns_vqa_root, 'data/reason', self.data_folder_name, 'val_res_P.txt')
            with open(path_trainLoss, 'wb') as f:
                pickle.dump(trainPlot_loss, f)
            with open(path_valLoss, 'wb') as f:
                pickle.dump(valPlot_loss, f)
            with open(path_valAcc, 'wb') as f:
                pickle.dump(validationPlot_acc, f) 
            f.close()   
            with open(path_valRes, 'w') as f1:
                for line in print_res:
                    f1.write(f"{line}\n")        
            f1.close()
    
    
    def check_val_loss(self):
        loss = 0
        t = 0
        for x, y, _, _ ,_ in self.val_loader:
            self.model.set_input(x, y)
            loss += self.model.supervised_forward()
            t += 1
        return loss / t if t is not 0 else 0

    def computeReward(self,predicted, ans, function):
        if function == 'partial':
        	r = numpy.where(ans == 1)
        	ans_ones = set(list(r[0]))
        	r = numpy.where(predicted == 1)
        	pred_ones = set(list(r[0]))
        	tp = len(list(ans_ones & pred_ones))
        	fp = len(list(pred_ones.difference(ans_ones)))
        	fn = len(list(ans_ones.difference(pred_ones)))
        	jaccard_index = tp/(tp + fp + fn)
        	return jaccard_index
		        
            #comm = numpy.sum(predicted == ans)
            #return comm/len(predicted)
        else:
            if numpy.array_equal(predicted, ans):
                return 1
            else:
            	return 0
    	
    	
    def check_val_accuracy(self, val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_scenes,  val_batches_list_ct):
        reward = 0
        val_pgm = 0
        t = 0
        print_res = []
        for x, y, ans, scene, constraint_type in zip(val_batches_list_x, val_batches_list_y, val_batches_list_ans, val_batches_list_scenes, val_batches_list_ct):
            self.model.set_input(x, y)
            pred = self.model.parse()
            reward1,val_pgm_acc, print_res_batch = self.get_batch_reward(x,y, pred, ans, scene, constraint_type, 'val')
            reward += reward1
            val_pgm += val_pgm_acc
            print_res.extend(print_res_batch)
            t += 1
        reward = reward / t if t is not 0 else 0
        val_pgm = val_pgm / t if t is not 0 else 0
        
        return reward, val_pgm, print_res 

    def check_program(self, pred, gt):
        len_pred = 0
        flag = False
        for i in range(len(pred)):
            if pred[i]==2:
                flag = True
                break
        if (flag):
            len_pred = i
        else:
            len_pred = i+1
        pred = pred[0:len_pred]
        for i in range(1, len(gt)):
            if gt[i]==2:
                break
        len_gt = i
        #print('gt old:', gt)
        gt = gt[1:len_gt]
        #input(gt)
        #if len_pred!= len_gt:
        #    return False
        for i in range(len(pred)):
            if pred[i] == 2: 
                break
            if pred[i] not in gt:
                return False
        for i in range(len(gt)):
            if gt[i] not in pred:
                return False
        return True

    
    def get_batch_reward(self, quests, gt, programs, answers, scene, constraint_type, split):
        pg_np = programs.cpu().detach().numpy()
        ans_np = answers.cpu().detach().numpy()
        ct_np = constraint_type.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        reward = 0
        val_pgm_accuracy = 0
        print_res = []
        for i in range(len(pg_np)):
            
            ans = ans_np[i]
            ans_tokens = [self.vocab["labels_idx_to_token"][j]  for j, x in enumerate(list(ans)) if ans[j]==1]
            ans_tokens_str = ' '.join(ans_tokens)
            pred_pgm = getToken_program(pg_np[i], self.vocab['program_idx_to_token'])
            
            if split == "train":
                pred= solve(pred_pgm, scene[i],  ct_np[i], split, self.train_scene_path, self.env_folder)
            else:
                pred = solve(pred_pgm, scene[i],  ct_np[i], split, self.val_scene_path, self.env_folder)
            
            
           
            if len(pred) != 0:
                a = [self.vocab['labels'][d] for d in pred]
                b = [1 if c in a else 0 for c in range(len(self.vocab['labels']))]
                predicted = numpy.array(b)
                reward = reward + self.computeReward(predicted, ans, 'partial')
                
                
                ##if numpy.array_equal(predicted, ans):
                    ##reward += 1.0
                    #gt_token = getToken_program(gt[i], self.vocab['program_idx_to_token'])
                    #print(pred_pgm, gt_token, predicted)
                    #input(ans)
                    
            if split=='val':
              
              quest_token = getToken(quests[i], self.vocab['question_idx_to_token'])
              gt_token = getToken_program(gt[i], self.vocab['program_idx_to_token'])
              if self.check_program(pg_np[i], gt[i]):
              	val_pgm_accuracy = val_pgm_accuracy+1
              	 
              print_res.append("Question:"+quest_token+"\n GT:"+gt_token+"\n Pred pg:"+pred_pgm+"\n Ans:"+ans_tokens_str+"\n Pred ans: "+str(pred))
        reward /= pg_np.shape[0]
        val_pgm_accuracy /= pg_np.shape[0]
        return reward,val_pgm_accuracy, print_res

    def log_stats(self, tag, value, t):
        if self.visualize_training and self.logger is not None:
            self.logger.scalar_summary(tag, value, t)

    def log_params(self, t):
        if self.visualize_training and self.logger is not None:
            for tag, value in self.model.seq2seq.named_parameters():
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, self._to_numpy(value), t)
                if value.grad is not None:
                    self.logger.histo_summary('%s/grad' % tag, self._to_numpy(value.grad), t)

    def _to_numpy(self, x):
        return x.data.cpu().numpy()
