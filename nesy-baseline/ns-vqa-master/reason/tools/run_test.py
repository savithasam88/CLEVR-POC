import os, sys
import json
from typing import Counter
from pathlib import Path
import numpy

sys.path.append(os.path.join(Path(__file__).parents[1], 'options'))
sys.path.append(os.path.join(Path(__file__).parents[1]))

from executors.aspsolver import solve, getToken, getToken_program
from test_options import TestOptions
from datasets import get_dataloader
from models.parser import Seq2seqParser
import utils.utils as utils
import torch


import warnings


def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    return q_type

def computeReward(predicted, ans, function):
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
def check_program(pred, gt):
    """Check if the input programs matches"""
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
    gt = gt[1:len_gt]
    for i in range(len(pred)):
        if pred[i] == 2:
            break
        if pred[i] not in gt:
            return False
    for i in range(len(gt)):
        if gt[i] not in pred:
            return False
    return True

if torch.cuda.is_available():
  print("cuda available..")  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)
warnings.filterwarnings("ignore")

opt = TestOptions().parse()
loader = get_dataloader(opt, 'val')
model = Seq2seqParser(opt).to(device)

if (opt.load_checkpoint_path is not None):
  checkpoint = torch.load(opt.load_checkpoint_path)
  
  
print('| running test')
stats = {
    'exact_correct_ans': 0,
    'partial_correct_ans':0,
    'correct_prog': 0,
    'partial':0,
    'total': 0
}


vocab = utils.load_vocab(opt.clevr_vocab_path)
test_scene_path = opt.clevr_val_scene_path
env_folder =  opt.clevr_constraint_scene_path
#count=0
print_res = [] 
for x, y, answer, idx,constraint_type in loader:

    x = x.to(device = device)
    y = y.to(device = device)
    #count=count+1
    
    model.set_input(x, y)
    
    programs = model.parse()
    pg_np = programs.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    ans_np = answer.cpu().detach().numpy()
    
    ct_np = constraint_type.cpu().detach().numpy()
    
    for i in range(pg_np.shape[0]): 
        ans = ans_np[i]
        
        #print_res.append(idx)
        
        ans_tokens = [vocab["labels_idx_to_token"][j]  for j, ab in enumerate(list(ans)) if ans[j]==1]
        ans_tokens_str = ' '.join(ans_tokens)
        
        pred_pgm = getToken_program(pg_np[i], vocab['program_idx_to_token'])
        
        gt_pgm = getToken_program(y[i], vocab['program_idx_to_token'])
        
        
        pred= solve(pred_pgm, idx[i],  ct_np[i], 'val', test_scene_path, env_folder)
        
        ans_eq = False
        if pred != None:
            a = [vocab['labels'][d] for d in pred]
            b = [1 if c in a else 0 for c in range(len(vocab['labels']))]
            predicted = numpy.array(b)
            partial = computeReward(predicted, ans, 'partial')
            #stats['correct_ans']+=computeReward(predicted, ans, 'exact')
            if partial>0 and partial<1:
            	stats['partial_correct_ans']+=1 
            stats['partial'] = stats['partial']+partial 
            if numpy.array_equal(predicted, ans):
            	stats['exact_correct_ans']+=1
            	#print('Equal answers..', pred, predicted, ans)
            	ans_eq=True
    		
        if check_program(pg_np[i], y_np[i]):
            stats['correct_prog'] += 1
            #print('Equal programs..', pg_np[i], y_np[i])
            if(not (ans_eq)):
            	print('Testing question number:', idx[i])
            	print('Constraint type:', ct_np[i])
            	print('GT pgm:', gt_pgm, ans_tokens_str)
            	print('Predicted program:', pred_pgm, pred)
            	#print('Predicted vectors:', predicted, ans)
            	#print('Ground truth ans:', ans_tokens_str)
            	#print('Predicted ans:', pred)
        else:
        	if(ans_eq):
        		print('Testing question number-NEQ:', idx[i])
        		print('Constraint type:', ct_np[i])
        		print('GT pgm:', gt_pgm, ans_tokens_str)
        		print('Predicted program:', pred_pgm, pred)
    			
       
        stats['total'] += 1
        
        
    print('| %d/%d questions processed, accuracy %f' % (stats['total'], len(loader.dataset), stats['exact_correct_ans'] / stats['total']))

result = {
    'program_acc': stats['correct_prog'] / stats['total'],
    'exact_acc': stats['exact_correct_ans'] / stats['total'],
    'partial_acc':stats['partial'] / stats['total'],
    'exact_correct_ans':stats['exact_correct_ans'],
    'partial_correct_ans':stats['partial_correct_ans']
}
print(result)
utils.mkdirs(os.path.dirname(opt.save_result_path))
with open(opt.save_result_path, 'w') as fout:
    json.dump(result, fout)
print('| result saved to %s' % opt.save_result_path)
    
