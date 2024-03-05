

file1 = open('/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/data/reason/output-2000/val_pretrain_analyse.txt', 'r')
count = 1
count_cons = 0
pgm_acc = 0
pc_aw = 0
pc_ac = 0
pw_ac = 0
pw_aw = 0 
all_correct = 0
ccw = 0
wcc = 0
wcw =0

wwc = 0
www = 0
cwc = 0
cww = 0

while True:
    
  
    # Get next line from file
    line1 = file1.readline()
    if not line1:
        break
    if (":" not in line1):
      print(line1)
    quest = line1.split(":")[1]
    gt = file1.readline().split(":")[1]
    gt = gt.replace('<START>','')
    
    pred = file1.readline().split(":")[1]
    
    gt = gt.replace('<NULL>','')
    pred = pred.replace('<NULL>','')
    gt = gt.strip()
    pred = pred.strip()
    ans = file1.readline().split(":")[1]
    p_ans = file1.readline().split(":")[1]
    cons_pred = file1.readline().split(":")[1]
    cons_gt = file1.readline().split(":")[1]
    if (cons_pred == cons_gt):
      count_cons = count_cons+1
    if (gt==pred):
      
      pgm_acc = pgm_acc+1
      if ans!=p_ans:
        
        pc_aw = pc_aw+1
        print("PCAW::Question number:", count)
        print("Question:", quest)
        print("GT:", gt)
        print("Pred:", pred)
        print("G_ans:", ans)
        print("P_ans:", p_ans)
      else:
        pc_ac = pc_ac+1
    else:
      if ans!=p_ans:
        pw_aw = pw_aw+1
      else:
        pw_ac = pw_ac+1
        print("PWAC::Question number:", count)
        print("Question:", quest)
        print("GT:", gt)
        print("Pred:", pred)
        print("G_ans:", ans)
        print("P_ans:", p_ans)
    if (cons_pred == cons_gt and gt == pred and ans == p_ans):
      all_correct = all_correct+1
    elif (cons_pred == cons_gt and gt == pred and ans != p_ans):
      ccw = ccw+1
    elif (cons_pred != cons_gt and gt == pred and ans == p_ans):
      wcc = wcc+1
      print("WCC::Question number:", count)
      print("Question:", quest)
      print("GT:", gt)
      print("Pred:", pred)
      print("G_ans:", ans)
      print("P_ans:", p_ans)
      print("Cons_pred:",cons_pred)
      print("Cons_gt:",cons_gt)
    elif (cons_pred != cons_gt and gt == pred and ans != p_ans):
      wcw = wcw+1
    elif (cons_pred != cons_gt and gt != pred and ans == p_ans):
      wwc = wwc+1
    elif (cons_pred != cons_gt and gt != pred and ans != p_ans):
      www = www+1
    elif (cons_pred == cons_gt and gt != pred and ans == p_ans):
      cwc = cwc+1
    elif (cons_pred == cons_gt and gt != pred and ans != p_ans):
      cww = cww+1
    

    count = count+1
    
print("Program accuracy:", pgm_acc)
print("pc_aw:", pc_aw)
print("pc_ac:", pc_ac)
print("pw_ac:", pw_ac)
print("pw_aw:", pw_aw)
print("All correct:", all_correct)
print("Constraint prediction acc:", count_cons)
print("WWC:", wwc)
print("WWW:", www)
print("CWC:", cwc)
print("CWW:", cww)








