from .clevr_executor import ClevrExecutor


def get_executor(opt):
    print('| creating %s executor' % opt.dataset)
    if opt.dataset == 'clevr':
        train_scene_json = opt.clevr_train_scene_path
        val_scene_json = opt.clevr_val_scene_path
        complete_val_scene_json = opt.clevr_complete_val_scene_path
        complete_train_scene_json = opt.clevr_complete_train_scene_path
        constraints_json = opt.clevr_constraint_scene_path
        
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    executor = ClevrExecutor(train_scene_json, val_scene_json, vocab_json, complete_train_scene_json, complete_val_scene_json, constraints_json)
    return executor