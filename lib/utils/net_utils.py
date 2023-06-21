import torch
import os
import torch.nn.functional
from collections import OrderedDict

def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1, suffix = '', reset_scheduler=False):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0
    if suffix == '':
        split_char = '.'
    else:
        split_char = '_' 
    pths = [int(pth.split(split_char)[0]) for pth in os.listdir(model_dir) if pth.endswith(suffix+'.pth')]
    
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch

    filetype = suffix+'.pth'
    filename = ("{}"+filetype).format(pth)
    
    print('Load model: {}'.format(os.path.join(model_dir, filename)))
    pretrained_model = torch.load(os.path.join(model_dir, filename))
    try:
        net.load_state_dict(pretrained_model['net'], strict=True)
    except Exception as e:
        print(">>>>> Error loading strict network:")
        print(e)
        print("<<<<< Loading network with strict = False")
        net.load_state_dict(pretrained_model['net'], strict=False)
    if scheduler is not None:
        if not reset_scheduler:
            optim.load_state_dict(pretrained_model['optim'])
            scheduler.load_state_dict(pretrained_model['scheduler'])
    if recorder is not None:
        recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1

def load_model_arcface(net, centroids, optim, scheduler, recorder, model_dir, resume=True, epoch=-1, suffix = '', reset_scheduler=False):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0
    if suffix == '':
        split_char = '.'
    else:
        split_char = '_' 
    pths = [int(pth.split(split_char)[0]) for pth in os.listdir(model_dir) if pth.endswith(suffix+'.pth')]
    
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch

    filetype = suffix+'.pth'
    filename = ("{}"+filetype).format(pth)
    
    print('Load model: {}'.format(os.path.join(model_dir, filename)))
    pretrained_model = torch.load(os.path.join(model_dir, filename))
    try:
        net.load_state_dict(pretrained_model['net'], strict=True)
    except Exception as e:
        print(">>>>> Error loading strict network:")
        print(e)
        print("<<<<< Loading network with strict = False")
        net.load_state_dict(pretrained_model['net'], strict=False)
    centroids.load_state_dict(pretrained_model['centroids'])
    if scheduler is not None:
        if not reset_scheduler:
            optim.load_state_dict(pretrained_model['optim'])
            scheduler.load_state_dict(pretrained_model['scheduler'])
    if recorder is not None:
        recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1

def load_model_emilio(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1, suffix = '', reset_scheduler=False):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0
    if suffix == '':
        split_char = '.'
    else:
        split_char = '_' 
    pths = [int(pth.split(split_char)[0]) for pth in os.listdir(model_dir) if pth.endswith(suffix+'.pth')]
    
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch

    filetype = suffix+'.pth'
    filename = ("{}"+filetype).format(pth)
    
    print('Load model: {}'.format(os.path.join(model_dir, filename)))
    pretrained_model = torch.load(os.path.join(model_dir, filename))
    mask_values = pretrained_model.pop('mask_values', [0, 1])
    fixed_pretrained_model = {}
    for key in pretrained_model:
        newkey = "backbone." + key
        fixed_pretrained_model[newkey] = pretrained_model[key]

    net.load_state_dict(fixed_pretrained_model, strict=True)
    return pth + 1


def save_model(net, optim, scheduler, recorder, epoch, model_dir,suffix=''):
    os.system('mkdir -p {}'.format(model_dir))
    filetype = suffix+'.pth'
    filename = ("{}"+filetype).format(epoch)
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, ('{}'+filetype).format(epoch)))

    if suffix == '':
        split_char = '.'
    else:
        split_char = '_' 

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split(split_char)[0]) for pth in os.listdir(model_dir)  if filetype in pth]
    if len(pths) <= 200:
        return
    os.system('rm {}'.format(os.path.join(model_dir, ('{}'+filetype).format(min(pths)))))

def save_model_arcface(net, centroids, optim, scheduler, recorder, epoch, model_dir,suffix=''):
    os.system('mkdir -p {}'.format(model_dir))
    filetype = suffix+'.pth'
    filename = ("{}"+filetype).format(epoch)
    torch.save({
        'net': net.state_dict(),
        'centroids': centroids.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, ('{}'+filetype).format(epoch)))

    if suffix == '':
        split_char = '.'
    else:
        split_char = '_' 

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split(split_char)[0]) for pth in os.listdir(model_dir)  if filetype in pth]
    if len(pths) <= 200:
        return
    os.system('rm {}'.format(os.path.join(model_dir, ('{}'+filetype).format(min(pths)))))



def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('Load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'], strict=strict)

    return pretrained_model['epoch'] + 1
