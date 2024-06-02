from .build import build_model


import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial


#from .protonet import ProtoNet
#from .deploy import ProtoNet_Finetune, ProtoNet_Auto_Finetune, ProtoNet_AdaTok, ProtoNet_AdaTok_EntMin


def get_backbone(arch, pretrained, args):

    if arch == 'deit_base_patch16_224':
        
        # use timm
        '''model = create_model(
            arch,
            pretrained=pretrained,
            num_classes=args.nb_classes
        )'''
        
        '''from .mini_vision_transformer import VisionTransformer, _cfg
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()'''
        
        from . import vision_transformer as vit
        if args.expand_method == "auto-learngene" or args.expand_method == "heuristic-learngene":
            model = vit.__dict__['vit_base'](patch_size=16, num_classes=args.nb_classes)
        elif args.expand_method == "weight_assignment" or args.expand_method == "weight_clone":
            model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        else:
            model = vit.__dict__['vit_base'](patch_size=16, num_classes=1000)
        #state_dict = torch.load('./models/checkpoint/deit_base_patch16_224-b5f2ef4d.pth')['model']
        
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"  # DeiT-base
        #url = "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth"  # DeiT-base distilled 384 (1000 epochs)
        state_dict = torch.hub.load_state_dict_from_url(url=url)
        # state_dict = torch.load('./models/checkpoint/deit_base_patch16_224-b5f2ef4d.pth')

        if args.expand_method == "auto-learngene":
            keys_to_remove = ['blocks.6.', 'blocks.7.', 'blocks.8.', 'blocks.9.', 'blocks.10.', 'blocks.11.']
            state_dict['model'] = {k: v for k, v in state_dict['model'].items() if not any(key in k for key in keys_to_remove)}
            print(state_dict['model'].keys())
        elif args.expand_method == "heuristic-learngene":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.', 'blocks.6.', 'blocks.7.', 'blocks.8.']
            state_dict['model'] = {k: v for k, v in state_dict['model'].items() if not any(key in k for key in keys_to_remove)}
            print(state_dict['model'].keys())    
        
        #elif args.expand_method == "weight_assignment" or args.expand_method == "weight_clone":

        for k in ['head.weight', 'head.bias']:
            if k in state_dict['model']:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict['model'][k]

        model.load_state_dict(state_dict['model'], strict=False)
        print('Pretrained weights found at {}'.format(url))
        
        #model.load_state_dict(state_dict, strict=True)
        #print('Successfully load deit_base_patch16_224')

    elif arch == 'deit_base_w/o_ffn_patch16_224':

        from . import vision_transformer as vit
        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"  # DeiT-base
        state_dict = torch.hub.load_state_dict_from_url(url=url)

        for k in ['head.weight', 'head.bias']:
            if k in state_dict['model']:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict['model'][k]
        
        state_dict['model'] = {k: v for k, v in state_dict['model'].items() if 'mlp' not in k}

        print(state_dict['model'].keys())
        inherit_params = sum(p.numel() for p in state_dict['model'].values())
        print(f"Inheriting parameters in the model: {inherit_params}")
        inherit_params_in_M = inherit_params / 1e6
        print(f"Inheriting parameters in the model: {inherit_params_in_M:.2f}M")

        model.load_state_dict(state_dict['model'], strict=False)
        print('Pretrained weights found at {}'.format(url))

    elif arch == 'deit_base_w/o_pretrained_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_base'](patch_size=16, num_classes=args.nb_classes)


    elif arch == 'deit_small_w/o_ffn_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small'](patch_size=16, num_classes=args.nb_classes)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        
        state_dict = {k: v for k, v in state_dict.items() if 'mlp' not in k}

        print(state_dict.keys())
        inherit_params = sum(p.numel() for p in state_dict.values())
        print(f"Inheriting parameters in the model: {inherit_params}")
        inherit_params_in_M = inherit_params / 1e6
        print(f"Inheriting parameters in the model: {inherit_params_in_M:.2f}M")

        model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {}'.format(url))

    elif arch == 'deit_small_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small'](patch_size=16, num_classes=1000)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        if args.expand_method == "auto-learngene":
            keys_to_remove = ['blocks.6.', 'blocks.7.', 'blocks.8.', 'blocks.9.', 'blocks.10.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "heuristic-learngene":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.', 'blocks.6.', 'blocks.7.', 'blocks.8.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())  

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {}'.format(url))
        
        #model.load_state_dict(torch.load('./models/checkpoint/deit_small_patch16_224-cd65a155.pth'), strict=True)
    
    elif arch == 'deit_tiny_attn2_ffn6':
        from . import vision_transformer as vit
        #model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        model = vit.__dict__['deit_tiny_attn2_ffn6'](patch_size=16, num_classes=1000)
    
    elif arch == 'deit_tiny_attn2_ffn9':
        from . import vision_transformer as vit
        #model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        model = vit.__dict__['deit_tiny_attn2_ffn9'](patch_size=16, num_classes=1000)

    elif arch == 'deit_tiny_attn2_ffn12':
        from . import vision_transformer as vit
        #model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        model = vit.__dict__['deit_tiny_attn2_ffn12'](patch_size=16, num_classes=1000)

    elif arch == 'deit_small_attn4_ffn6':
        from . import vision_transformer as vit
        #model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        model = vit.__dict__['deit_small_attn4_ffn6'](patch_size=16, num_classes=1000)
    
    elif arch == 'deit_small_attn4_ffn9':
        from . import vision_transformer as vit
        #model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        model = vit.__dict__['deit_small_attn4_ffn9'](patch_size=16, num_classes=1000)

    elif arch == 'deit_small_attn4_ffn12':
        from . import vision_transformer as vit
        #model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        model = vit.__dict__['deit_small_attn4_ffn12'](patch_size=16, num_classes=1000)
    
    elif arch == 'deit_tiny_patch16_224':
        from . import vision_transformer as vit
        #model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=1000)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        if args.expand_method == "auto-learngene":
            keys_to_remove = ['blocks.6.', 'blocks.7.', 'blocks.8.', 'blocks.9.', 'blocks.10.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "heuristic-learngene":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.', 'blocks.6.', 'blocks.7.', 'blocks.8.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())  

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        
        model.load_state_dict(state_dict, strict=False)
        print('Successfully load deit_tiny_patch16_224')

        
        '''model = create_model(
            arch,
            pretrained=pretrained,
            num_classes=args.nb_classes
        )'''


    else:
        raise ValueError(f'{args.arch} is not conisdered in the current code.')

    return model


def get_model(arch, pretrained, args):    
    backbone = get_backbone(arch, pretrained, args)

    return backbone