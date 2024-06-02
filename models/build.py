from .vision_transformer import DeiT_mixture_learngene


def build_model(config, ancestry_model):
    model_type = config.MODEL.TYPE
    if  model_type == 'deit':
        hdp = config.MODEL.HDP.HDP
        # if hdp:
        #     assert isinstance(hdp, str)
        #     assert hdp in ['q', 'k', 'qk', 'qkv'], hdp
        model = DeiT_mixture_learngene(ancestry_model, 
                    config.MODEL.HDP.DISTRIBUTION, config.MODEL.HDP.FFN_RATIOS, config.MODEL.HDP.DESCENDANT_FFN_RATIOS, config.MODEL.HDP.FFN_INHERIT, config.MODEL.DEIT.NUM_HEADS, config.MODEL.DEIT.NUM_HEADS_LEARNGENE,
                    config.MODEL.DEIT.NUM_HEADS_DESCENDANT, embed_dim=config.MODEL.DEIT.EMBED_DIM, patch_size=16, num_classes=config.MODEL.NUM_CLASSES, depth=config.MODEL.DEIT.DEPTHS, 
                    hdp=config.MODEL.HDP.HDP,
                    hdp_ratios=config.MODEL.HDP.HDP_RATIOS,
                    hdp_non_linear=config.MODEL.HDP.NON_LINEAR,)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
