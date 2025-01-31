
import torch.nn as nn
import torch
import math

from timm.models.layers import ClassifierHead



class FeaturemapAnalyseHead(nn.Module):
    def __init__(self, model, criterion, logger, args, steps2log=None):
        super().__init__()


        self.num_featuremaps = (len(model.blocks) * 2)
        self.args = args

        self.logger = logger

        self.criterion = criterion
        self.steps2log = steps2log

        self.cls_heads = nn.ModuleList()
        self.patch_heads = nn.ModuleList()

        for i in range(self.num_featuremaps):
                #self.cls_heads.append(deepcopy(model.head))
                self.cls_heads.append(nn.Linear(model.head.in_features, model.head.out_features))
                #fm_head = ClassifierHead(model.head.in_features, model.head.out_features)
                #if args.test_complex_patch_head:

                fm_head = nn.Sequential()
                for i in range(2):
                    fm_head.append(
                        nn.Conv2d(model.head.in_features, model.head.in_features, 3, stride=1, padding=1,
                                  bias=False))
                    fm_head.append(nn.BatchNorm2d(model.head.in_features))
                    fm_head.append(nn.ReLU())
                fm_head.append(ClassifierHead(model.head.in_features, model.head.out_features))
                self.patch_heads.append(fm_head)


    def fm_emb_to_spatial(self, x):
        batch_size, num_token, embed_dim = x.shape
        patch_size = int(math.sqrt(num_token))
        return x[:, 1:].transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)

    def fm_spatial_to_emb(self, x, cls_token):
        x = x.flatten(2).transpose(1, 2)                                 # (B, 196, dim)
        x = torch.cat([cls_token, x], dim=1)
        return x

    def forward(self, x_inter):
        p_inter_cls_in = []
        p_inter_cls_out = []
        p_inter_patch_in = []
        p_inter_patch_out = []


        for i, softHeadCls, softHeadPatch in zip(range(len(self.cls_heads)), self.cls_heads, self.patch_heads):
            softInput_in = x_inter[i].detach()
            softInput_out = x_inter[i+1].detach()

            p_inter_cls_in.append(softHeadCls(softInput_in[:, 0]))
            p_inter_cls_out.append(softHeadCls(softInput_out[:, 0]))

            p_inter_patch_in.append(softHeadPatch(self.fm_emb_to_spatial(softInput_in)))
            p_inter_patch_out.append(softHeadPatch(self.fm_emb_to_spatial(softInput_out)))

        return {"p_cls_in": p_inter_cls_in, "p_cls_out": p_inter_cls_out,
                "p_patch_in": p_inter_patch_in, "p_patch_out": p_inter_patch_out}

    def log_metrics(self, model, s_x_list, targets, step):
        return 0




