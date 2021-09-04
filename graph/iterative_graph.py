import os
import os.path as osp
import numpy as np
import copy
from scipy.spatial import distance

import mmcv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from mmpose.models import build_posenet
from mmpose.models.detectors.associative_embedding import AssociativeEmbedding
# not occured error (plz ignore)
from brl_graph.graph.layers import Layers


# class IterativeGCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout=0.5):
#         super(IterativeGCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#         self.cls_layer = nn.Linear

class IterativeGraph(nn.Module):
    def __init__(
        self,
        backbone_dict,
        num_joints,
        hid_expansion=2,
        dropout=0.2,
        bias=False):
        super(IterativeGraph, self).__init__()
        # torch.set_default_dtype(torch.double)

        # self.backbone = AssociativeEmbedding(
        #     backbone=backbone_dict['model_cfg'],
        #     keypoint_head=backbone_dict['head_cfg'],
        #     train_cfg=backbone_dict['train_cfg'],
        #     pretrained=backbone_dict['pretrained']
        # ).double()

        # self.backbone.cuda()

        self.backbone = build_posenet(backbone_dict)

        self.num_joints = num_joints
        self.in_channels = num_joints + 2 # joints + coord((x,y))
        self.hid_channels = self.in_channels * hid_expansion
        self.cls_channels = num_joints
        self.reg_channels = 2
        self.bias = bias
        
        self.linear1 = nn.Linear(self.in_channels, self.hid_channels, bias=self.bias)
        self.ln1 = nn.LayerNorm(self.hid_channels)
        self.linear2 = nn.Linear(self.hid_channels, self.hid_channels, bias=self.bias)
        self.ln2 = nn.LayerNorm(self.hid_channels)
        # self.layers = self._make_layers(
        #     self.in_channels,
        #     self.hid_channels).double()
        self.cls = nn.Linear(self.hid_channels, self.cls_channels, bias=self.bias)
        self.reg = nn.Linear(self.hid_channels, self.reg_channels, bias=self.bias)
        
        # self.cls_loss = F.cross_entropy()
        # self.reg_loss = F.mse_loss()

    def _make_layers(self, in_c, hid_c):
        linear1 = nn.Linear(in_c, hid_c)
        bn1 = nn.BatchNorm1d(hid_c)
        linear2 = nn.Linear(hid_c, hid_c)
        bn2 = nn.BatchNorm1d(hid_c)
        
        layers = [linear1, bn1, linear2, bn2]

        return nn.Sequential(*layers)


    def forward(self, data_batch, train_mode=True):
        '''
        img(torch.Tensor[NxCximgHximgW]): Input image.
            targets(List(torch.Tensor[NxKxHxW])): Multi-scale target heatmaps.
            masks(List(torch.Tensor[NxHxW])): Masks of multi-scale target
                                              heatmaps
            joints(List(torch.Tensor[NxMxKx2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            img_metas(dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

        16(batch size) <class 'torch.Tensor'> torch.Size([16, 17, 192, 192])
        16(batch size) <class 'torch.Tensor'> torch.Size([16, 17, 192, 192])
        
        N: Batch Size
        K: The number of keypoint on specific datasets (e.g. the 17 on the COCO)
        (x, y): The coordinates of keypoint

        
        gloc, ploc: [N, K, 2]
        gdist, pdist, gt_adj: [N, K, K] (N, 17, 17)
        '''

        # gt_heatmap = data_batch['targets'].type(torch.DoubleTensor)
        
        gt_heatmap = data_batch['targets'][1]
        # batch_imgs = data_batch['img'].type(torch.DoubleTensor).cuda()

        if train_mode:
            low_res, high_res = self.backbone(**data_batch, only_res=True, use_double=False)

        elif not train_mode:
            low_res, high_res = self.backbone(
                **data_batch, return_loss=False, return_heatmap=True, only_res=True, use_double=False)


        gloc, ploc = self._get_gt_kpt_loc(gt_heatmap, self.num_joints), self._get_pred_kpt_loc(high_res)
        gdist, pdist = self._get_distance(gloc, ploc, self.num_joints)
        gt_adj = self._generate_gt_adj(gdist, self.num_joints)


        ploc_outputs = []
        # cls_results = []
        # cls_counter = np.zeros((17), dtype=np.int)
        
        for batch_idx in range(len(ploc)):
            # reg_tmp = torch.zeros((self.num_joints))
            # cls_tmp = []

            cp_pdist = torch.clone(pdist[batch_idx])
            cp_ploc = torch.clone(ploc[batch_idx])

            # cp_pdist = pdist[batch_idx].clone().detach().requires_grad_(True)
            # cp_ploc = ploc[batch_idx].clone().detach().requires_grad_(True)
            # cp_ploc = torch.from_numpy(np.array(ploc[batch_idx], dtype='double'))

            for kid, (loc, dist) in enumerate(zip(cp_ploc, cp_pdist)):
                # coord = torch.FloatTensor(loc)
                inputs = torch.cat((loc, dist), dim=0).cuda()
                # inputs = inputs.type(torch.FloatTensor).cuda()
                
                out = F.relu(self.ln1(self.linear1(inputs)))
                out = F.relu(self.ln2(self.linear2(out)))

                cls = self.cls(out)
                cls_out = F.log_softmax(cls, dim=-1)
                top_1_cls = torch.argmax(cls_out)
                reg = self.reg(out)
                # cls_counter[top_1_cls] += 1

                gt_adj[batch_idx][kid][top_1_cls], gt_adj[batch_idx][top_1_cls][kid] = 1, 1
                cp_ploc[top_1_cls] = reg
                re_calcul_dist = self._re_clacul_dist(cp_ploc, top_1_cls)
                
                cp_pdist[:, top_1_cls], cp_pdist[top_1_cls, :] = re_calcul_dist, re_calcul_dist


                assert torch.equal(cp_pdist[:, top_1_cls], cp_pdist[top_1_cls, :])
                assert torch.equal(cp_pdist[kid, top_1_cls], cp_pdist[top_1_cls, kid])
                
                cp_ploc = cp_ploc.type(torch.FloatTensor)
            ploc_outputs.append(cp_ploc)

        if train_mode:
            # gloc = nn.LayerNorm()
            
            # for i, j in zip(ploc_outputs, gloc):
            #     print(i.dtype, j.dtype)

            return ploc_outputs, gloc

        else:
            return ploc_outputs


    def _get_gt_kpt_loc(self, gt_heatmap, num_joints):
        gt_loc = []
        
        for batch_idx, img in enumerate(gt_heatmap): # per image
            tmp = []
            for kpt_idx, heat_kpt in enumerate(img): # heatmap
                # max_score = torch.max(heat_kpt).cpu().data.numpy()
                heat_cp = heat_kpt.cpu().data.numpy()
                kpt_loc = list(np.unravel_index(heat_cp.argmax(), heat_cp.shape))
                # kpt_loc.append(max_score)

                tmp.append(kpt_loc)

            gt_loc.append(tmp)

        assert len(gt_loc) == gt_heatmap.size(0)
        # gt_loc = torch.tensor(gt_loc, dtype=torch.double)
        gt_loc = torch.tensor(gt_loc, dtype=torch.float32)
        return gt_loc


    def _get_pred_kpt_loc(self, pred_heatmap):
        pred_loc = []

        for batch_idx, img in enumerate(pred_heatmap): # per image
            tmp = []
            for kpt_idx, heat_kpt in enumerate(img): # heatmap
                max_score = torch.max(heat_kpt).cpu().data.numpy()
                heat_cp = heat_kpt.clone().cpu().data.numpy()
                kpt_loc = list(np.unravel_index(heat_cp.argmax(), heat_cp.shape))
                # kpt_loc.append(max_score)
                # kpt_loc = torch.tensor(kpt_loc)
                tmp.append(kpt_loc)

            pred_loc.append(tmp)

        assert len(pred_loc) == pred_heatmap.size(0)
        # pred_loc = torch.tensor(pred_loc, dtype=torch.double)
        pred_loc = torch.tensor(pred_loc)

        return pred_loc


    def _get_distance(self, gt, pred, num_joints):
        # gt_dist = torch.empty((len(gt), num_joints, num_joints), dtype=torch.double)
        # pred_dist = torch.empty((len(pred), num_joints, num_joints), dtype=torch.double)
        gt_dist = torch.empty((len(gt), num_joints, num_joints))
        pred_dist = torch.empty((len(pred), num_joints, num_joints))

        for batch_idx, (gbatch, pbatch)  in enumerate(zip(gt, pred)):
            kpt_len = len(gbatch)    
            # tmp_gdist = torch.zeros((kpt_len, kpt_len), dtype=torch.double)
            # tmp_pdist = torch.zeros((kpt_len, kpt_len), dtype=torch.double)
            tmp_gdist = torch.zeros((kpt_len, kpt_len))
            tmp_pdist = torch.zeros((kpt_len, kpt_len))

            for kid in range(kpt_len):
                if kid == kpt_len-1:
                    break
                # print(f"kid {kid} --> gt: {gbatch[kid][:2]} / pred: {pbatch[kid][:2]}")
                '''
                numpy나 torch tensor의 경우 [:, :] 형태 가능
                내장 list에서 다차원 슬라이싱 할 경우, 무조건 차원별롣 대괄호 써야함
                a[0, :2] 불가 --> a[0][:2] 가능
                '''
                
                target_kid = kid+1 
                if np.array_equal([0, 0], gbatch[kid][:2]):
                    # print(f"gt kid {kid} --> [0, 0]")
                    tmp_gdist[kid, :], tmp_gdist[:, kid] = 0, 0

                if np.array_equal([0, 0], pbatch[kid][:2]):
                    # print(f"pred kid {kid} --> [0, 0]")
                    tmp_pdist[kid, :], tmp_pdist[:, kid] = 0, 0
                
                else:
                    while target_kid < kpt_len:
                        if not np.array_equal([0, 0], pbatch[target_kid][:2]):
                            # print(f'pred kid {kid} and pred target_kid {target_kid}')
                            pdist = distance.euclidean(pbatch[kid][:2], pbatch[target_kid][:2])
                            tmp_pdist[kid, target_kid], tmp_pdist[target_kid, kid] = pdist, pdist
                        
                        if not np.array_equal([0, 0], gbatch[kid][:2]):
                            if not np.array_equal([0, 0], gbatch[target_kid][:2]):
                                gdist = distance.euclidean(gbatch[kid][:2], gbatch[target_kid][:2])
                                tmp_gdist[kid, target_kid], tmp_gdist[target_kid, kid] = gdist, gdist
                        
                        target_kid += 1

                # if not np.array_equal([0, 0], gbatch[kid][:2]) and not np.array_equal([0, 0], pbatch[kid][:2]):
                #     # print(f"gt and pred kid {kid} --> not [0, 0]")
                #     while target_kid < kpt_len:
                #         if not np.array_equal([0, 0], gbatch[target_kid][:2]):
                #             gdist = distance.euclidean(gbatch[kid][:2], gbatch[target_kid][:2])
                #             tmp_gdist[kid, target_kid], tmp_gdist[target_kid, kid] = gdist, gdist

                #         if not np.array_equal([0, 0], pbatch[target_kid][:2]):
                #             pdist = distance.euclidean(pbatch[kid][:2], pbatch[target_kid][:2])
                #             tmp_pdist[kid, target_kid], tmp_pdist[target_kid, kid] = pdist, pdist

                #         target_kid += 1

                tmp_gdist[kid, kid], tmp_pdist[kid, kid] = 0, 0
                
            assert torch.all(tmp_gdist.transpose(0, 1).transpose(0, 1) == tmp_gdist)
            assert torch.all(tmp_pdist.transpose(0, 1).transpose(0, 1) == tmp_pdist)

            gt_dist[batch_idx] = tmp_gdist
            pred_dist[batch_idx] = tmp_pdist
        
        assert len(gt_dist) == len(gt)
        assert len(pred_dist) == len(pred)
        assert gt_dist[0].size(0) == np.shape(gt[0])[0]
        assert pred_dist[0].size(0) == np.shape(pred[0])[0]

        # gt_dist = torch.tensor(gt_dist, dtype=torch.double)
        # pred_dist = torch.tensor(pred_Dist, dtype=torch.double)
        return gt_dist, pred_dist


    def _generate_gt_adj(self, gdist, num_joints):
        gt_adj = []
        gt_dict = {
            '0': [1, 2],
            '1': [0, 2, 3],
            '2': [0, 1, 4],
            '3': [1, 5],
            '4': [2, 6],
            '5': [3, 6, 7, 11],
            '6': [4, 5, 8, 12],
            '7': [5, 9],
            '8': [6, 10],
            '9': [7],
            '10': [8],
            '11': [5, 12, 13],
            '12': [6, 11, 14],
            '13': [11, 15],
            '14': [12, 16],
            '15': [13],
            '16': [14]}

        for batch in gdist:
            tmp_adj = np.zeros((num_joints, num_joints))
            for idx, kpt in enumerate(batch):
                limb_list = gt_dict[str(idx)]
                for near_kpt in limb_list:
                    if kpt[near_kpt] != 0 and tmp_adj[idx, near_kpt] != 1: # if the keypoint label exist
                        tmp_adj[idx, near_kpt], tmp_adj[near_kpt, idx] = 1, 1
            
            assert np.allclose(tmp_adj, tmp_adj.T)
            gt_adj.append(tmp_adj)

        assert len(gt_adj) == len(gdist)
        assert gt_adj[0].shape == (num_joints, num_joints)

        gt_adj = torch.tensor(gt_adj, dtype=torch.int)
        return gt_adj

    
    def _re_clacul_dist(self, changed_ploc, pred_kpt):
        # out_dist = torch.zeros(len(changed_ploc), dtype=torch.double)
        out_dist = torch.zeros(len(changed_ploc))
        pred_kpt_data = changed_ploc[pred_kpt].data

        for idx, loc in enumerate(changed_ploc):
            if idx == pred_kpt:
                continue

            dist = distance.euclidean(pred_kpt_data, loc.data)
            
            # dist = torch.tensor(dist, dtype=torch.double)
            dist = torch.tensor(dist)

            out_dist[idx] = dist

        return out_dist



        




                    


