import numpy as np
import copy, math
import torch
import mmcv


# class GraphMaker(nn.Module):
#     def __init__():


def get_distance(pose_results):


    kpt_results = np.random.rand(1, 17, 2)

    # kpt 하나 추출
    # 추출된 keypoint와 나머지 keypoint 간의 단순 거리 계산
    
    '''
    1: 2 - [1, 2], [1, 3]
    2: 3 - (2, 1), [2, 3], [2, 4]
    3: 3 - (3, 1), (3, 2), [3, 5]
    4: 2 - (4, 2), [4, 6]
    5: 2 - (5, 3), [5, 7]
    6: 4 - (6, 4), [6, 7], [6, 8], [6, 12]
    7: 4 - (7, 5), (7, 6). [7, 9], [7, 13]
    8: 2 - (8, 6), [8, 10]
    9: 2 - (9, 7), [9, 11]
    10: 1 - (10, 8)
    11: 1 - (11, 9)
    12: 3 - (12 ,6), [12, 13], [12, 14]
    13: 3 - (13, 7), (13, 12), [13 ,15]
    14: 2 - (14, 12), [14, 16]
    15: 2 - (15, 13), [15, 17]
    16: 1 - (15, 13)
    17: 1 - (16, 14)
    '''
    gt_skeleton = [
        [1,2], [1,3],[2,3],[2,4],
        [3,5], [4,6], [5,7],
        [6,7], [6,8], [6, 12],
        [7,9], [7, 13],
        [8, 10], [9, 11],
        [12, 13], [12, 14], [14, 16],
        [13, 15], [15, 17]
    ]

    one_link_kpt = [9, 10, 15, 16]
    two_link_kpt = [0, 3, 4, 7, 8, 13, 14]
    three_link_kpt = [1, 2, 11, 12]
    four_link_kpt = [5, 6]

    kpt_len = len(pose_results[0]['keypoints'])
    kpt_adj = np.zeros(shape=(kpt_len, kpt_len), dtype='int64')
    adj_distance = []
    kpt_matching = []
    skeleton = []

    for pred_kpt in pose_results:
        person_kpt = copy.deepcopy(pred_kpt['keypoints'])
        adj_dist = np.zeros(shape=(len(person_kpt), len(person_kpt)))
        limblist = []


        for kid, loc in enumerate(person_kpt):
            if kid == len(person_kpt)-1:
                break
            target_kid = kid+1
            
            while target_kid < len(person_kpt):
                dist_x = loc[0] - person_kpt[target_kid][0]
                dist_y = loc[1] - person_kpt[target_kid][1]

                dist = math.sqrt(pow(dist_x, 2)+pow(dist_y, 2))
                adj_dist[kid][target_kid], adj_dist[target_kid][kid] = dist, dist
                target_kid += 1

            adj_dist[kid][kid] = 0

            # if kid < len(person_kpt)-1:
            if kid in one_link_kpt:
                link_point = 1
            elif kid in two_link_kpt:
                link_point = 2
            elif kid in three_link_kpt:
                link_point = 3
            elif kid in four_link_kpt:
                link_point = 4
            
            # print(f'link point: {link_point}')
            idx_interval = kid+1
            kid_dist = copy.deepcopy(adj_dist[kid][idx_interval:])
            # print(kid_dist)
            count_nonzero = np.count_nonzero(kpt_adj[kid][:idx_interval]==1)
            if link_point -  count_nonzero > 0:
                link_point = link_point - count_nonzero

                for _ in range(link_point):
                    min_dist = np.min(kid_dist)
                    min_dist_kpt = np.argmin(kid_dist)
                    origin_min_idx = min_dist_kpt + idx_interval

                    kid_dist[min_dist_kpt] = float('inf')
                    kpt_adj[kid][origin_min_idx], kpt_adj[origin_min_idx][kid] = 1, 1

                    if not [origin_min_idx+1, kid+1] in limblist:
                        limblist.append([kid+1, origin_min_idx+1])

        adj_distance.append(adj_dist)
        skeleton.append(limblist)
        # print(adj_dist)
        # print(kpt_adj) 

        # print("gt skeleton")
        # print(gt_skeleton)
        # print("---"*30)
        # print("pred skeleton")
        # print(skeleton)
        # exit()

    return skeleton


def get_graph_loss(gt, pred):
    assert np.shape(gt) == np.shape(pred)
    # kpt 하나 추출
    # 추출된 keypoint와 나머지 keypoint 간의 단순 거리 계산
    
    '''
    1: 2 - [1, 2], [1, 3]
    2: 3 - (2, 1), [2, 3], [2, 4]
    3: 3 - (3, 1), (3, 2), [3, 5]
    4: 2 - (4, 2), [4, 6]
    5: 2 - (5, 3), [5, 7]
    6: 4 - (6, 4), [6, 7], [6, 8], [6, 12]
    7: 4 - (7, 5), (7, 6). [7, 9], [7, 13]
    8: 2 - (8, 6), [8, 10]
    9: 2 - (9, 7), [9, 11]
    10: 1 - (10, 8)
    11: 1 - (11, 9)
    12: 3 - (12 ,6), [12, 13], [12, 14]
    13: 3 - (13, 7), (13, 12), [13 ,15]
    14: 2 - (14, 12), [14, 16]
    15: 2 - (15, 13), [15, 17]
    16: 1 - (15, 13)
    17: 1 - (16, 14)
    '''
    gt_skeleton = [
        [1,2], [1,3],[2,3],[2,4],
        [3,5], [4,6], [5,7],
        [6,7], [6,8], [6, 12],
        [7,9], [7, 13],
        [8, 10], [9, 11],
        [12, 13], [12, 14], [14, 16],
        [13, 15], [15, 17]
    ]

    one_link_kpt = [9, 10, 15, 16]
    two_link_kpt = [0, 3, 4, 7, 8, 13, 14]
    three_link_kpt = [1, 2, 11, 12]
    four_link_kpt = [5, 6]

    batch_len = len(gt)
    kpt_len = len(gt[0])
    pred_adj = np.zeros(shape=(kpt_len, kpt_len), dtype='int64')
    pred_dist = []
    pred_adj = []

    gt_dist_adj = []
    gt_adj = []
    adj_distance = []
    kpt_matching = []
    skeleton = []

    for img_idx, person in enumerate(pred):
        pred_tmp = np.zeros(shape=(kpt_len, kpt_len))

        '''
        person: [x, y, score]
        '''
        for kid, loc in enumerate(person): # start from one keypoint
            if kid == kpt_len-1:
                break
            target_kid = kid+1
            
            while target_kid < kpt_len:
                dist_x = loc[0] - person[target_kid][0]
                dist_y = loc[1] - person[target_kid][1]

                dist = math.sqrt(pow(dist_x, 2)+pow(dist_y, 2))
                pred_dist[kid][target_kid], pred_dist[target_kid][kid] = dist, dist
                target_kid += 1

            pred_dist[kid][kid] = 0

            # if kid < len(person_kpt)-1:
            if kid in one_link_kpt:
                link_point = 1
            elif kid in two_link_kpt:
                link_point = 2
            elif kid in three_link_kpt:
                link_point = 3
            elif kid in four_link_kpt:
                link_point = 4
            
            # print(f'link point: {link_point}')
            idx_interval = kid+1
            kid_dist = copy.deepcopy(pred_dist[kid][idx_interval:])
            # print(kid_dist)
            count_nonzero = np.count_nonzero(pred_adj[kid][:idx_interval]==1)
            if link_point -  count_nonzero > 0:
                link_point = link_point - count_nonzero

                for _ in range(link_point):
                    min_dist = np.min(kid_dist)
                    min_dist_kpt = np.argmin(kid_dist)
                    origin_min_idx = min_dist_kpt + idx_interval

                    kid_dist[min_dist_kpt] = float('inf')
                    pred_adj[kid][origin_min_idx], pred_adj[origin_min_idx][kid] = 1, 1

                    if not [origin_min_idx+1, kid+1] in limblist:
                        limblist.append([kid+1, origin_min_idx+1])

        
        # gt_kpt, pred_kpt = gt[img_idx], pred[img_idx]


        # for kid, (g_k, p_k) in enumerate(zip(gt_kpt, pred_kpt)):
        #     if kid == kpt_len - 1:
        #         break
        #     target_kid = kid+1

        #     while target_kid < kpt_len:
        #         gt_dist_x = g_k[0] = gt_kpt[target_kid][0]
        #         gt_dist_y = g_k[1] = gt_kpt[target_kid][1]

        #         gt_dist = math.sqrt(pow(gt_dist_x, 2) + pow(gt_dist_y, 2))
        #         gt_dist_adj[kid][target_kid], gt_dist_adj[target_kid][kid] = gt_dist, gt_dist

        #         pred_dist_x = p_k[0] = pred_kpt[target_kid][0]
        #         pred_dist_y = p_k[1] = pred_kpt[target_kid][1]

        #         pred_dist = math.sqrt(pow(pred_dist_x, 2) + pow(pred_dist_y, 2))
        #         pred_dist_adj[kid][target_kid], pred_dist_adj[target_kid][kid] = pred_dist, pred_dist

        #         target_kid += 1
        #     gt_dist_adj[kid][kid], pred_dist_adj[kid][kid] = 0, 0

        f

        adj_distance.append(adj_dist)
        skeleton.append(limblist)
        # print(adj_dist)
        # print(kpt_adj) 

        # print("gt skeleton")
        # print(gt_skeleton)
        # print("---"*30)
        # print("pred skeleton")
        # print(skeleton)
        # exit()

    

    return skeleton

