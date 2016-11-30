# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:58:09 2016

@author: benjamin
"""

import json
import numpy as np
import h5py
import os

def test_parametric_pose_NMS(delta1,delta2,mu,gamma):
    scoreThreds = 0.0
    
    #prepare data
    h5file = h5py.File("./preds/test.h5",'r')
    preds = np.array(h5file['preds'])
    scores = np.array(h5file['scores'])    
    indexs = [line.rstrip(' ').rstrip('\r').rstrip('\n') for line in open("annot/mpii-test0.09/index.txt")]
    scores_proposals = np.loadtxt("annot/mpii-test0.09/score.txt")
    #get bounding box sizes    
    bbox_file = h5py.File("annot/mpii-test0.09/test-bbox.h5",'r')
    xmax=np.array(bbox_file['xmax']); xmin=bbox_file['xmin']; ymax=np.array(bbox_file['ymax']); ymin=bbox_file['ymin']
    widths=xmax-xmin; heights=ymax-ymin;
    alpha = 0.1
    Sizes=alpha*np.maximum(widths,heights)
    
    #set the corresponding dir
    if (os.path.exists("./NMS") == False):
        os.mkdir("./NMS")

    os.chdir("./NMS")
    
    NMS_preds = open("pred.txt",'w')
    NMS_scores = open("scores.txt",'w')
    proposal_scores = open("scores-proposals.txt",'w')
    NMS_index = open("index.txt",'w')
    num_human = 0
    
    #loop through every image
    for i in xrange(len(indexs)):
        index = indexs[i].split(' '); 
        img_name = index[0]; start = int(index[1])-1; end = int(index[2])-1;
        
        #initialize scores and preds coordinates
        img_preds = preds[start:end+1]; img_scores = np.mean(scores[start:end+1],axis = 1)
        img_ids = np.arange(end-start+1); ref_dists = Sizes[start:end+1];keypoint_scores = scores[start:end+1];
        pro_score = scores_proposals[start:end+1]
        
        #do NMS by parametric
        pick = []
        merge_ids = []
        while(img_scores.size != 0):
            
            #pick the one with highest score
            pick_id = np.argmax(img_scores)  
            pick.append(img_ids[pick_id])
            
            #get numbers of match keypoints by calling PCK_match 
            ref_dist=ref_dists[img_ids[pick_id]]
            simi = get_parametric_distance(pick_id,img_preds, keypoint_scores,ref_dist, delta1, delta2, mu)
            
            #delete humans who have more than matchThreds keypoints overlap with the seletced human.
            delete_ids = np.arange(img_scores.shape[0])[simi > gamma]
            if (delete_ids.size == 0):
                delete_ids = pick_id
            merge_ids.append(img_ids[delete_ids])
            img_preds = np.delete(img_preds,delete_ids,axis=0); img_scores = np.delete(img_scores, delete_ids)
            img_ids = np.delete(img_ids, delete_ids); keypoint_scores = np.delete(keypoint_scores,delete_ids,axis=0)
        
        #write the NMS result to files
        pick = [Id+start for Id in pick] 
        merge_ids = [Id+start for Id in merge_ids]
        assert len(merge_ids) == len(pick)
        preds_pick = preds[pick]; scores_pick = scores[pick];
        num_pick = 0
        for j in xrange(len(pick)):
            
            #first compute the average score of a person
            ids = np.arange(16)
            if (scores_pick[j,0,0] < 0.1): ids = np.delete(ids,0);
            if (scores_pick[j,5,0] < 0.1): ids = np.delete(ids,5);
            mean_score = np.mean(scores_pick[j,ids,0])
            if (mean_score < scoreThreds):
                continue
            
            # merge poses
            merge_id = merge_ids[j]  
            score = scores_proposals[pick[j]]
            merge_poses,merge_score = merge_pose(preds_pick[j],preds[merge_id],scores[merge_id],Sizes[pick[j]])
            
            ids = np.arange(16)
            if (merge_score[0] < 0.1): ids = np.delete(ids,0);
            if (merge_score[5] < 0.1): ids = np.delete(ids,5);
            mean_score = np.mean(merge_score[ids])
            if (mean_score < scoreThreds):
                continue
            
            #add the person to predict
            num_pick += 1
            NMS_preds.write("{}".format(img_name))
            NMS_scores.write("{}".format(img_name))
            proposal_scores.write("{}\n".format(score))
            
            for point_id in xrange(16):
                NMS_preds.write("\t{}\t{}".format(int(merge_poses[point_id,0]),int(merge_poses[point_id,1])))
                NMS_scores.write("\t{}".format(merge_score[point_id]))
            NMS_preds.write("\n")
            NMS_scores.write("\n")
        NMS_index.write("{} {} {}\n".format(img_name, num_human+1, num_human + num_pick))
        num_human += num_pick
        
    NMS_preds.close();NMS_scores.close();NMS_index.close(); proposal_scores.close()
    
def get_parametric_distance(i,all_preds, keypoint_scores,ref_dist, delta1, delta2, mu):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = np.sqrt(np.sum(np.square(pick_preds[np.newaxis,:]-all_preds),axis=2))/ref_dist
    mask = (dist <= 1)
    # defien a keypoints distances
    score_dists = np.zeros([all_preds.shape[0], 16])
    keypoint_scores = np.squeeze(keypoint_scores)
    if (keypoint_scores.ndim == 1) :
        keypoint_scores = keypoint_scores[np.newaxis,:]
    # the predicted scores are repeated up to do boastcast
    pred_scores = np.tile(pred_scores, [1,all_preds.shape[0]]).T
    score_dists[mask] = np.tanh(pred_scores[mask]/delta1)*np.tanh(keypoint_scores[mask]/delta1)
    # if the keypoint isn't inside the bbox, set the distance to be 10
#    dist[dist>1] = 10
    point_dist = np.exp((-1)*dist/delta2)
    final_dist = np.sum(score_dists,axis=1)+mu*np.sum(point_dist,axis=1)
    return final_dist
    
def merge_pose(refer_pose, cluster_preds, cluster_keypoint_scores, ref_dist):
    dist = np.sqrt(np.sum(np.square(refer_pose[np.newaxis,:]-cluster_preds),axis=2))
    # mask is an nx16 matrix
    mask = (dist <= ref_dist)
    final_pose = np.zeros([16,2]); final_scores = np.zeros(16)
    if (cluster_preds.ndim == 2):
        cluster_preds = cluster_preds[np.newaxis,:,:]
        cluster_keypoint_scores = cluster_keypoint_scores[np.newaxis,:]
    if (mask.ndim == 1):
        mask = mask[np.newaxis,:]
    for i in xrange(16):
        cluster_joint_scores = cluster_keypoint_scores[:,i][mask[:,i]]
        
        # pick the corresponding i's matched keyjoint locations and do an weighed sum.
        cluster_joint_location = cluster_preds[:,i,:][np.tile(mask[:,i,np.newaxis],(1,2))].reshape(np.sum(mask[:,i,np.newaxis]),-1)

        # get an normalized score
        normed_scores = cluster_joint_scores / np.sum(cluster_joint_scores)
        # merge poses by a weighted sum
        final_pose[i,0] = np.dot(cluster_joint_location[:,0], normed_scores)
        final_pose[i,1] = np.dot(cluster_joint_location[:,1], normed_scores)
        final_scores[i] = np.max(cluster_joint_scores)
    return final_pose, final_scores
    

def get_result():
    delta1 = 0.01; mu = 2.08; delta2 = 2.08;
    gamma = 22.48;
    test_parametric_pose_NMS(delta1, delta2, mu, gamma)

get_result()