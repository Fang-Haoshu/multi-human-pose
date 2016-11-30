function setpred()
filedir = './NMS/';

load('./mpii_human_pose_v1_u12_1.mat','RELEASE')
annolist_test = RELEASE.annolist(RELEASE.img_train == 0);
load('./groups_v12.mat','groups');
[imgidxs_multi_test,rectidxs_multi_test] = getMultiPersonGroups(groups,RELEASE,false);
annolist_test_multi = annolist_test(imgidxs_multi_test);
% multi-person rectangles
for imgidx = 1:length(annolist_test_multi)
   annolist_test_multi(imgidx).annorect = annolist_test_multi(imgidx).annorect(rectidxs_multi_test{imgidx});
end
        

pred_file = 'pred.txt';
scores_file = 'scores.txt';
index_file = 'index.txt';

part = fopen([filedir pred_file],'r');
scores_id = fopen([filedir scores_file],'r');
index = fopen([filedir index_file],'r');

textformat = '%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d';
pred_points = textscan(part,textformat);

textformat = '%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f';
pred_scores = textscan(scores_id, textformat);

textformat = '%s %d %d';
part_index = textscan(index,textformat);

for i = 1:numel(part_index{2})
    %load ground truth group annotation. 
     GTgroup = annolist_test_multi(i);
     % the bounding box of a group
     xmin=8000;ymin=8000;xmax= -8000;ymax= -8000;
     for j = 1:numel(GTgroup.annorect)
         x = GTgroup.annorect(j).objpos.x; y = GTgroup.annorect(j).objpos.y; 
         scale =GTgroup.annorect(j).scale;
         x1=x-50*scale;y1=y-50*scale; x2=x+50*scale;y2=y+50*scale;
         if (x1<xmin) xmin=x1; end; if (y1<ymin) ymin=y1; end
         if (x2>xmax) xmax=x2; end; if (y2>ymax) ymax=y2; end
     end
    
    start = part_index{2}(i); endpoint = part_index{3}(i);
    pred(i).image.name = part_index{1}{i};
    pred_lines = 1;
    for j = start:endpoint

        % if the object center is outside the bounding box, discount this
        % object.
         obj_center.x = (pred_points{2*7}(j)+pred_points{2*8}(j))/2;
         obj_center.y = (pred_points{2*7+1}(j)+pred_points{2*8+1}(j))/2;
         if (obj_center.x < xmin || obj_center.x > xmax || obj_center.y < ymin || obj_center.y > ymax)
             continue 
        end

            ids = 1;
            for k = 1:16
                pred(i).annorect(pred_lines).annopoints.point(ids).x = double(pred_points{2*k}(j));
                pred(i).annorect(pred_lines).annopoints.point(ids).y = double(pred_points{2*k+1}(j));
                pred(i).annorect(pred_lines).annopoints.point(ids).id = k-1;
                pred(i).annorect(pred_lines).annopoints.point(ids).score = pred_scores{k+1}(j);

                ids = ids+1;
            end
            pred_lines = pred_lines + 1;

    end
end
save('pred_keypoints_mpii_multi.mat','pred');
fclose(part); fclose(scores_id); fclose(index);

end