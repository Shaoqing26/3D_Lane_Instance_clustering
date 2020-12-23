# 3D_Lane_Instance_clustering
Instance lane ID after extract 3d lane pointcloud

## Before start this project
  1.You should get pure 3d_pointcloud of lane with Algrithum e.g. PointNet++  
  Just like :
  ![raw1](https://github.com/Shaoqing26/3D_Lane_Instance_clustering/blob/main/pcd_deal/result_IMG/raw.png)
  ![raw2](https://github.com/Shaoqing26/3D_Lane_Instance_clustering/blob/main/pcd_deal/result_IMG/raw_2.png)
## Main deal with project
1.Use the tradition way to get instance lane id.  
2.use the simplest idea to finish work without Deeplearning

## AIMS
1.when we get pure-3d-lane Pointcloud and we wanna to distinct the lane id instance  

## Result
![cluster_all](https://github.com/Shaoqing26/3D_Lane_Instance_clustering/blob/main/pcd_deal/result_IMG/line_Instance_Seg.png)

![line_instance_seg](https://github.com/Shaoqing26/3D_Lane_Instance_clustering/blob/main/pcd_deal/result_IMG/cluster_all.png)
![line_seg](https://github.com/Shaoqing26/3D_Lane_Instance_clustering/blob/main/pcd_deal/result_IMG/result_line.png)
![curve_seg](https://github.com/Shaoqing26/3D_Lane_Instance_clustering/blob/main/pcd_deal/result_IMG/curve_seg.png)

# we can see result img above,instance lane id has individual coulor

## How do i get the result
### important api : line Ransac and DBSCAN cluster
1.segment one lane pointcloud to more  
2.use dbscan cluster one of pice PC and use ransac to regress line   
3.use the line function to rotated all cloud make it vertical with X  
4.resegment all cloud to pices and recluster and ransac  
5. we concat with zip and sorted in code
  
  FOR MORE INFO AND DETAIL IN CODE.
