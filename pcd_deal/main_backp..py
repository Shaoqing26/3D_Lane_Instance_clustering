import argparse
from pypcd import pypcd
import numpy as np
import open3d
import random
import math
import pandas as pd
# import geopandas as gp
import matplotlib.pyplot as plt
from sklearn import linear_model
from itertools import zip_longest
from shapely import wkt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

parser = argparse.ArgumentParser()
parser.add_argument('--pcd_path', default='', type=str)
args = parser.parse_args()

class lane_ID:
    def __init__(self):
        self.min_points_x = .0
        self.min_points_y = .0
        self.concat_all_PC  = []

    def read_pcd(self,pcd_path):
        pcd = pypcd.PointCloud.from_path(pcd_path)
        pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
        print(pcd.pc_data['x'])
        pcd_np_points[:, 0] = np.transpose(pcd.pc_data['x'])
        pcd_np_points[:, 1] = np.transpose(pcd.pc_data['y'])
        pcd_np_points[:, 2] = np.transpose(pcd.pc_data['z'])
        pcd_np_points[:, 3] = np.transpose(pcd.pc_data['rgb'])

        xy_pc = pcd_np_points[:,:3]
        self.min_points_x,self.min_points_y = np.min(xy_pc[:,:2],axis = 0)
        xy_pc = xy_pc - np.append(np.min(xy_pc[:,:2],axis = 0),0)
        return xy_pc
    def vis(self,point):
        pcobj = open3d.geometry.PointCloud()
        pcobj.points = open3d.utility.Vector3dVector(point)

        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1
        opt.line_width = 100
        opt.show_coordinate_frame = True
        seg = []
        w = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 1, 0])
        seg.append(w)
        seg.append(pcobj)

        open3d.visualization.draw_geometries(seg)
    def ransac_line(self,points):
        # model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), max_trials=2000,
                                              min_samples=10,
                                              residual_threshold=0.1,
                                              stop_probability=0.999,
                                              random_state=0)
        X = points[:, 0]
        y = points[:, 1]
        ransac.fit(X.reshape(-1,1),y)

        print('Slope:%.3f;Intercept:%.3f' % (ransac.estimator_.coef_[0], ransac.estimator_.intercept_))


        inlier_mask = ransac.inlier_mask_  # 内点掩码
        #    print(inlier_mask)
        outlier_mask = np.logical_not(inlier_mask)  # 外点掩码
        line_X = np.arange(14, 18, 0.2)
        line_y = np.arange(0, 100, 0.2)
        plt.xlim(0, 100)
        plt.ylim(0, 120)
        line_y_ransac = ransac.predict(line_X[:, np.newaxis])
        plt.scatter(X,y)
        plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
        plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='OutLiers')
        plt.plot(line_X, line_y_ransac, color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper left')
        plt.show()
        return ransac.estimator_.coef_[0], ransac.estimator_.intercept_
    def cluster(self,points):
        X = points[:,0:2]
        X = StandardScaler().fit_transform(X)

        # db = DBSCAN(eps=0.05, min_samples=10).fit(X)
        db = DBSCAN(eps=0.1, min_samples=8).fit(X)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        return labels
    def labels_rvz(self,labels,points):
        pcobj = open3d.geometry.PointCloud()
        pcobj.points = open3d.utility.Vector3dVector(points[:, 0:3])
        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1
        opt.line_width = 150
        opt.show_coordinate_frame = True
        show_pc = []

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        for i in range(n_clusters_):
            labels_points = np.where(labels == i)
            pc = pcobj.select_down_sample(list(labels_points[0]))
            pc.paint_uniform_color([random.uniform(0,1.0) for _ in range(3)] )
            show_pc.append(pc)
        w = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 1, 0])
        show_pc.append(w)
        open3d.visualization.draw_geometries(show_pc)
        # vis.draw_geometries(show_pc)
        # show_pc.clear()
    def rotate_pc_(self,K,intercept_,points):
        ''' n * 3 * 3*3
        which 3*3 is:
                cosθ     -sinθ    0
                sinθ    cosθ    0
                0        0       1

        '''
        ang = math.degrees(math.atan(K))
        rad = math.atan(K)
        ang_radia = math.pi/2 - rad

        print("We need rotated {} deegrees".format(ang))

        if rad <= 0:
            ang_radia = -(math.pi/2 + rad)
        elif rad > 0 :
            ang_radia = math.pi/2 - rad
        else:
            print('ang fault:',rad)
            assert ang_radia not in range(-math.pi/2,math.pi/2)
        matrix = np.zeros((3,3))
        matrix[0][0] = math.cos(ang_radia)
        matrix[0][1] = -math.sin(ang_radia)
        matrix[1][0] = math.sin(ang_radia)
        matrix[1][1] = math.cos(ang_radia)
        matrix[2][2] = 1

        new_Pc = np.transpose(np.dot(matrix,np.transpose(points)))
        return new_Pc
    def split_PC(self,points,per_size):
        max_y = np.max(points[:,1],axis = 0)
        grid_num = math.ceil(max_y/per_size)
        grid_PC  = []

        for i in range(0,grid_num):
            new_pc = points[np.where((points[:,1] < per_size *(i+1)) &
                                           (points[:,1] >= per_size *i))]
            if len(new_pc) > 100:
                grid_PC.append(new_pc)
            else:
                continue

        return grid_PC
    def caculate_x_mean(self,points):
        return np.mean(points[:,0])
    def merge_ID(self,grid_PC,labels_All_PC):
        # grid_PC = grid_PC +
        pice_labels_PC = []
        labels_PC = []
        pice_labels_PC_L_ = []
        mean_x_ = []
        for i in range(0,len(labels_All_PC)):
            n_clusters_ = len(set(labels_All_PC[i])) - (1 if -1 in labels_All_PC[i] else 0)
            #sub_set of one pcd
            for j in range(0,n_clusters_):
                labels_points_idx = np.where(labels_All_PC[i] == j)
                pice_labels_PC.append(grid_PC[i][labels_points_idx])
                pice_labels_PC_L_.append(all_pc_rotated[i][labels_points_idx])
                # self.rvis_All_Labels(pice_labels_PC)
                # mean_x_.append(caculate_x_mean(points[labels_points_idx]))
            # to do : sort with mean_x
            # pice_labels_PC.sort(key=lambda t:np.mean(t[:,0]))
            a = zip(pice_labels_PC,pice_labels_PC_L_)
            b = sorted(a,key=lambda x:np.mean(x[1][:,0]))
            # pice_labels_PC.sort(key=lambda t:all_pc_rotated[i][:,0].mean)

            # labels_PC.append(pice_labels_PC.copy())
            labels_PC.append(list(zip(*b))[0])
            pice_labels_PC.clear()
            pice_labels_PC_L_.clear()
            # for sublist in zip(*a):
            #     cat_list.append(np.concatenate(sublist))
        # cat_list = [ np.concatenate(i) for i in zip(*labels_PC)]
        cat_list = [ np.concatenate(i) for i in zip_longest(*labels_PC,fillvalue=np.empty((0,3)))]
        for g in range(0,len(cat_list)):
            cat_list[g] = cat_list[g] + [self.min_points_x,self.min_points_y,0]
        # np.save('data_.npy',labels_PC)
        self.concat_all_PC.append(cat_list)
        return cat_list
    # def merge_cloud(labels_PC_new):
    #     for i in range(0,len(labels_PC_new)):
    def rvis_All_Labels(self,points):
        seg = []
        for i in range(0,len(points)):

            pcobj = open3d.geometry.PointCloud()
            pcobj.points = open3d.utility.Vector3dVector(points[i])
            pcobj.paint_uniform_color([random.uniform(0, 1.0) for _ in range(3)])
            seg.append(pcobj)

        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1
        opt.line_width = 100
        opt.show_coordinate_frame = True
        w = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 1, 0])
        seg.append(w)
        open3d.visualization.draw_geometries(seg)
    def vis_All_Concat_PC(self):
        cat_list = [np.concatenate(i) for i in zip_longest(*self.concat_all_PC,fillvalue=np.empty((0,3)))]
        self.rvis_All_Labels(cat_list)
'''
if __name__ == '__main__':

    points = read_pcd('/home/ral/BD_Work/Lane_seg/Exstract_lane/cmake-build-debug/20200603095539_YIFEIYANG_SHANGHAI_DSX116_LiDAR_2_0435.pcd')
    vis(points)
    labels = cluster(points)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    max = 0
    id = 0
    labels_points_id = []
    for i in range(n_clusters_):
        labels_points = np.where(labels == i)
        if len(labels_points[0]) > max :
            id = i
            labels_points_id = labels_points[0]
            max = len(labels_points[0])
    # vis(points[labels_points_id,:])
    K,intercept_ =ransac_line(points[labels_points_id,:2])


    point_Rotated_Pc = rotate_pc_(K,intercept_,points)
    vis(point_Rotated_Pc)
    point_Rotated_Pc[:, 1] = 0
    labels = cluster(point_Rotated_Pc)
    labels_rvz(labels,points)
'''
import os
if __name__ == '__main__':
    data_dir = '/home/ral/BD_Work/Lane_seg/pcd_deal/lane_PC'
    a = lane_ID()
    assert os.path.exists(data_dir)
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        print("Current_FileName: {}".format(file))

        points = a.read_pcd(file_path)
        # points = a.read_pcd('/home/ral/BD_Work/Lane_seg/pcd_deal/lane_PC/20200603095539_YIFEIYANG_SHANGHAI_DSX116_LiDAR_2_0441.pcd')
        # a.vis(points)

        grids_pc = a.split_PC(points,25)

        labels_m = {}
        points_m = {}
        all_pices_labels = []
        all_pc_rotated = []
        for i in range(0,len(grids_pc)):
            # vis(grids_pc[i])
            labels = a.cluster(grids_pc[i])
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            max = 0
            id = 0
            labels_points_id = []
            for j in range(n_clusters_):
                labels_points = np.where(labels == j)
                if len(labels_points[0]) > max :
                    id = j
                    labels_points_id = labels_points[0]
                    max = len(labels_points[0])
            # vis(points[labels_points_id,:])
            K,intercept_ =a.ransac_line(grids_pc[i][labels_points_id,:2])

            point_Rotated_Pc = a.rotate_pc_(K,intercept_,grids_pc[i])

            # vis(point_Rotated_Pc)
            point_Rotated_Pc[:, 1] = 0
            labels = a.cluster(point_Rotated_Pc)
            all_pices_labels.append(labels)
            all_pc_rotated.append(point_Rotated_Pc)
            #labels_rvz(labels,grids_pc[i])
        all_Done_PC = a.merge_ID(grids_pc,all_pices_labels)
        # a.rvis_All_Labels(all_Done_PC)
    a.vis_All_Concat_PC()



