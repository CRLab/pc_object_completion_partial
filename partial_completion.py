#! /usr/bin/env python

import rospy
import actionlib
import pc_pipeline_msgs.msg
import mcubes
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import collada
import shape_msgs.msg
import geometry_msgs.msg
import tempfile
import subprocess 
import rospkg
from curvox import mesh_conversions
import os 

PERCENT_PATCH_SIZE = (4.0/5.0)
PERCENT_X = 0.5
PERCENT_Y = 0.5
PERCENT_Z = 0.45

def get_voxel_resolution(pc, patch_size):
    assert pc.shape[1] == 3
    min_x = pc[:, 0].min()
    min_y = pc[:, 1].min()
    min_z = pc[:, 2].min()
    max_x = pc[:, 0].max()
    max_y = pc[:, 1].max()
    max_z = pc[:, 2].max()

    max_dim = max((max_x - min_x),
                  (max_y - min_y),
                  (max_z - min_z))

    print max_dim
    print PERCENT_PATCH_SIZE
    print patch_size

    voxel_resolution = (1.0*max_dim) / (PERCENT_PATCH_SIZE * patch_size)
    return voxel_resolution


def get_center(pc):

    assert pc.shape[1] == 3
    min_x = pc[:, 0].min()
    min_y = pc[:, 1].min()
    min_z = pc[:, 2].min()
    max_x = pc[:, 0].max()
    max_y = pc[:, 1].max()
    max_z = pc[:, 2].max()

    center = (min_x + (max_x - min_x) / 2.0,
              min_y + (max_y - min_y) / 2.0,
              min_z + (max_z - min_z) / 2.0)

    return center


def create_voxel_grid_around_point_scaled(points, patch_center,
                                          voxel_resolution, num_voxels_per_dim,
                                          pc_center_in_voxel_grid):
    voxel_grid = np.zeros((num_voxels_per_dim,
                           num_voxels_per_dim,
                           num_voxels_per_dim,
                           1), dtype=np.float32)

    centered_scaled_points = np.floor(
        (points - np.array(patch_center) + np.array(
            pc_center_in_voxel_grid) * voxel_resolution) / voxel_resolution)

    mask = centered_scaled_points.max(axis=1) < num_voxels_per_dim
    centered_scaled_points = centered_scaled_points[mask]

    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    mask = centered_scaled_points.min(axis=1) > 0
    centered_scaled_points = centered_scaled_points[mask]

    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    csp_int = centered_scaled_points.astype(int)

    mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2],
            np.zeros((csp_int.shape[0]), dtype=int))

    voxel_grid[mask] = 1

    return voxel_grid

def rescale_mesh(vertices,
            patch_center,
            voxel_resolution,
            pc_center_in_voxel_grid):
    
    # Reverse of the following function solve for points
    # vertices = (points - np.array(patch_center) + np.array(pc_center_in_voxel_grid) * voxel_resolution) / voxel_resolution
    return vertices * voxel_resolution - np.array(pc_center_in_voxel_grid) * voxel_resolution + np.array(patch_center)

class ObjectCompletionAction(object):
    # create messages that are used to publish feedback/result
    _feedback = pc_pipeline_msgs.msg.CompletePartialCloudFeedback()
    _result = pc_pipeline_msgs.msg.CompletePartialCloudResult()

    

    def __init__(self, name):
        self._action_name = name

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("pc_object_completion_partial")
        self.mlx_script_filepath = pkg_path + "/smooth_partial.mlx"
        self._as = actionlib.SimpleActionServer(self._action_name, 
                                                pc_pipeline_msgs.msg.CompletePartialCloudAction, 
                                                execute_cb=self.execute_cb, 
                                                auto_start = False)
        self._as.start()
    

    
    def execute_cb(self, goal):
        rospy.loginfo("received new goal")
        points = []

        gen = pc2.read_points(goal.partial_cloud, skip_nans=True, field_names=("x", "y", "z"))
        for p in gen:
            points.append(p)
            
        pc = np.array(points)
        patch_size = 120
        vox_resolution = get_voxel_resolution(pc, patch_size)

        center = get_center(pc)
        
        pc_center_in_voxel_grid = (patch_size*PERCENT_X, patch_size*PERCENT_Y, patch_size*PERCENT_Z)

        voxel_grid = create_voxel_grid_around_point_scaled(pc, 
                                            center,
                                            vox_resolution, 
                                            patch_size,
                                            pc_center_in_voxel_grid)
        
        rospy.loginfo("about to run mcubes")
        v, t = mcubes.marching_cubes(voxel_grid[:,:,:,0], 0.5)
        v = rescale_mesh(v, 
            center,
            vox_resolution, 
            pc_center_in_voxel_grid)

        unsmoothed_handle, unsmoothed_filename = tempfile.mkstemp(suffix=".dae")
        smoothed_handle, smoothed_filename = tempfile.mkstemp(suffix=".ply")
        mcubes.export_mesh(v, t, unsmoothed_filename, "model")
        cmd_str = "meshlabserver -i " + unsmoothed_filename + " -o " + smoothed_filename + " -s " + str(self.mlx_script_filepath) 
        subprocess.call(cmd_str.split())

        
        mesh = mesh_conversions.read_mesh_msg_from_ply_filepath(smoothed_filename)

        if os.path.exists(unsmoothed_filename):
            os.remove(unsmoothed_filename)
        if os.path.exists(smoothed_filename):
            os.remove(smoothed_filename)
        
        self._result.mesh = mesh
        rospy.loginfo('Succeeded')
        self._as.set_succeeded(self._result)


if __name__ == '__main__':
    rospy.init_node('partial_object_completion')
    server = ObjectCompletionAction('partial_object_completion')
    rospy.spin()
