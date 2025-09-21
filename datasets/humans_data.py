import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import open3d as o3d
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
from utils.file_utils import *

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py

synsetid_to_cate = {
    'Subject001': 'Subject001',
    'Subject002': 'Subject002',
    'Subject003': 'Subject003',
    'Subject004': 'Subject004'
}

cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15KPC(Dataset):
    def __init__(self, root_dir, subdirs, tr_sample_size=10000,
                 te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, box_per_shape=False,
                 random_subsample=False,
                 normalize_std_per_axis=False,
                 normalize_trans=True, normalize_az=True, all_points_mean=None, 
                 all_points_std=None, input_dim=3, 
                 use_mask=False, ntrain=False, 
                 pose_format='angles', output_dir=None):
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs[0]
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.use_mask = use_mask
        self.box_per_shape = box_per_shape

        if use_mask:
            self.mask_transform = PointCloudMasks(radius=5, elev=5, azim=90)

        print ("Reading data...........", os.path.join(root_dir, self.subdirs))
        # directly extract point clouds from ply files
        if normalize_az:
            ply_files = natsorted(os.listdir(os.path.join(root_dir, self.subdirs)))
        else:
            ply_files = natsorted(os.listdir(os.path.join(root_dir, self.subdirs))) #[:200]

        # read pose joints file
        # root joint (x, y, z) --> (43, 44, 45)
        pose_dir = "/CT/siddharth/work/data/" + self.subdirs + "/new_capture"
        if normalize_az:
            with open(os.path.join(pose_dir, "data_export_normalized.mddd"), "r") as f:
                pose_joints_lines = f.readlines()
        else:
            with open(os.path.join(pose_dir, "data_export.mddd"), "r") as f:
                pose_joints_lines = f.readlines()
        pose_joints_lines = pose_joints_lines[1:]

        print (len(pose_joints_lines))
        
        if pose_format == 'angles':
            # read pose angles file
            if normalize_az:
                with open("/CT/siddharth/work/data/"+self.subdirs+"/new_capture/poseAnglesRotationNormalized.motion", "r") as f:
                    pose_lines = f.readlines()
            else:
                with open("/CT/siddharth/work/data/"+self.subdirs+"/new_capture/poseAngles.motion", "r") as f:
                    pose_lines = f.readlines()
            pose_lines = pose_lines[1:]
        elif pose_format == 'joints':
            pose_lines = pose_joints_lines.copy()
        else:
            raise NotImplementedError                 

        print ("ntrain ", ntrain)

        if ntrain is not None:
            s, e, i = ntrain
            ply_files = ply_files[s:e:i]
            pose_joints_lines = pose_joints_lines[s:e:i]
            pose_lines = pose_lines[s:e:i]

        print ("Num of ply files to read", len(ply_files))
        print (ply_files[:5])

        print (ply_files[-5:])

        self.all_points = []
        self.all_colors = []
        self.all_normals = []
        for ply_fname in tqdm(ply_files):

            try:
                ply_fpath = os.path.join(root_dir, self.subdirs, ply_fname)
                point_cloud_obj = o3d.io.read_point_cloud(ply_fpath)
                point_cloud = np.asarray(point_cloud_obj.points) # (10k, 3)
                point_colors = np.asarray(point_cloud_obj.colors)
                point_normals = np.asarray(point_cloud_obj.normals)
            except:
                print ("error")
                continue

            self.all_points.append(point_cloud[np.newaxis, ...])
            self.all_colors.append(point_colors[np.newaxis, ...])
            self.all_normals.append(point_normals[np.newaxis, ...])
        

        def parse_pose_files(pose_lines):
            all_pose = []
            for pl in pose_lines:
                da = pl.split(" ")
                da_new = list(filter(None, da))
                da_new_np = np.array(da_new, dtype=np.float32)
                all_pose.append(da_new_np[np.newaxis, ...])
            return all_pose

        self.all_joints = parse_pose_files(pose_joints_lines)
        self.all_pose = parse_pose_files(pose_lines)

        if self.split == 'train':
            # Shuffle the index deterministically (based on the number of examples)
            self.shuffle_idx = list(range(len(self.all_points)))
            random.Random(38383).shuffle(self.shuffle_idx)
            self.all_points = [self.all_points[i] for i in self.shuffle_idx]
            self.all_colors = [self.all_colors[i] for i in self.shuffle_idx]
            self.all_normals = [self.all_normals[i] for i in self.shuffle_idx]
            self.all_joints = [self.all_joints[i] for i in self.shuffle_idx]
            self.all_pose = [self.all_pose[i] for i in self.shuffle_idx]
        

        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.all_normals = np.concatenate(self.all_normals)
        self.all_colors = np.concatenate(self.all_colors)
        print ("all points ", self.all_points.shape)
        self.all_joints = np.concatenate(self.all_joints)
        self.all_pose = np.concatenate(self.all_pose)

        total_points = self.all_points.shape[1]

        self.all_joints = self.all_joints[:, 1:]
        self.all_pose = self.all_pose[:, 1:]


        # Bring root to 0
        if not normalize_az:
            self.all_points = self.all_points - self.all_joints.reshape((-1, 23, 3))[:,3:4,:]

        self.all_points_zero_centered = self.all_points
        self.all_normals_zero_centered = self.all_normals
 
        if pose_format == 'joints':
            
            self.all_pose = self.all_pose.reshape((-1, 23, 3)) - self.all_joints.reshape((-1, 23, 3))[:,3:4,:]
            self.all_pose = self.all_pose.reshape((-1, 69))
            self.all_pose_zero_centered = self.all_pose

        self.all_joints = self.all_joints.reshape((-1, 23, 3)) - self.all_joints.reshape((-1, 23, 3))[:,3:4,:]
        self.all_joints = self.all_joints.reshape((-1, 69))
        self.all_joints_zero_centered = self.all_joints

        # Normalization
        
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        elif self.box_per_shape:
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.min(axis=1).reshape(B, 1, input_dim)

            self.all_points_std = self.all_points.max(axis=1).reshape(B, 1, input_dim) - self.all_points.min(axis=1).reshape(B, 1, input_dim)

        else:  # normalize across the dataset
            path_to_mean_std = os.path.join('/CT/siddharth/work/data', self.subdirs, 'new_capture')
            if normalize_az:
                path_to_mean_std = os.path.join(path_to_mean_std, 'norm_az')
            else:
                path_to_mean_std = os.path.join(path_to_mean_std, 'global')

            if len(self.all_normals) > 0:
                path_to_mean_std = os.path.join(path_to_mean_std, 'normals')

            os.makedirs(path_to_mean_std, exist_ok=True)
            samples_range = ':'.join([str(s), str(e), str(i)])

            try:
                 
                self.all_points_mean = np.load(path_to_mean_std + '/samples_' + str(total_points) + '_' + samples_range + '_mean.npy')
                self.all_points_std = np.load(path_to_mean_std + '/samples_' + str(total_points) + '_' + samples_range + '_std.npy')
                print ("Mean and std deviation loaded for ", str(s), str(e), str(i))
                
            except:
                assert self.split == 'train'
                print ("normalize across dataset")
                print ("Taking mean of ", self.all_points.shape[0], " point clouds")

                self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
                if normalize_std_per_axis: # default = False
                    self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
                else:
                    self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

                np.save(path_to_mean_std + '/samples_' + str(total_points) + '_' + samples_range + '_mean.npy', self.all_points_mean)
                np.save(path_to_mean_std + '/samples_' + str(total_points) + '_' + samples_range + '_std.npy', self.all_points_std)


        if normalize_trans:
            raise NotImplementedError
            
        else:
            print ("Normalize for mean and std")
            if pose_format == 'joints':
                
                 self.all_pose = (self.all_pose.reshape(-1, 23, 3) - self.all_points_mean) / self.all_points_std
                 self.all_pose = self.all_pose.reshape(-1, 69)

            self.all_joints = (self.all_joints.reshape(-1, 23, 3) - self.all_points_mean) / self.all_points_std
            
            self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std

        if self.box_per_shape:
            self.all_points = self.all_points - 0.5


        # print normalize the colors
        if len(self.all_colors) > 0:
            if np.amax(self.all_colors) > 1:
                self.all_colors = self.all_colors / 127.5 + (-1.0)
            else:
                self.all_colors = self.all_colors * 2.0 + (-1.0)

        # visualize some sample train points
        outf_syn = os.path.join(output_dir, "sample_train")
        os.makedirs(outf_syn, exist_ok=True)

        unshuffled_all_points = self.all_points.copy()
        unshuffled_all_points_zero_centered = self.all_points_zero_centered.copy()


        test_diversity = False
        if self.split != 'train' and test_diversity:
            self.all_points = self.all_points[0]
            self.all_points = np.repeat(self.all_points[None,:], self.all_pose.shape[0], axis=0)
            self.all_pose = self.all_pose[0]
            self.all_pose = np.repeat(self.all_pose[None,:], self.all_points.shape[0], axis=0)

        # test_perc = 0.1
        # train_length = int(self.all_points.shape[1] * (1. - test_perc))
        self.train_points = self.all_points #[:, :9000]
        self.train_colors = self.all_colors
        self.train_normals = self.all_normals

        self.test_points = self.all_points #[:1000, :, :] #[:, train_length:]
        self.test_colors = self.all_colors
        self.test_normals = self.all_normals


        self.train_pose = self.all_pose.astype(np.float32) #[:, :train_length]
        self.test_pose = self.all_pose.astype(np.float32) #[:, train_length:]
        self.all_joints = self.all_joints.astype(np.float32) 

        print("Total number of data:%d" % len(self.train_points))
        print("Number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape or self.box_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points #[:, :10000]
        self.test_points = self.all_points #[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        # print ("idx ", idx)
        tr_out = self.train_points[idx]
        tr_out_colors = self.train_colors[idx]
        tr_out_normals = self.train_normals[idx]

        tr_pose = self.train_pose[idx]
        tr_pose_joints = self.all_joints[idx]

        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()
        if len(tr_out_colors) > 0:
            tr_out_colors = torch.from_numpy(tr_out_colors[tr_idxs, :]).float()
        if len(tr_out_normals) > 0:
            tr_out_normals = torch.from_numpy(tr_out_normals[tr_idxs, :]).float()
        
        te_out = self.test_points[idx]
        te_out_colors = self.test_colors[idx]
        te_out_normals = self.test_normals[idx]

        te_pose = self.test_pose[idx]

        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float() 
        if len(te_out_colors) > 0:
            te_out_colors = torch.from_numpy(te_out_colors[te_idxs, :]).float()
        if len(te_out_normals) > 0:
            te_out_normals = torch.from_numpy(te_out_normals[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)

        out = {
            'idx': idx,
            'train_points': tr_out,
            'test_points': te_out,
            'train_normals': tr_out_normals,
            'test_normals': te_out_normals,
            'train_pose': tr_pose,
            'test_pose': te_pose,
            'train_joints': tr_pose_joints,
            'train_colors': tr_out_colors,
            'test_colors': te_out_colors,
            'mean': m, 'std': s          
        }

        if self.use_mask:
            tr_mask = self.mask_transform(tr_out)
            out['train_masks'] = tr_mask

        return out


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root_dir="/CT/ashwath/work/DDC_DATA/Subject001",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False, normalize_trans=True, normalize_az=True,
                 box_per_shape=False, random_subsample=False,
                 all_points_mean=None, all_points_std=None,
                 use_mask=False, ntrain=None, pose_format='angles', output_dir=None):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeNet15kPointClouds, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape, box_per_shape=box_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            normalize_trans=normalize_trans,
            normalize_az=normalize_az,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, 
            all_points_std=all_points_std,
            input_dim=3, use_mask=use_mask, 
            ntrain=ntrain, pose_format=pose_format, output_dir=output_dir)



class VizData(Dataset):
    def __init__(self, opt, outf_syn=None, norm_az=False):

        self.tr_joint_poses, self.te_joint_poses, \
        self.tr_out_pose, self.te_out_pose = get_viz_data(opt, outf_syn, norm_az)

        if opt.pose_format == 'joints':
            self.tr_out_pose = self.tr_joint_poses.copy() #clone()
            self.te_out_pose = self.te_joint_poses.copy() #clone()
 

    def __len__(self):
        return len(self.tr_joint_poses)

    def __getitem__(self, idx):
        tr_out_pose = self.tr_out_pose[idx]
        tr_joint_poses = self.tr_joint_poses[idx]

        te_out_pose = self.te_out_pose[idx]

        te_joint_poses = self.te_joint_poses[idx]

        out = {
            'idx': idx,
            'train_joints': tr_joint_poses,
            'test_joints': te_joint_poses,
            'train_pose': tr_out_pose,
            'test_pose': te_out_pose,
        }

        return out



class PointCloudMasks(object):
    '''
    render a view then save mask
    '''
    def __init__(self, radius : float=10, elev: float =45, azim:float=315, ):

        self.radius = radius
        self.elev = elev
        self.azim = azim


    def __call__(self, points):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        camera = [self.radius * np.sin(90-self.elev) * np.cos(self.azim),
                  self.radius * np.cos(90 - self.elev),
                  self.radius * np.sin(90 - self.elev) * np.sin(self.azim),
                  ]

        _, pt_map = pcd.hidden_point_removal(camera, self.radius)

        mask = torch.zeros_like(points)
        mask[pt_map] = 1

        return mask