import os
import random
import sys

from shutil import copyfile
import datetime
from natsort import natsorted
from tqdm import tqdm
import glob as glob

import open3d as o3d
import torch

import logging
logger = logging.getLogger()

import numpy as np

def set_global_gpu_env(opt):

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    torch.cuda.set_device(opt.gpu)


def copy_source(file, output_dir):
    if not os.path.exists(os.path.join(output_dir, os.path.basename(file))): 
        copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def save_obj(pc, outd, epoch, mode, all_poses, tc, tn):

    os.makedirs(outd, exist_ok=True)

    if tc and tn:
        raise NotImplementedError

    #if pose_format == 'joints':
    #print ("all_poses inside save_obj ", all_poses.shape)
    if all_poses is not None and len(all_poses.shape) < 3:
        #all_poses = all_poses.reshape((-1, 27, 3))
        all_poses = all_poses.reshape((pc.shape[0], -1, 3))

    for i in range(pc.shape[0]):
        fname = mode + "_epoch_" + str(epoch) + "_" + str(i) + ".ply" 

        #print ("pc shape ", pc[i].shape)
        #print ("pc value ", pc[i, 0])
        #print ("pc ", pc[i, 0])
        #print ("pc ", pc[i, 9999])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[i, :, :3])

        if tc: 
            pcd.colors = o3d.utility.Vector3dVector(np.clip(pc[i, :, 3:], 0, 1))
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pc[i, :, :3])+np.array([0, 1, 0]))

        #if pc.shape[2] > 3:
        if not tc and tn:
            pcd.normals = o3d.utility.Vector3dVector(pc[i, :, 3:])

        # Write the point cloud to a PLY file
        o3d.io.write_point_cloud(os.path.join(outd, fname), pcd, write_ascii=True)

        #vertices = ""
        #for v in pc[i]:

        #    
        #    # print (v)
        #    if pc.shape[2] == 3:
        #        vertices += (" ".join(["v"] + list(map(str, [v[0], v[1], v[2], 0, 255, 0]))) + "\n")
        #    else:
        #        vertices += (" ".join(["v"] + list(map(str, v))) + "\n")
        #    # print (vertices)

        #print (os.path.join(outd, fname))
        #with open(os.path.join(outd, fname), "w") as f:
        #    
        #        f.write(vertices)


        # check to plot pose
        if all_poses is not None:
            fname = mode + "_epoch_" + str(epoch) + "_" + str(i) + "_pose.obj"
            vertices = ""
            for v in all_poses[i]:
                vertices += " ".join(["v", str(v[0]), str(v[1]), str(v[2]), "255 0 0"+"\n"])

            print (os.path.join(outd, fname))
            with open(os.path.join(outd, fname), "w") as f:
                
                    f.write(vertices)


def get_poses(pose_format, pose_lines, pc_files, smpl_files, category, idxs, nc, norm_az, tc, tn):

    def lines_to_poses(pose_lines):
        out_pa = []
        for pl in pose_lines:
            da = pl.split(" ")
            da_new = list(filter(None, da))
            da_new_np = np.array(da_new, dtype=np.float32)
            out_pa.append(da_new_np[np.newaxis, ...])

        out_pa = np.concatenate(out_pa)[:, 1:]#.float()
        return out_pa


    def parse_pose_files(pose_files):

        all_poses = [] 
        for pf in pose_files:
            
            with open(pf, "r") as f:
                lines = f.readlines()

            vertices = []
            for line in lines:
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)
            
            vertices = np.array(vertices)[:, :3]

            all_poses.append(vertices[np.newaxis, ...])
        all_poses = np.concatenate(all_poses)

        return all_poses


    def lines_to_pcs(ply_files):
        all_points = []
        for ply_fname in tqdm(ply_files):
            ## obj_fname = os.path.join(sub_path, x)
            # obj_fname = os.path.join(root_dir, subd, mid + ".npy")
            # print ("ply ", ply_fname)
            try:
                # point_cloud = np.load(obj_fname)  # (15k, 3)
                # ply_fpath = os.path.join(root_dir, self.subdirs, ply_fname)
                point_cloud_obj = o3d.io.read_point_cloud(ply_fname)
                point_cloud = np.asarray(point_cloud_obj.points) # (10k, 3)

                if category in ['FranziPCD_full', 'VladPCD_full', 'SiddharthPCD_full', 'ParulPCD_full']:
                    pass
                else:
                    rnd_idxs = np.random.choice(point_cloud.shape[0], 10000)
                    point_cloud = point_cloud[rnd_idxs, :]

                if tn:
                    point_normals = np.asarray(point_cloud_obj.normals)
                    point_cloud = np.concatenate([point_cloud, point_normals], axis=1)

                if tc:
                    point_colors = np.asarray(point_cloud_obj.colors)
                    point_cloud = np.concatenate([point_cloud, point_colors], axis=1)
                
                #if nc == 6:
                #    #point_colors = np.asarray(point_cloud_obj.colors)
                #    #if len(point_colors) > 0:
                #    #    point_cloud = np.concatenate([point_cloud, point_colors], axis=1)
                #    #else:
                #    point_normals = np.asarray(point_cloud_obj.normals)
                #    point_cloud = np.concatenate([point_cloud, point_normals], axis=1)
            except:
                continue

            # assert point_cloud.shape[0] == 10000
            all_points.append(point_cloud[np.newaxis, ...])
        return np.concatenate(all_points)

    
    if category in ['FranziPCD_full', 'VladPCD_full', 'SiddharthPCD_full', 'ParulPCD_full']:
        out_pa = lines_to_poses(pose_lines)[idxs]
    else:
        out_pa = parse_pose_files(pose_lines)[idxs]

    # read only relevant files
    pcs = None
    if pc_files is not None:
        filtered_files = [pc_files[ii] for ii in idxs]
        pcs = lines_to_pcs(filtered_files)

        # pcs = lines_to_pcs(files)[idxs]
        print ("all point in viz ", pcs.shape)

    smpls = None
    if smpl_files is not None:
        filtered_files = [smpl_files[ii] for ii in idxs]
        smpls = lines_to_pcs(filtered_files)

        print ("all smpl points in viz ", smpls.shape)
    

    if pose_format == 'joints':
        #if category == "VladPCD_full":
        #    if pcs is not None:
        #        if not norm_az:
        #            #pcs = np.concatenate([pcs[:,:,:3] - out_pa.reshape((-1, 27, 3))[:,14:15,:], pcs[:,:,3:]], axis=2)
        #            pcs = np.concatenate([pcs[:,:,:3] - out_pa.reshape((-1, 23, 3))[:,3:4,:], pcs[:,:,3:]], axis=2)
        #        else:
        #            pass
        #            pcs = np.concatenate([pcs[:,:,:3], pcs[:,:,3:]], axis=2)
        #    out_pa = out_pa.reshape((-1, 27, 3)) - out_pa.reshape((-1, 27, 3))[:,14:15,:]
        #    # out_pa = out_pa.reshape((-1, 81))
        #elif category == "FranziPCD_full":
        if pcs is not None:
            if not norm_az:
                if category in ['FranziPCD_full', 'VladPCD_full', 'SiddharthPCD_full', 'ParulPCD_full']:
                    pcs = np.concatenate([pcs[:,:,:3] - out_pa.reshape((-1, 23, 3))[:,3:4,:], pcs[:,:,3:]], axis=2)
                    if smpls is not None:
                        smpls = np.concatenate([smpls[:,:,:3] - out_pa.reshape((-1, 23, 3))[:,3:4,:], smpls[:,:,3:]], axis=2)
                else:
                    pcs = np.concatenate([pcs[:,:,:3] - out_pa.reshape((-1, 55, 3))[:,0:1,:], pcs[:,:,3:]], axis=2)
                    if smpls is not None:
                        smpls = np.concatenate([smpls[:,:,:3] - out_pa.reshape((-1, 55, 3))[:,0:1,:], smpls[:,:,3:]], axis=2)
            else:
                pass
                pcs = np.concatenate([pcs[:,:,:3], pcs[:,:,3:]], axis=2)
        if category in ['FranziPCD_full', 'VladPCD_full', 'SiddharthPCD_full', 'ParulPCD_full']:
            out_pa = out_pa.reshape((-1, 23, 3)) - out_pa.reshape((-1, 23, 3))[:,3:4,:]
        else:
            out_pa = out_pa.reshape((-1, 55, 3)) - out_pa.reshape((-1, 55, 3))[:,0:1,:]
        # out_pa = out_pa.reshape((-1, 69))

    return out_pa, pcs, smpls


def get_viz_data(opt, outf_syn, norm_az=False, use_smpl=False):

    category = opt.category
    dataroot = opt.dataroot
    nc = opt.in_ch
    tn = opt.train_normals
    tc = opt.train_colors

    smpl_files = None
    if category in ['FranziPCD_full', 'VladPCD_full', 'SiddharthPCD_full', 'ParulPCD_full']:
        files = natsorted(glob.glob(os.path.join(dataroot, category, 'depthmap_*.ply')))
        print ("num files in viz ", len(files))
        
        if use_smpl:
            smpl_path = "/CT/siddharth/static00/data/smpl_fitted"
            smpl_subject = category.split("PCD")[0] + "_normalized"
            smpl_files = natsorted(glob.glob(os.path.join(smpl_path, smpl_subject, 'depthmap_*.ply')))
    else:
        files_train = natsorted(glob.glob(os.path.join(dataroot, category, 'train', '*.ply')))
        pose_files_train = natsorted(glob.glob(os.path.join(dataroot, category, 'train', '*.obj')))

        files_test = natsorted(glob.glob(os.path.join(dataroot, category, 'test', '*.ply')))
        pose_files_test = natsorted(glob.glob(os.path.join(dataroot, category, 'test', '*.obj')))

    if category == "FranziPCD_full":
        # IDs divided by 10 
        train_idxs = [0, 200, 400, 600,  800, 1100, 3400, 7800] # , 8200, 10800, 13620, 15460] # 1800, 4400, 7550
        valid_idxs = [17100, 17200, 17500, 17660, 17740, 18140, 18320, 18490] #, 18690, 18760, 19130]
        #train_idxs = [0, 20, 40, 60,  80, 110, 180, 340, 440, 755, 780, 820, 1080, 1362, 1546]
        #valid_idxs = [1710, 1720, 1750, 1766, 1774, 1814, 1832, 1849, 1869, 1876, 1913]
    elif category == "VladPCD_full":
        train_idxs = [0, 80, 700, 1300, 1800, 3300, 4400, 15400] # 220, 350, 500, 4000, 8800, 11500
        valid_idxs = [17000, 17450, 18050, 18200, 18250, 18280, 18790, 18850] #17200, 18500
    elif category == "SiddharthPCD_full":
        train_idxs = [0, 80, 700, 1300, 1800, 3300, 4400, 15400] # 220, 350, 500, 4000, 8800, 11500
        valid_idxs = [17000, 17450, 18050, 18200, 18250, 18280, 18790, 18850] #17200, 18500
    elif category == "ParulPCD_full":
        train_idxs = [0, 80, 700, 1300, 1800, 3300, 4400, 15400] # 220, 350, 500, 4000, 8800, 11500
        valid_idxs = [17000, 17450, 18050, 18200, 18250, 18280, 18790, 18850] #17200, 18500
    elif category == "rp_felice_posed_004":
        train_idxs = [0, 80, 160, 240, 320, 400, 480, 560] # 220, 350, 500, 4000, 8800, 11500
        valid_idxs = [0, 50, 100, 150, 200, 250, 300, 330] #17200, 18500
    elif category == "rp_christine_posed_027":
        train_idxs = [0, 80, 160, 240, 320, 400, 480, 560] # 220, 350, 500, 4000, 8800, 11500
        valid_idxs = [0, 50, 100, 150, 200, 250, 300, 330] #17200, 18500
    elif category == "rp_carla_posed_004":
        train_idxs = [0, 80, 160, 240, 320, 400, 480, 560] # 220, 350, 500, 4000, 8800, 11500
        valid_idxs = [0, 50, 100, 150, 200, 250, 300, 330] #17200, 18500
    else:
        raise NotImplementedError

    # load poses and point clouds

    if category in ['FranziPCD_full', 'VladPCD_full', 'SiddharthPCD_full', 'ParulPCD_full']:
        # read pose joints file
        pose_dir = "/CT/siddharth/work/data/" + category + "/new_capture"
        print ("norm_az ", norm_az)
        if norm_az:
            with open(pose_dir + "/data_export_normalized.mddd", "r") as f:
                pose_lines = f.readlines()[1:]
        else:
            with open(pose_dir + "/data_export.mddd", "r") as f:
                pose_lines = f.readlines()[1:]

        train_joint_poses, train_pcs, train_smpls = get_poses('joints', pose_lines, files, smpl_files, category, train_idxs, nc, norm_az, tc, tn)
        valid_joint_poses, valid_pcs, valid_smpls = get_poses('joints', pose_lines, files, smpl_files, category, valid_idxs, nc, norm_az, tc, tn)

        # read pose angles file
        if norm_az:
            with open("/CT/siddharth/work/data/"+category+"/new_capture/poseAnglesRotationNormalized.motion", "r") as f:
                pose_lines = f.readlines()[1:]
        else:
            with open("/CT/siddharth/work/data/"+category+"/new_capture/poseAngles.motion", "r") as f:
                pose_lines = f.readlines()[1:]
    
        train_angle_poses, _, _ = get_poses('angles', pose_lines, None, None, category, train_idxs, nc, norm_az, tc, tn)
        valid_angle_poses, _, _ = get_poses('angles', pose_lines, None, None, category, valid_idxs, nc, norm_az, tc, tn)

    else:

        train_joint_poses, train_pcs, train_smpls = get_poses('joints', pose_files_train, files_train, smpl_files, category, train_idxs, nc, norm_az, tc, tn)
        valid_joint_poses, valid_pcs, valid_smpls = get_poses('joints', pose_files_test, files_test, smpl_files, category, valid_idxs, nc, norm_az, tc, tn)

        train_angle_poses = None
        valid_angle_poses = None

    save_obj(train_pcs, outf_syn, epoch="GT", mode="train", all_poses=train_joint_poses, tc=tc, tn=tn) 
    save_obj(valid_pcs, outf_syn, epoch="GT", mode="test", all_poses=valid_joint_poses, tc=tc, tn=tn)

    if use_smpl:
        save_obj(train_smpls, outf_syn, epoch="GT_smpl", mode="train", all_poses=train_joint_poses, tc=tc, tn=tn) 
        save_obj(valid_smpls, outf_syn, epoch="GT_smpl", mode="test", all_poses=valid_joint_poses, tc=tc, tn=tn)

    #if pose_format == 'angles': # don't do anything
    #    pass
    #elif pose_format == 'joints': # normalize
    #    pass
    #else:
    #    raise NotImplementedError

    if use_smpl:
        return train_joint_poses, valid_joint_poses, train_angle_poses, valid_angle_poses, train_smpls, valid_smpls, train_pcs, valid_pcs
    else:
        return train_joint_poses, valid_joint_poses, train_angle_poses, valid_angle_poses

def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(log_format)
    logger.addHandler(err_handler)
    logger.setLevel(logging.INFO)

    return logger


def get_output_dir(prefix, exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(prefix, 'output/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir



def set_seed(opt):

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.gpu is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def setup_output_subdirs(output_dir, *subfolders):

    output_subdirs = output_dir
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    subfolder_list = []
    for sf in subfolders:
        curr_subf = os.path.join(output_subdirs, sf)
        try:
            os.makedirs(curr_subf)
        except OSError:
            pass
        subfolder_list.append(curr_subf)

    return subfolder_list
