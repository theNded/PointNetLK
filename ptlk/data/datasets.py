""" datasets """

import numpy
import torch
import torch.utils.data


from . import globset
from . import mesh
from .. import so3
from .. import se3


class ModelNet(globset.Globset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.offread
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class ShapeNet2(globset.Globset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """
    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = mesh.objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class CADset4tracking(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 rigid_transform,
                 source_modifier=None,
                 template_modifier=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(pm)
        igt = self.rigid_transform.igt

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class CADset4tracking_fixed_perturbation(torch.utils.data.Dataset):
    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __init__(self,
                 dataset,
                 perturbation,
                 source_modifier=None,
                 template_modifier=None,
                 fmt_trans=False):
        self.dataset = dataset
        self.perturbation = numpy.array(
            perturbation)  # twist (len(dataset), 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans  # twist or (rotation and translation)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0)  # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0)  # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0)  # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R  # rotation
            g[:, 0:3, 3] = q  # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0)  # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        twist = torch.from_numpy(numpy.array(
            self.perturbation[index])).contiguous().view(1, 6)
        pm, _ = self.dataset[index]
        x = twist.to(pm)
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt

import os
import glob
import numpy as np
from scipy.linalg import expm, norm
import MinkowskiEngine as ME


def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(
        randg.rand(3) - 0.5,
        rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


class IndoorPairDataset(torch.utils.data.Dataset):
    OVERLAP_RATIO = None
    AUGMENT = None

    def __init__(self,
                 dataset_path,
                 phase,
                 num_points=4096,
                 random_rotation=True,
                 rotation_range=360,
                 voxel_size=0.025,
                 manual_seed=False):
        self.files = []

        self.num_points = num_points
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.voxel_size = voxel_size
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

        self.root = root = dataset_path
        self.phase = phase
        print("Loading the subset {} from {}".format(phase, root))

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(root + "/" + fname)
            assert len(fnames_txt
                       ) > 0, "Make sure that the path {} has data {}".format(
                           root, fname)

            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])

    def reset_seed(self, seed=0):
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file0 = os.path.join(self.root, self.files[idx][0])
        file1 = os.path.join(self.root, self.files[idx][1])
        data0 = np.load(file0)
        data1 = np.load(file1)
        xyz0 = data0["pcd"]
        xyz1 = data1["pcd"]

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
            T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
            trans = T1 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)
        else:
            trans = np.identity(4)

        # Voxelization and sampling
        sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size,
                                        return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size,
                                        return_index=True)
        xyz0 = xyz0[np.random.choice(sel0, self.num_points)]
        xyz1 = xyz1[np.random.choice(sel1, self.num_points)]

        pointcloud1 = xyz0
        pointcloud2 = xyz1

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), trans.astype('float32')


class ThreeDMatchPairDataset03(IndoorPairDataset):
    OVERLAP_RATIO = 0.3
    DATA_FILES = {
        'train': './config/train_3dmatch.txt',
        'val': './config/val_3dmatch.txt',
        'test': './config/test_3dmatch.txt'
    }


class ThreeDMatchPairDataset05(ThreeDMatchPairDataset03):
    OVERLAP_RATIO = 0.5


class ThreeDMatchPairDataset07(ThreeDMatchPairDataset03):
    OVERLAP_RATIO = 0.7


#EOF
