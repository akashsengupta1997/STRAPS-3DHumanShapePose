# -*- coding: utf-8 -*-
# Code from https://github.com/mkocabas/VIBE

import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags

import config


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer():
    def __init__(self, resolution=(256, 256)):
        self.resolution = resolution

        self.faces = np.load(config.SMPL_FACES_PATH)
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        # light_pose[:3, 3] = [1, 1, 2]
        # self.scene.add(light, pose=light_pose)

    def render(self, verts, cam, img=None, angle=None, axis=None, mesh_filename=None, color=[0.8, 0.3, 0.3],
               return_mask=False):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle is not None and axis is not None:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        if cam.shape[-1] == 4:
            sx, sy, tx, ty = cam
        elif cam.shape[-1] == 3:
            s, tx, ty = cam
            sx = sy = s

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        rgb, rend_depth = self.renderer.render(self.scene, flags=RenderFlags.RGBA)
        valid_mask = (rend_depth > 0)
        if return_mask:
            return valid_mask
        else:
            if img is None:
                img = np.zeros((self.resolution[0], self.resolution[1], 3))
            valid_mask = valid_mask[:, :, None]
            output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
            image = output_img.astype(np.uint8)

            self.scene.remove_node(mesh_node)
            self.scene.remove_node(cam_node)

            return image
