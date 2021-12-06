import torch
import numpy as np
from tools.conversion_util import rotmat2aa
from smplx import SMPL
import trimesh  # install by `pip install trimesh`
import vedo  # install by `pip install vedo`
from vedo.io import screenshot
def convert_results_to_axis_angle(npy_path):
    predict_sequence = np.load(npy_path)
    print(predict_sequence)
    trans=predict_sequence[:,6:9]
    print(trans)
    pose_sequence = predict_sequence[:,9:]
    pose_sequence=pose_sequence.reshape((-1,24,9))
    print(pose_sequence.shape)
    axis_angle_sequence=rotmat2aa(pose_sequence)
    axis_angle_sequence=axis_angle_sequence.reshape(-1,72)
    print(axis_angle_sequence.shape)
    return trans,axis_angle_sequence
def sequnce_to_smpl(trans,axis_angle_sequence,smpl_model_path):
    smpl_trans=np.array(trans,dtype=float)
    print(type(smpl_trans))
    smpl_poses=axis_angle_sequence
    smpl = SMPL(model_path=smpl_model_path,gender="MALE")
    _global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float()
    _body_pose=torch.from_numpy(smpl_poses[:, 1:]).float()
    _transl=torch.from_numpy(smpl_trans).float()
    for i in range(axis_angle_sequence.shape[0]):   
        #smpl_scaling=np.array([1]); 
        #print(axis_angle_sequence[i])
        vertices = smpl.forward(
            global_orient=_global_orient,
            body_pose=_body_pose,
            transl=_transl,
            #scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
            ).vertices.detach().numpy()[i]  # first frame
        faces = smpl.faces
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
        vedo.show(mesh,offscreen=True,interactive=True)
        print("result_{}".format(i))
        screenshot(filename="./video/result_{}".format(i))





if __name__ == "__main__":
    npy_path = './outputs/gJB_sBM_cAll_d08_mJB5_ch01_mBR0.npy'
    trans,axis_angle_sequence=convert_results_to_axis_angle(npy_path)
    sequnce_to_smpl(trans,axis_angle_sequence,"./smpl_models/SMPL_MALE.pkl")
