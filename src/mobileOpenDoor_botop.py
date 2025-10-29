import numpy as np
import robotic as ry
from termcolor import cprint
from scipy.spatial.transform import Rotation

from MarkerPose import test_from_camera

gripper = 'l_gripper'
# camera = 'cameraWrist'
camera = 'cameraWrist'
tau = 0.01

# Marker offset assumptions (mirror door, marker near panel center).
# All vectors are expressed in door coordinates while the door is closed.
T_DOOR_MARKER = np.eye(4)
T_DOOR_MARKER[:3, 3] = np.array([0.05, -0.9125, 0.885])

T_DOOR_MARKER1 = np.eye(4)
T_DOOR_MARKER1[:3, 3] = np.array([0.05, -0.9125, 0.885])
T_DOOR_MARKER2 = np.eye(4)
T_DOOR_MARKER2[:3, 3] = np.array([0.05, -0.41, 0.875])
T_DOOR_MARKER3 = np.eye(4)
T_DOOR_MARKER3[:3, 3] = np.array([0.05, -0.905, 0.645])
T_DOOR_LIST = [T_DOOR_MARKER1, T_DOOR_MARKER2, T_DOOR_MARKER3]
DOOR_to_MARKER = {
    0: 429,
    1: 622,
    2: 172,
}

T_WALL_MARKER1 = np.eye(4)
T_WALL_MARKER1[:3, 3] = np.array([0.04, -1.06, 0.88])
T_WALL_MARKER2 = np.eye(4)
T_WALL_MARKER2[:3, 3] = np.array([0.04, -1.06, 0.64])
T_WALL_LIST = [T_WALL_MARKER1, T_WALL_MARKER2]
WALL_to_MARKER = {
    0: 279,
    1: 735,
}

BASE_TO_BASECAMERA = np.eye(4)
BASE_TO_BASECAMERA[:3,:3] = np.array([[0, 0, -1], [1, 0, 0], [0,-1,0]])
BASE_TO_BASECAMERA[:3, 3] = np.array([-.35, .1, 0.4])

ROBOT_JOINT_COUNT = 0

DOOR_IDS = [429, 622, 172]
WALL_IDS = [279, 735]


# MARKER_TO_HANDLE_OFFSET = np.array([0.0, 0.085, 17.25])

# Updated when loadConfig() runs so other modules can read the estimated poses.
door_pose_world = None
handle_pose_world = None

def se3_to_matrix(r_dict, t_dict):
    res = {}

    for k in r_dict.keys():
        _T = np.eye(4)
        _T[:3, 3] = t_dict[k]
        _T[:3, :3] = ry.Quaternion().setExp(r_dict[k]).getMatrix()
        res[k] = _T

    return res

def matrix_to_se3(Ts:np.array):

    res = []
    for T in Ts:
        _r = ry.Quaternion().setMatrix(T[:3, :3]).getLog()
        _t = T[:3, 3]
        res.append(_t + _r)
    return np.ndarray(res)
    




class ManpulaitonHelper2(ry.KOMO_ManipulationHelper):

    def __init__(self):
        super().__init__()

    def handle_grasp(self, time, gripper, obj, joint, palm, margin=0.02):
        """Align the gripper to grasp the door handle while respecting clearance."""
        config = self.komo.getConfig()
        size = config.getFrame(obj).getSize()[:2]

        # position: center along axis, stay within z-range
        xy_alignment = np.array([[1, 0, 0], [0, 1, 0]]) * 1e1
        xz_alignment = np.array([[1, 0, 0], [0, 0, 1]]) * 1e1
        z_axis = np.array([[0, 0, 1]]) * 1e1
        # self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, xy_alignment)
        self.komo.addObjective([time], ry.FS.positionRel, [obj, gripper], ry.OT.eq, np.array([1,0,0]))
        self.komo.addObjective([time], ry.FS.positionRel, [obj, gripper], ry.OT.eq, z_axis, [0.01])
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, z_axis, np.array([0., 0., .5 * size[0] - margin]))
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, -z_axis, np.array([0., 0., -.5 * size[0] + margin]))

        # orientation: keep the gripper orthogonal to the handle plane
        self.komo.addObjective([time - .2, time], ry.FS.scalarProductXZ, [gripper, obj], ry.OT.eq, [1e0])
        # self.komo.addObjective([time - .2, time], ry.FS.scalarProductXZ, [gripper, joint], ry.OT.ineq, scale=[1e0], target=[0.7])

        # no collision with palm
        self.komo.addObjective([time - .3, time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-.001])


def acquire_marker_pose():
    """Fetch marker pose from the camera pipeline and return translation, rotation."""
    t, r, rotations, translations= test_from_camera(calibration_file="config/camera_calibration.yaml")

    for key in rotations.keys():
        rotations[key] = np.asarray(rotations[key], dtype=float).reshape(3,)
    for key in translations.keys():
        translations[key] = np.asarray(translations[key], dtype=float).reshape(3,)

    return np.asarray(r, dtype=float).reshape(3,), np.asarray(t, dtype=float).reshape(3,), rotations, translations


def place_door_scene(C: ry.Config, door_pose: np.ndarray=np.array([-1, 0, 0, 1, 0, 0, 0])) -> ry.Frame:
    """Load the door scenario and set the base frame pose explicitly."""
    door_frame = C.getFrame('door_base')
    if door_frame is None:
        door_frame = C.addFile("./scenario/door_with_walls_mirror.g")
    door_frame.setPose(door_pose)
    return door_frame


def update_handle_pose(config: ry.Config, handle_translation: np.ndarray) -> np.ndarray:
    """Shift the handle marker to match the estimated position in world coordinates."""
    handle_frame = config.getFrame('handle_marker')
    handle_pose = np.array(handle_frame.getPose(), dtype=float)
    handle_pose[:3] = handle_translation
    handle_frame.setPose(handle_pose)
    return handle_pose


def pose_estimation():
    rh, th, r_dict, t_dict = acquire_marker_pose()
    _phi_h = np.concatenate([th, rh])

    pose_dict = se3_to_matrix(r_dict, t_dict)

    # door pose
    T_door_joint_base = []
    for i, T_door_marker in enumerate(T_DOOR_LIST):
        marker_id = DOOR_to_MARKER[i]
        marker_T = pose_dict[marker_id]
        T_base_marker = BASE_TO_BASECAMERA @ marker_T @ np.linalg.inv(T_door_marker)
        T_door_joint_base.append(T_base_marker)
    T_door_joint_base = np.array(T_door_joint_base)
    phi_door_joint = matrix_to_se3(T_door_joint_base)
    phi_door_joint = phi_door_joint.mean(axis=0)

    door_pose_vec = np.zeros(7)
    quat = ry.Quaternion().setMatrix(phi_door_joint[3:]).asArr()
    door_pose_vec[:3] = phi_door_joint[:3]
    door_pose_vec[3:] = quat

    # wall pose
    T_door_base = []
    for i, T_wall_marker in enumerate(T_WALL_LIST):
        marker_id = WALL_to_MARKER[i]
        marker_T = pose_dict[marker_id]
        T_base_marker = BASE_TO_BASECAMERA @ marker_T @ np.linalg.inv(T_wall_marker)
        T_door_base.append(T_base_marker)
    T_door_base = np.array(T_door_base)
    phi_door_base = matrix_to_se3(T_door_base)
    phi_door_base = phi_door_base.mean(axis=0)

    base_pose_vec = np.zeros(7)
    quat = ry.Quaternion().setMatrix(phi_door_base[3:]).asArr()
    base_pose_vec[:3] = phi_door_base[:3]
    base_pose_vec[3:] = quat

    q_door = phi_door_base[-1] - phi_door_joint[-1]

    return door_pose_vec, base_pose_vec, q_door



def loadConfig(C: ry.Config):
    
    C.addFrame('cameraBaseMarker', camera).setShape(ry.ST.marker, [.3])

    base_to_camera, _ = C.eval(ry.FS.poseRel, [camera, 'ranger_rot'])
    BASE_TO_BASECAMERA[:3, :3] = ry.Quaternion().set(base_to_camera[3:]).getMatrix()
    BASE_TO_BASECAMERA[:3, 3] = base_to_camera[:3]

    # door default pose
    door_base_frame = place_door_scene(C)
    door_marker, _ = C.eval(ry.FS.poseRel, frames=['aruco1', 'door_joint']) # TODO: calib DOOR TO MARKER
    T_DOOR_MARKER[:3, :3] = ry.Quaternion().set(door_marker[3:]).getMatrix()
    C.view(True)

    # get marker
    # marker_translation, marker_rotation = acquire_marker_pose()
    # cprint(f'marker tranlation: {marker_translation},\n marker rotaion: {marker_rotation}', 'red')
    # T_camera_marker  = np.eye(4)
    # T_camera_marker[:3, 3] = marker_translation
    # T_camera_marker[:3, :3] = ry.Quaternion().setExp(marker_rotation).getMatrix()

    # TODO: change this part to the marker pose given by camera
    _, _, rotations, translations = acquire_marker_pose()
    marker_id = DOOR_to_MARKER[0]
    if marker_id not in rotations or marker_id not in translations:
        raise RuntimeError(f"Marker {marker_id} not detected by RealSense.")
    pose_camera_marker = np.zeros(7, dtype=float)
    pose_camera_marker[:3] = translations[marker_id]
    pose_camera_marker[3:] = ry.Quaternion().setExp(rotations[marker_id]).asArr()
    T_camera_marker = np.eye(4)
    T_camera_marker[:3, :3] = ry.Quaternion().set(pose_camera_marker[3:]).getMatrix()
    T_camera_marker[:3, 3] = pose_camera_marker[:3] 
    # T_camera_marker[0,3] += 1.

    # 
    # T_base_door = BASE_TO_BASECAMERA @ T_camera_marker @ np.linalg.inv(T_DOOR_MARKER)

    # door_pose_vec = np.zeros(7)
    # R = T_base_door[:3, :3]
    # quat = ry.Quaternion().setMatrix(R).asArr()
    # door_pose_vec[:3] = T_base_door[:3, 3]
    # door_pose_vec[3:] = quat
    # door_base_frame = place_door_scene(C, door_pose=door_pose_vec)

    door_joint_base, door_base, q_door = pose_estimation()
    place_door_scene(C, door_pose=door_base)
    C.getFrame('door_joint').setJointState(q_door)

    C.view(True)

    return C


def execute_paths(bot: ry.BotOp, config: ry.Config, *paths):
    """Stream a sequence of joint-space paths to the robot."""
    for path in paths:
        for waypoint in path:
            bot.moveTo(waypoint)
            bot.wait(config)


def pullDoor(C_plan: ry.Config, C: ry.Config, bot: ry.BotOp, home_q):
    cprint('Pull the door', 'red')

    col_pairs = C_plan.getCollidablePairs()
    print(col_pairs)

    # setup the KOMO problem
    helper = ManpulaitonHelper2()
    helper.setup_sequence(C_plan, 2, 1e-2, 1e-1, True, False, False)
    helper.handle_grasp(1., gripper, 'handle_body2', 'door_joint', 'l_palm', 0.02)

    helper.freeze_joint([0, 1.], ['handle_joint'])
    helper.freeze_joint([0, 1.], ['door_joint'])
    helper.freeze_joint([0, 2], ['ranger_rot'])


    # pull the door
    helper.komo.addObjective([2], ry.FS.qItself, ['door_joint'], ry.OT.eq, [1e1], [0.5])
    helper.komo.addFrameDof('grasp_handle', gripper, ry.JT.free, True, 'handle_body2', None)
    helper.komo.addObjective([1, 2], ry.FS.poseDiff, ['handle_body2', 'grasp_handle'], ry.OT.eq, [1e1], [0.])

    helper.solve(2)
    helper.komo.set_viewer(C_plan.get_viewer())
    helper.komo.view(True)

    if not helper.feasible:
        print(helper.komo.report())
        return None

    rrt = helper.sub_rrt(0, col_pairs)
    ret = rrt.solve()
    if not ret.feasible:
        print("RRT did not find a solution!")
        return
    
    path_rrt = rrt.get_resampledPath(50)

    phase1 = helper.sub_motion(0, True, 1e-2, 1e-1)
    phase1.komo.initWithPath(path_rrt)
    phase1.freeze_joint([0., 1.], ['handle_joint', 'door_joint'])
    phase1.no_collisions([0., 0.9], [gripper, 'handle_body2'], margin=0.03)
    phase1.solve()
    if not phase1.feasible:
        print(phase1.komo.report())
        return None

    phase2 = helper.sub_motion(1)
    # phase2.freeze_relativePose([0., 1.], gripper, 'handle_body2')
    phase2.solve()
    if not phase2.feasible:
        print(phase2.ret)
        phase2.komo.view(True, 'phase2 not feasible')
        return

    execute_paths(bot, C, phase1.path)
    bot.gripperMove(ry._left, width=.0, speed=.1)
    execute_paths(bot, C, phase2.path[:, :-2])
    
    bot.gripperMove(ry._left)
    while not bot.gripperDone(ry._left):
        bot.sync(C)
    bot.moveTo(home_q, 0.1)
    bot.wait(C)


def go_through(C: ry.Config, bot: ry.BotOp):
    cprint('Go through', 'red')
    ry.KOMO_ManipulationHelper()


if __name__ == "__main__":
    C = ry.Config()
    C.addFile(ry.raiPath('scenarios/panda_ranger.g')).setPosition([0., 0., 0])

    bot = ry.BotOp(C, useRealRobot=False)
    bot.sync(C)
    home_q = bot.get_q()
    
    C_plan = ry.Config()
    C_plan.addConfigurationCopy(C)

    C_plan = loadConfig(C_plan)

    # pcl_frame = C.addFrame("pcl_frame", camera)
    # while True:
    #     rgb, _, pcl = bot.getImageDepthPcl(camera)
    #     # bot.sync(C)
    #     pcl_frame.setPointCloud(pcl, rgb)
    #     C.view()

    pullDoor(C_plan, C, bot, home_q)
