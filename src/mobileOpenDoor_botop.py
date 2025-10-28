import numpy as np
import robotic as ry
from termcolor import cprint

from MarkerPose import test_from_camera

gripper = 'l_gripper'
tau = 0.01

# Marker offset assumptions (mirror door, marker near panel center).
# All vectors are expressed in door coordinates while the door is closed.
MARKER_TO_DOOR_BASE_OFFSET = np.array([0.0, -0.9125, 1.045])
BASE_TO_BASECAMERA = np.array([[ 0.9989,  0.0464, -0.0112,  0.1149],
                                [ 0.003 ,  0.1737 , 0.9848, -0.0891],
                                [ 0.0477, -0.9837,  0.1734,  0.7865],
                                [ 0.  ,    0.   ,   0.    ,  1.    ]])

# MARKER_TO_HANDLE_OFFSET = np.array([0.0, 0.085, 17.25])

# Updated when loadConfig() runs so other modules can read the estimated poses.
door_pose_world = None
handle_pose_world = None


class ManpulaitonHelper2(ry.KOMO_ManipulationHelper):

    def __init__(self):
        super().__init__()

    def handle_grasp(self, time, gripper, obj, joint, palm, margin=0.02):
        """Align the gripper to grasp the door handle while respecting clearance."""
        config = self.komo.getConfig()
        size = config.getFrame(obj).getSize()[:2]

        # position: center along axis, stay within z-range
        xy_alignment = np.array([[1, 0, 0], [0, 1, 0]]) * 1e1
        z_axis = np.array([[0, 0, 1]]) * 1e1
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, xy_alignment)
        self.komo.addObjective(
            [time],
            ry.FS.positionRel,
            [gripper, obj],
            ry.OT.ineq,
            z_axis,
            np.array([0., 0., .5 * size[0] - margin]),
        )
        self.komo.addObjective(
            [time],
            ry.FS.positionRel,
            [gripper, obj],
            ry.OT.ineq,
            -z_axis,
            np.array([0., 0., -.5 * size[0] + margin]),
        )

        # orientation: keep the gripper orthogonal to the handle plane
        self.komo.addObjective([time - .2, time], ry.FS.scalarProductXZ, [gripper, obj], ry.OT.eq, [1e0])
        self.komo.addObjective([time - .2, time], ry.FS.scalarProductXZ, [gripper, joint], ry.OT.ineq, scale=[1e0], target=[0.7])

        # no collision with palm
        self.komo.addObjective([time - .3, time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-.001])


def acquire_marker_pose():
    """Fetch marker pose from the camera pipeline and return translation, rotation."""
    t, r = test_from_camera()
    return np.asarray(t, dtype=float).reshape(3,), np.asarray(r, dtype=float).reshape(3,)


def place_door_scene(C: ry.Config, door_pose: np.ndarray) -> ry.Frame:
    """Load the door scenario and set the base frame pose explicitly."""
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


def loadConfig():

    # setup configuration
    C = ry.Config()

    # mobile base position
    C.addFile(ry.raiPath('scenarios/panda_ranger.g')).setPosition([0., 0., 0])

    # get marker
    marker_translation, _ = acquire_marker_pose()
    door_translation = marker_translation + MARKER_TO_DOOR_BASE_OFFSET
    door_pose_vec = np.array([door_translation[0], door_translation[1], door_translation[2], 1., 0., 0., 0.], dtype=float)
    T_door_to_marker = np.eye(4)
    T_door_to_marker[:3,3] = door_translation.reshape(3,1)
    T = T_door_to_marker * BASE_TO_BASECAMERA

    # door_translation = [-1, 0., 0]

    # door
    door_base_frame = place_door_scene(C, door_pose_vec)

    global door_pose_world, handle_pose_world
    door_pose_world = np.array(door_base_frame.getPose(), dtype=float)

    print("Door pose (world):", door_pose_world)
    print("Handle pose (world):", handle_pose_world)

    C.view(True)

    return C


def execute_paths(bot: ry.BotOp, config: ry.Config, *paths):
    """Stream a sequence of joint-space paths to the robot."""
    for path in paths:
        for waypoint in path:
            bot.moveTo(waypoint)
            bot.wait(config)


def pullDoor(C: ry.Config, bot: ry.BotOp):
    cprint('Pull the door', 'red')

    col_pairs = C.getCollidablePairs()
    print(col_pairs)

    # setup the KOMO problem
    helper = ManpulaitonHelper2()
    helper.setup_sequence(C, 2, 1e-2, 1e-1, True, False, False)
    helper.handle_grasp(1., gripper, 'handle_body2', 'door_joint', 'l_palm', 0.02)

    helper.freeze_joint([0, 1.], ['handle_joint'])
    helper.freeze_joint([0, 1.], ['door_joint'])
    helper.freeze_joint([0, 2], ['ranger_rot'])

    ry.Quaternion().set

    # pull (or TODO: push) the door
    helper.komo.addObjective([2], ry.FS.qItself, ['door_joint'], ry.OT.eq, [1e1], [0.5])
    helper.komo.addFrameDof('grasp_handle', gripper, ry.JT.free, True, 'handle_body2', None)
    helper.komo.addObjective([1, 2], ry.FS.poseDiff, ['handle_body2', 'grasp_handle'], ry.OT.eq, [1e1], [0.])

    helper.solve(2)
    helper.komo.set_viewer(C.get_viewer())
    helper.komo.view(True)

    if not helper.feasible:
        print(helper.komo.report())
        return None

    rrt = helper.sub_rrt(0, col_pairs)
    rrt.solve()
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
    phase2.freeze_relativePose([0., 1.], gripper, 'handle_body2')
    phase2.solve()
    if not phase2.feasible:
        print(phase2.ret)
        phase2.komo.view(True, 'phase2 not feasible')
        return

    execute_paths(bot, C, phase1.path)
    bot.gripperMove(ry._left, width=.0, speed=.1)
    execute_paths(bot, C, phase2.path)


def go_through(C: ry.Config, bot: ry.BotOp):
    cprint('Go through', 'red')
    ry.KOMO_ManipulationHelper()


if __name__ == "__main__":
    C = loadConfig()
    bot = ry.BotOp(C, useRealRobot=False)

    pullDoor(C, bot)
