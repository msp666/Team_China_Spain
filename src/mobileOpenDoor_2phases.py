import robotic as ry
import numpy as np
import time

import getParameters

gripper = 'l_gripper'
tau = 0.01
ROBOT_FRAME_INDEX = 0
ROBOT_JOINT_INDEX = 0

class ManpulaitonHelper2(ry.KOMO_ManipulationHelper):

    def __init__(self):
        super().__init__()

    def grasp_cylinder(self, time, gripper, obj, palm, margin=0.02):
        pass


def loadConfig():

    # setup configuration
    C = ry.Config()
    frame = C.addFile(ry.raiPath('scenarios/panda_ranger.g'))\
            .setPosition([0., 0., 0])
    panda_joints = C.getJointNames()
    
    global ROBOT_JOINT_INDEX, q0
    ROBOT_JOINT_INDEX = len(panda_joints)
    C.addFile("./scenario/door.g")\
     .setPosition([-2., 0., 0])
    print(frame.name)

    C.view(True)
    
    return C

def pullDoor(C: ry.Config):
    
    sim = ry.Simulation(C, ry.SimulationEngine.physx, verbose=0)
    col_pairs = C.getCollidablePairs()
    print(col_pairs)

    dofID = C.getJointIDs()

    '''setup the  KOMO problem'''
    M = ry.KOMO_ManipulationHelper()
    M.setup_sequence(C, 2, 1e-2, 1e-1, True, False, False)
    M.grasp_cylinder(1., gripper, 'handle_body2', 'l_palm', 0.1)

    # M.komo.addObjective([1], ry.FS.position, ['ranger_base_link'], ry.OT.eq, [1e1, 1e1, 0], [-3., 0., 0.])
    M.freeze_joint([0, 1.], ['handle_joint'])
    M.freeze_joint([0, 1.], ['door_joint'])
    M.freeze_joint([0,2], ['ranger_rot'])

    # pull (or TODO: push) the door
    M.komo.addObjective([2], ry.FS.qItself, ['door_joint'], ry.OT.eq, [1e1], [-0.5])
    # M.freeze_relativePose([1.3], gripper, 'handle_body2')
    # M.komo.addObjective([1,3], ry.FS.positionRel, ['handle_body2', gripper], ry.OT.eq, [1e1], [0.])
    M.komo.addFrameDof('grasp_handle', gripper, ry.JT.free, True, 'handle_body2', None)
    # M.komo.addObjective([1, 3], ry.FS.positionRel, ['grasp_handle', gripper], ry.OT.eq, [1e1], [0.])
    M.komo.addObjective([1, 2], ry.FS.poseDiff, ['handle_body2', 'grasp_handle'], ry.OT.eq, [1e1], [0.])
    # print(M.komo.getConfig().getFrameNames())
    

    M.solve(2)
    M.komo.set_viewer(C.get_viewer())
    M.komo.view(True)
    
    if not M.feasible: 
        print(M.komo.report())
        return None
    path = M.path
 
    R1 = M.sub_rrt(0, col_pairs)
    R1.solve()
    path_rrt = R1.get_resampledPath(50)

    M1 = M.sub_motion(0, True, 1e-2, 1e-1)
    M1.komo.initWithPath(path_rrt)
    M1.freeze_joint([0., 1.], ['handle_joint', 'door_joint'])
    M1.no_collisions([0., 0.9], [gripper, 'handle_body2'], margin=0.03)
    M1.solve()
    if not M1.feasible:
        print(M1.komo.report())
        return None
    
    M2 = M.sub_motion(1)
    M2.freeze_relativePose([0., 1.], gripper, 'handle_body2')
    M2.solve()
    if not M2.feasible:
        print(M2.ret)
        M2.komo.view(True, 'phase2 not feasible')
        return
    
    path1 = M1.path[:, 0:ROBOT_JOINT_INDEX]
    path2 = M2.path[:, 0:ROBOT_JOINT_INDEX]

    joints = C.getJointNames()
    # different from Cpp where we can select joint in the Config which is a member of sim, here we directly select 
    # it outside at original Config. We can also write this in Cpp, because sim is constructed with the reference of Config
    C.selectJoints(joints[0:ROBOT_JOINT_INDEX])
    sim.resetSplineRef()
    sim.setSplineRef(path1, [4.])
    while sim.getTimeToSplineEnd() > 0.:
        time.sleep(tau)
        sim.step([], tau, ry.ControlMode.spline)
        C.view()

    # sim.moveGripper(gripper, 0.0, .5)
    # while not sim.gripperIsDone(gripper):
    #     time.sleep(tau)
    #     sim.step([], tau, ry.ControlMode.spline)
    #     C.view()

    sim.setSplineRef(path2, [3.])
    while sim.getTimeToSplineEnd() > 0.:
        time.sleep(tau)
        sim.step([], tau, ry.ControlMode.spline)
        C.view()


if __name__ == "__main__":
    C = loadConfig()
    pullDoor(C)