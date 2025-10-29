door_base: {multibody: true, shape: marker, size:[.5]}
door_joint(door_base): {Q:"t(-0.05 0 0) d(0 1 0 0)", joint: hingeZ, q: 0, shape: marker, size:[.3], limits:[-1.5, 1.5], motorKp=0., motorKd:.5, mass: 5.8, inertia: [0.927285, 0.00114363, -0.00285855, 1.26317, 0.000657603, 0.338493]}
door_body(door_joint): {Q: "t(0 -0.49 1.125) d(0 -1 0 0)", shape:box, size: [.1, .98, 2.25], color: [.8 .5 .2], contact:-1, mass:.5}

aruco1(door_base): {Q: [0.05, -0.9125, 0.905, 0.5, 0.5, 0.5, 0.5 ], shape: marker, size: [0.1]}

left_wall(door_base): {
    Q: "t(0 0.65 1)",
    shape: box,
    size: [.1, 1.2, 2.4],
    color: [.7 .7 .7],
    contact: 1
}

right_wall(door_base): {
    Q: "t(0 -1.65 1)",
    shape: box,
    size: [.1, 1.2, 2.4],
    color: [.7 .7 .7],
    contact: 1
}

handle_joint_origin(door_joint):{ Q: [0., -0.9125, 1.045, 1, 0, 0, 0]}
handle_joint(handle_joint_origin): {
   joint: hingeX,
   limits:[0, 0.8],
   motorKp: 0., motorKd: 1.
}
handle_body1(handle_joint): {
    Q: "t(0.085 0 0 ) d(90 0 1 0)", 
    shape: capsule, 
    size:[.07 .01], 
    color: [1 1 1],
    contact: -1,
    mass: .01,
    }

handle_body2(handle_joint){
    Q: "t(.115 .05 0) d(90 -1 0 0)",
    shape: capsule,
    size:[.1 .01],
    color:[1 1 1],
    contact:-1,
    mass:.01
}

handle_marker(handle_body2){
    shape: marker
    size: [0.1]
}


