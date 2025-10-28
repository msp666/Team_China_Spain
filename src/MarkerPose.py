from test_aruco_detection import *

dictionary = "DICT_ARUCO_ORIGINAL"
calibration_file = "./config/camera_calibration.yaml" # calibration file path
# tester = ArUcoDetectionTester(dictionary, 0.095, calibration_file, robust_mode=True, enhance_image=True)
# corners, ids, rejected, display_image = tester.detect_markers(frame)


def test_from_camera(camera_id=0, dictionary='DICT_ARUCO_ORIGINAL', marker_size=0.095, 
                     calibration_file=None, width=1280, height=720, use_realsense=True,
                     robust_mode=True, enhance_image=True):
    """Test ArUco detection from camera"""
    
    # Try RealSense first if requested
    cap = None
    is_realsense = False
    
    if use_realsense and REALSENSE_AVAILABLE:
        print("Attempting to open RealSense camera...")
        try:
            cap = RealSenseCamera(width, height)
            is_realsense = True
            print(f"✓ Using RealSense camera")
        except Exception as e:
            print(f"⚠ RealSense initialization failed: {e}")
            print(f"  Falling back to standard camera {camera_id}...")
            cap = None
    
    tester = ArUcoDetectionTester(dictionary, marker_size, calibration_file, 
                                   robust_mode=robust_mode, enhance_image=enhance_image)
    
    print(f"✓ Robust mode: {'ON' if robust_mode else 'OFF'}")
    print(f"✓ Image enhancement: {'ON' if enhance_image else 'OFF'}")
    
    camera_type = "RealSense" if is_realsense else "Standard"
    window_name = f"ArUco Detection Test - {camera_type} Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("\nControls:")
    print("  ESC - Quit")
    print("  SPACE - Pause/Resume")
    print("  S - Save screenshot")
    print("  C - Print statistics")
    print(f"\nStarting detection with {camera_type} camera...\n")
    
    paused = False
    last_time = time.time()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            tester.fps_list.append(fps)
            
            # Detect markers
            corners, ids, rejected, display_image = tester.detect_markers(frame)

            if ids is not None and tester.camera_matrix is not None:
                for corner, idx in zip(corners, ids):
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corner, 
                        tester.marker_size, 
                        tester.camera_matrix, 
                        tester.dist_coeffs
                    )
                    
                    # rvec: rotation (3x1)
                    # tvec: translation (3x1) - relative to camera
                    
                    tvec = tvec[0][0]
                    distance = np.linalg.norm(tvec)
                    
                    print(f"Marker ID {idx[0]}:")
                    print(f"  pos (x,y,z): ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})m")
                    print(f"  dist: {distance:.3f}m")
                    if idx == "429":
                        break

            
            # Draw info
            display_image = tester.draw_info(display_image, corners, ids, fps)
            
            # Show image
            cv2.imshow(window_name, display_image)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'⏸ Paused' if paused else '▶ Resumed'}")
        elif key == ord('s') or key == ord('S'):  # S
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"aruco_test_{timestamp}.jpg"
            cv2.imwrite(filename, display_image)
            print(f"✓ Saved screenshot: {filename}")
        elif key == ord('c') or key == ord('C'):  # C
            tester.print_statistics()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    tester.print_statistics()

    return tvec, rvec


