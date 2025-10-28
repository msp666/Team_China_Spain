#!/usr/bin/env python3
"""
ArUco Marker Detection Test Script
Quick testing tool for ArUco marker recognition
Supports camera, image files, and video files as input
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import yaml

# Try to import RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None


class RealSenseCamera:
    """RealSense camera wrapper"""
    
    def __init__(self, width=1280, height=720, fps=30):
        """Initialize RealSense camera"""
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not available")
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Start pipeline
        try:
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            print(f"✓ RealSense camera initialized: {width}x{height}@{fps}fps")
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense camera: {e}")
    
    def read(self):
        """Read a frame from RealSense camera"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame:
                return False, None
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image
        except Exception as e:
            print(f"Error reading RealSense frame: {e}")
            return False, None
    
    def release(self):
        """Release RealSense camera"""
        try:
            self.pipeline.stop()
        except:
            pass
    
    def isOpened(self):
        """Check if camera is opened"""
        return True


class ArUcoDetectionTester:
    """Simple ArUco marker detection tester"""
    
    def __init__(self, dictionary='DICT_7X7_1000', marker_size=0.095, calibration_file=None, 
                 robust_mode=True, enhance_image=True):
        """
        Initialize the ArUco detector
        
        Args:
            dictionary: ArUco dictionary name
            marker_size: Size of the marker in meters (for pose estimation)
            calibration_file: Path to camera calibration file (optional)
            robust_mode: Use robust parameters for detecting multiple markers
            enhance_image: Apply image enhancement for better detection
        """
        # Get ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        
        # Create detector with optimized parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        
        if robust_mode:
            # 优化参数以支持同时检测多个markers
            # 自适应阈值窗口 - 更大的范围以适应不同尺寸的markers
            self.detector_params.adaptiveThreshWinSizeMin = 3
            self.detector_params.adaptiveThreshWinSizeMax = 23
            self.detector_params.adaptiveThreshWinSizeStep = 10
            self.detector_params.adaptiveThreshConstant = 7
            
            # 允许更小的markers - 对远处的markers重要
            self.detector_params.minMarkerPerimeterRate = 0.01  # 降低最小周长率
            self.detector_params.maxMarkerPerimeterRate = 4.0
            
            # 多边形近似 - 稍微宽松以处理变形
            self.detector_params.polygonalApproxAccuracyRate = 0.05
            
            # 角点精细化 - 提高精度
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector_params.cornerRefinementWinSize = 5
            self.detector_params.cornerRefinementMaxIterations = 30
            self.detector_params.cornerRefinementMinAccuracy = 0.1
            
            # 减少边界忽略 - 允许靠边的markers
            self.detector_params.minDistanceToBorder = 1
            
            # 允许markers更近 - 对密集排列的markers重要
            self.detector_params.minMarkerDistanceRate = 0.01
            
            # 透视移除参数 - 提高解码准确性
            self.detector_params.perspectiveRemovePixelPerCell = 4
            self.detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
            
            # 位错误容忍 - 允许一定的打印/扫描误差
            self.detector_params.maxErroneousBitsInBorderRate = 0.35
            self.detector_params.minOtsuStdDev = 5.0  # 降低Otsu阈值的标准差要求
            self.detector_params.errorCorrectionRate = 0.6  # 错误纠正率
        else:
            # 标准参数
            self.detector_params.adaptiveThreshWinSizeMin = 3
            self.detector_params.adaptiveThreshWinSizeMax = 23
            self.detector_params.adaptiveThreshConstant = 7
            self.detector_params.minMarkerPerimeterRate = 0.03
            self.detector_params.maxMarkerPerimeterRate = 4.0
            self.detector_params.polygonalApproxAccuracyRate = 0.03
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector_params.cornerRefinementWinSize = 5
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        self.enhance_image = enhance_image
        
        self.marker_size = marker_size
        
        # Load calibration if provided
        self.camera_matrix = None
        self.dist_coeffs = None
        if calibration_file and Path(calibration_file).exists():
            self.load_calibration(calibration_file)
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.total_markers = 0
        self.start_time = time.time()
        self.fps_list = []
        
    def load_calibration(self, calib_file):
        """Load camera calibration from file"""
        try:
            with open(calib_file, 'r') as f:
                data = yaml.safe_load(f)
                self.camera_matrix = np.array(data['camera_matrix'])
                self.dist_coeffs = np.array(data['distortion_coefficients'])
                print(f"✓ Loaded calibration from {calib_file}")
        except Exception as e:
            print(f"✗ Failed to load calibration: {e}")
            
    def detect_markers(self, image):
        """
        Detect ArUco markers in an image with enhanced preprocessing
        
        Returns:
            corners: List of detected marker corners
            ids: List of detected marker IDs
            image_with_markers: Image with markers drawn
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 图像增强 - 提高检测鲁棒性
        if self.enhance_image:
            # 1. 直方图均衡化 - 改善对比度
            gray = cv2.equalizeHist(gray)
            
            # 2. 高斯模糊 - 减少噪声（轻微）
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 3. 锐化（可选，根据需要）
            # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            # gray = cv2.filter2D(gray, -1, kernel)
        
        # 主检测
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # 多尺度检测 - 如果检测到的markers少于预期
        # 对图像进行轻微缩放再检测，可以找到更多markers
        if ids is None or len(ids) < 8:
            # 尝试稍微放大图像（1.1倍）
            h, w = gray.shape
            scaled_up = cv2.resize(gray, (int(w*1.1), int(h*1.1)), interpolation=cv2.INTER_CUBIC)
            corners_up, ids_up, _ = self.detector.detectMarkers(scaled_up)
            
            # 调整坐标回原始尺寸
            if ids_up is not None and len(ids_up) > 0:
                for corner in corners_up:
                    corner /= 1.1
                
                # 合并结果（去重）
                if ids is None:
                    corners, ids = corners_up, ids_up
                else:
                    existing_ids = set(ids.flatten())
                    for i, id_val in enumerate(ids_up.flatten()):
                        if id_val not in existing_ids:
                            corners = list(corners) + [corners_up[i]]
                            ids = np.vstack([ids, ids_up[i:i+1]])
                            existing_ids.add(id_val)
        
        # 二次检测 - 使用稍微缩小的图像
        if ids is None or len(ids) < 8:
            scaled_down = cv2.resize(gray, (int(w*0.9), int(h*0.9)), interpolation=cv2.INTER_AREA)
            corners_down, ids_down, _ = self.detector.detectMarkers(scaled_down)
            
            if ids_down is not None and len(ids_down) > 0:
                for corner in corners_down:
                    corner /= 0.9
                
                if ids is None:
                    corners, ids = corners_down, ids_down
                else:
                    existing_ids = set(ids.flatten())
                    for i, id_val in enumerate(ids_down.flatten()):
                        if id_val not in existing_ids:
                            corners = list(corners) + [corners_down[i]]
                            ids = np.vstack([ids, ids_down[i:i+1]])
                            existing_ids.add(id_val)
        
        # Create visualization
        image_with_markers = image.copy()
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(image_with_markers, corners, ids)
            
            # If calibration is available, estimate pose and draw axes
            if self.camera_matrix is not None:
                for i, corner in enumerate(corners):
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corner, self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    # Draw coordinate axes
                    cv2.drawFrameAxes(image_with_markers, self.camera_matrix, 
                                     self.dist_coeffs, rvec[0], tvec[0], self.marker_size * 0.5)
            
            # Update statistics
            self.detection_count += 1
            self.total_markers += len(ids)
        
        # Draw rejected candidates (for debugging)
        if len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(image_with_markers, rejected, 
                                         borderColor=(100, 0, 255))
        
        self.frame_count += 1
        
        return corners, ids, rejected, image_with_markers
    
    def draw_info(self, image, corners, ids, fps):
        """Draw detection information on image"""
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # Semi-transparent info panel
        panel_height = 150
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        
        # Detection info
        y_offset = 25
        line_height = 25
        
        # Markers detected
        if ids is not None and len(ids) > 0:
            text = f"Markers Detected: {len(ids)}"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # List marker IDs
            ids_str = "IDs: " + ", ".join([str(id[0]) for id in ids])
            cv2.putText(image, ids_str, (10, y_offset + line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show pose info if available
            if self.camera_matrix is not None and corners is not None:
                for i, corner in enumerate(corners):
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corner, self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    tvec = tvec[0][0]
                    distance = np.linalg.norm(tvec)
                    pose_text = f"ID {ids[i][0]}: {distance*100:.1f} cm"
                    cv2.putText(image, pose_text, (10, y_offset + line_height * (i+2)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            text = "No Markers Detected"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(image, fps_text, (w - 150, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(image, frame_text, (w - 150, y_offset + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def print_statistics(self):
        """Print detection statistics"""
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        avg_markers = self.total_markers / self.detection_count if self.detection_count > 0 else 0
        
        print("\n" + "="*60)
        print("Detection Statistics:")
        print("="*60)
        print(f"Total Frames: {self.frame_count}")
        print(f"Frames with Detections: {self.detection_count}")
        print(f"Detection Rate: {detection_rate:.1f}%")
        print(f"Total Markers Detected: {self.total_markers}")
        print(f"Average Markers per Frame: {avg_markers:.2f}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Elapsed Time: {elapsed_time:.2f}s")
        print("="*60)


def test_from_camera(camera_id=0, dictionary='DICT_7X7_1000', marker_size=0.095, 
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
    
    # Fall back to standard camera
    if cap is None:
        if not REALSENSE_AVAILABLE and use_realsense:
            print(f"⚠ pyrealsense2 not installed. Using standard camera {camera_id}...")
        
        print(f"Opening standard camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"✗ Failed to open camera {camera_id}")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print(f"✓ Standard camera opened successfully")
        print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # 使用鲁棒模式检测多个markers
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


def test_from_image(image_path, dictionary='DICT_7X7_1000', marker_size=0.095, 
                    calibration_file=None, robust_mode=True, enhance_image=True):
    """Test ArUco detection from a static image"""
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"✗ Failed to load image: {image_path}")
        return
    
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    tester = ArUcoDetectionTester(dictionary, marker_size, calibration_file,
                                   robust_mode=robust_mode, enhance_image=enhance_image)
    
    # Detect markers
    start_time = time.time()
    corners, ids, rejected, display_image = tester.detect_markers(image)
    detection_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Print results
    print("\n" + "="*60)
    print("Detection Results:")
    print("="*60)
    print(f"Detection Time: {detection_time:.2f} ms")
    print(f"Markers Detected: {len(ids) if ids is not None else 0}")
    
    if ids is not None and len(ids) > 0:
        print(f"Marker IDs: {[id[0] for id in ids]}")
        
        # If calibration available, show distances
        if tester.camera_matrix is not None:
            print("\nPose Estimation:")
            for i, corner in enumerate(corners):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, marker_size, tester.camera_matrix, tester.dist_coeffs
                )
                tvec = tvec[0][0]
                distance = np.linalg.norm(tvec)
                print(f"  ID {ids[i][0]}: Position = ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})m, Distance = {distance:.3f}m")
    
    print(f"Rejected Candidates: {len(rejected)}")
    print("="*60)
    
    # Display image
    display_image = tester.draw_info(display_image, corners, ids, 0)
    
    window_name = "ArUco Detection Test - Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_image)
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Ask to save result
    save = input("\nSave result image? (y/n): ").lower()
    if save == 'y':
        output_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(output_path, display_image)
        print(f"✓ Saved result to: {output_path}")


def test_from_video(video_path, dictionary='DICT_7X7_1000', marker_size=0.095, 
                   calibration_file=None, robust_mode=True, enhance_image=True):
    """Test ArUco detection from a video file"""
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Video opened successfully")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    
    tester = ArUcoDetectionTester(dictionary, marker_size, calibration_file,
                                   robust_mode=robust_mode, enhance_image=enhance_image)
    
    window_name = "ArUco Detection Test - Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("\nControls:")
    print("  ESC - Quit")
    print("  SPACE - Pause/Resume")
    print("  S - Save screenshot")
    print("\nStarting detection...\n")
    
    paused = False
    last_time = time.time()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ End of video")
                break
            
            # Calculate processing FPS
            current_time = time.time()
            process_fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            
            # Detect markers
            corners, ids, rejected, display_image = tester.detect_markers(frame)
            
            # Draw info
            display_image = tester.draw_info(display_image, corners, ids, process_fps)
            
            # Show progress
            progress = (tester.frame_count / frame_count * 100) if frame_count > 0 else 0
            cv2.putText(display_image, f"Progress: {progress:.1f}%", 
                       (10, display_image.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
            filename = f"aruco_video_{timestamp}.jpg"
            cv2.imwrite(filename, display_image)
            print(f"✓ Saved screenshot: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    tester.print_statistics()


def list_available_dictionaries():
    """List all available ArUco dictionaries"""
    dictionaries = [
        'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
        'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
        'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
        'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
        'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16h5', 'DICT_APRILTAG_25h9',
        'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11'
    ]
    
    print("\nAvailable ArUco Dictionaries:")
    print("="*60)
    for d in dictionaries:
        print(f"  - {d}")
    print("="*60)


def load_config_defaults(config_file='config.yaml'):
    """Load default parameters from config file"""
    defaults = {
        'dictionary': 'DICT_7X7_1000',
        'marker_size': 0.095,
        'calibration_file': 'camera_calibration.yaml',
        'width': 1280,
        'height': 720,
        'use_realsense': True  # Default to RealSense if available
    }
    
    if Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                defaults['dictionary'] = config.get('aruco', {}).get('dictionary', defaults['dictionary'])
                defaults['marker_size'] = config.get('aruco', {}).get('marker_size', defaults['marker_size'])
                defaults['calibration_file'] = config.get('calibration', {}).get('calibration_file', defaults['calibration_file'])
                defaults['width'] = config.get('camera', {}).get('resolution', {}).get('width', defaults['width'])
                defaults['height'] = config.get('camera', {}).get('resolution', {}).get('height', defaults['height'])
                # Check if there's a camera type preference in config
                # Assume RealSense by default (can be overridden with --standard-camera flag)
                print(f"✓ Loaded defaults from {config_file}")
                if REALSENSE_AVAILABLE:
                    print(f"✓ RealSense support available")
                else:
                    print(f"⚠ RealSense support not available (pyrealsense2 not installed)")
        except Exception as e:
            print(f"⚠ Could not load config file: {e}. Using hardcoded defaults.")
    else:
        print(f"⚠ Config file '{config_file}' not found. Using hardcoded defaults.")
    
    return defaults


def main():
    # Load defaults from config.yaml
    defaults = load_config_defaults('config.yaml')
    
    parser = argparse.ArgumentParser(
        description='ArUco Marker Detection Test Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default camera (uses config.yaml defaults)
  python test_aruco_detection.py --camera
  
  # Test with specific camera and calibration
  python test_aruco_detection.py --camera 0 --calibration camera_calibration.yaml
  
  # Test with image file
  python test_aruco_detection.py --image test.jpg
  
  # Test with video file
  python test_aruco_detection.py --video test.mp4
  
  # Use different ArUco dictionary
  python test_aruco_detection.py --camera --dictionary DICT_4X4_50
  
  # Use custom config file
  python test_aruco_detection.py --camera --config my_config.yaml
  
  # List available dictionaries
  python test_aruco_detection.py --list-dicts
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--camera', type=int, nargs='?', const=0, metavar='ID',
                            help='Use camera (default: 0)')
    input_group.add_argument('--image', type=str, metavar='PATH',
                            help='Test with image file')
    input_group.add_argument('--video', type=str, metavar='PATH',
                            help='Test with video file')
    input_group.add_argument('--list-dicts', action='store_true',
                            help='List available ArUco dictionaries')
    
    # Config file
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file (default: config.yaml)')
    
    # Detection parameters (with defaults from config)
    parser.add_argument('--dictionary', type=str, default=defaults['dictionary'],
                       help=f"ArUco dictionary (default from config: {defaults['dictionary']})")
    parser.add_argument('--marker-size', type=float, default=defaults['marker_size'],
                       help=f"Marker size in meters (default from config: {defaults['marker_size']})")
    parser.add_argument('--calibration', type=str, default=defaults['calibration_file'],
                       help=f"Camera calibration file (default from config: {defaults['calibration_file']})")
    
    # Camera settings (with defaults from config)
    parser.add_argument('--width', type=int, default=defaults['width'],
                       help=f"Camera width (default from config: {defaults['width']})")
    parser.add_argument('--height', type=int, default=defaults['height'],
                       help=f"Camera height (default from config: {defaults['height']})")
    
    # Camera type selection
    camera_type_group = parser.add_mutually_exclusive_group()
    camera_type_group.add_argument('--realsense', action='store_true', default=defaults['use_realsense'],
                                  help='Use RealSense camera (default if available)')
    camera_type_group.add_argument('--standard-camera', action='store_true',
                                  help='Force use of standard OpenCV camera')
    
    # Detection optimization options
    parser.add_argument('--robust', action='store_true', default=True,
                       help='Use robust detection mode (optimized for multiple markers, default: ON)')
    parser.add_argument('--no-robust', dest='robust', action='store_false',
                       help='Disable robust mode (use standard parameters)')
    parser.add_argument('--enhance', action='store_true', default=True,
                       help='Apply image enhancement (default: ON)')
    parser.add_argument('--no-enhance', dest='enhance', action='store_false',
                       help='Disable image enhancement')
    
    args = parser.parse_args()
    
    # If custom config specified, reload defaults
    if args.config != 'config.yaml':
        defaults = load_config_defaults(args.config)
        # Update args with new defaults if not explicitly set by user
        # (This is a simplified approach; more complex logic could check sys.argv)
    
    # If no arguments, show help
    if len(vars(args)) == 0 or all(v is None or v == parser.get_default(k) 
                                    for k, v in vars(args).items()):
        parser.print_help()
        return
    
    # List dictionaries
    if args.list_dicts:
        list_available_dictionaries()
        return
    
    # Validate calibration file
    calibration_file = args.calibration if Path(args.calibration).exists() else None
    if calibration_file is None:
        print(f"⚠ Calibration file '{args.calibration}' not found. Pose estimation disabled.")
    
    # Determine camera type
    use_realsense = not args.standard_camera  # Use RealSense unless explicitly disabled
    
    # Run appropriate test
    try:
        if args.camera is not None:
            test_from_camera(args.camera, args.dictionary, args.marker_size, 
                           calibration_file, args.width, args.height, use_realsense,
                           args.robust, args.enhance)
        elif args.image:
            test_from_image(args.image, args.dictionary, args.marker_size, 
                          calibration_file, args.robust, args.enhance)
        elif args.video:
            test_from_video(args.video, args.dictionary, args.marker_size, 
                          calibration_file, args.robust, args.enhance)
        else:
            print("Please specify an input source (--camera, --image, or --video)")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

