import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'petra_info.settings')
django.setup()
from accounts.models import CCTV, LineZone, PolygonZone, Detection, tracker_status, ScriptStatus
import cv2
from django.utils import timezone
from django.core.files.base import ContentFile
from ultralytics import YOLO
import numpy as np
import supervision as sv
import time
from django.utils.crypto import get_random_string
# Load the YOLOv8 model
model = YOLO("yolov8n.pt")
names = model.names
tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()

def update_script_status(script_name, status):
    if ScriptStatus.objects.filter(script_name=script_name).exists():
        ScriptStatus.objects.update_or_create(script_name=script_name, defaults={'status': status})
    else:
        ScriptStatus.objects.create(script_name=script_name, status=status)

def wait_for_cameras(camera):
    # check start_detection status
    while True:
        camera = CCTV.objects.get(name=camera.name)
        if camera.start_detection:
            break
        time.sleep(5)
        update_script_status(f"Detection Model - {camera.name}", 'Waiting for start detection')
def wait_for_zones(camera):
    """
    Function to wait for the Polygon and Line zones to be available for the camera.
    """
    line_zone = None
    polygon_zone = None
    while True:
        try:
            # Try fetching LineZone and PolygonZone from the database
            line_zone = LineZone.objects.get(camera=camera)
            polygon_zone = PolygonZone.objects.get(camera=camera)
            break  # Exit the loop once both zones are available
        except (LineZone.DoesNotExist, PolygonZone.DoesNotExist):
            # If zones are not available, wait and retry
            update_script_status(f"Detection Model - {camera.name}", 'Waiting for zones')
            time.sleep(5)  # Wait for 5 seconds before checking again

    return line_zone, polygon_zone

def wait_for_stream(camera):
    """
    Function to wait for the camera stream to be available.
    """
    while True:
        try:
            camera = CCTV.objects.get(name=camera.name)
            cap = cv2.VideoCapture(camera.rtsp)
            update_script_status(f"Detection Model - {camera.name}", 'Waiting for stream')
            if cap.isOpened():
                cap.release()
                break  # Exit the loop if the stream is available
        except Exception as e:
            print(f"Error opening camera stream: {e}")
            time.sleep(5)  # Wait for 5 seconds before trying again
    process_camera_stream(camera.name)

def process_camera_stream(camera_id):
    camera = CCTV.objects.get(name=camera_id)
    print(f"Processing camera: {camera.name}")
    # Wait for the camera to start detection
    cap = cv2.VideoCapture(camera.rtsp)
    resize_width = 1200
    resize_height = 720
    frame_counter = 0
    start_point = None
    end_point = None
    polygon_points = None
    # Wait for the LineZone and PolygonZone to be available
    line_zone_data, polygon_zone_data = wait_for_zones(camera)
    # Process LineZone data
    linezone_coords = np.array(line_zone_data.coordinates)
    start_point = sv.Point(int(((linezone_coords[0]['x'])/100)*resize_width), int(((linezone_coords[0]['y'])/100)*resize_height))
    end_point = sv.Point(int(((linezone_coords[1]['x'])/100)*resize_width), int(((linezone_coords[1]['y'])/100)*resize_height))
    polygon_points = polygon_zone_data.coordinates
    for i,point in enumerate(polygon_points):
        polygon_points[i]=[int(point['x']/100*resize_width),int(point['y']/100*resize_height)]
    polygon_points =np.array(polygon_points)
    camera.running_status = True
    camera.save()
    # Define the zones for detection
    line_zone = sv.LineZone(start=start_point, end=end_point, triggering_anchors=[sv.Position.BOTTOM_CENTER])
    polygon_zone = sv.PolygonZone(polygon=polygon_points)
    while cap.isOpened():
        update_script_status(f"Detection Model - {camera.name}", 'Running')
        wait_for_cameras(camera)
        success, frame = cap.read()
        if not success:
            break
        if frame_counter % 2 != 0:
            frame_counter += 1
            continue
        org_frame = frame.copy()    
        frame = cv2.resize(frame, (resize_width, resize_height))
        # Perform YOLO detection
        results = model(frame, verbose=False, classes=[3])
        if results[0] is not None:
            detections = sv.Detections.from_ultralytics(results[0])
            mask = polygon_zone.trigger(detections)
            detections = detections[mask]
            detections = tracker.update_with_detections(detections)
            detections = smoother.update_with_detections(detections)
            crossed_in, crossed_out = line_zone.trigger(detections)
            if crossed_in.any() or crossed_out.any():
                print("Vehicle crossed the line")
            for cord, tracker_id, in_status, out_status in zip(detections.xyxy, detections.tracker_id, crossed_in, crossed_out):
                x1, y1, x2, y2 = cord
                x1 = int(x1 * org_frame.shape[1] / resize_width)
                y1 = int(y1 * org_frame.shape[0] / resize_height)
                x2 = int(x2 * org_frame.shape[1] / resize_width)
                y2 = int(y2 * org_frame.shape[0] / resize_height)
                tracker_id= str(tracker_id)+'-'+str(camera.category)
                if in_status:
                    tracker_status.objects.update_or_create(tracker_id=tracker_id, defaults={'status': 'In',"category": camera.category, 'last_detection_time': timezone.now(), 'completed': False})
                if tracker_status.objects.filter(tracker_id=tracker_id, status='In', category=camera.category).exists():
                    # Save cropped vehicle image
                    vehicle_image = org_frame[y1:y2, x1:x2]
                    _, buffer = cv2.imencode('.jpg', vehicle_image)
                    image_file = ContentFile(buffer.tobytes())
                    filename = f"vehicle_{tracker_id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
                    # add a random string to the filename to avoid overwriting using np.random.randint
                    image_file.name = filename+str(np.random.randint(0,1000))+".jpg"
                    Detection.objects.create(
                        tracker=tracker_status.objects.get(tracker_id=tracker_id),
                        unique_id = get_random_string(length=10),
                        camera=camera.name,
                        timestamp=timezone.now(),
                        image=image_file,
                    )
        frame_counter += 1
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    wait_for_stream(camera)

if __name__ == '__main__':
    # get arguments from command line
    import sys
    camera_id = sys.argv[1]
    try:
        update_script_status(f"Detection Model - {camera_id}", 'Running')
        process_camera_stream(camera_id)
    except Exception as e:
        update_script_status(f"Detection Model - {camera_id}", 'Error')