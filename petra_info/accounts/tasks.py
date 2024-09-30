from background_task import background
from .models import *
import cv2
from datetime import timedelta
from django.utils import timezone
from django.core.files.base import ContentFile
from ultralytics import YOLO
import numpy as np
import supervision as sv
import time
import os
from paddleocr import PaddleOCR
import re
from ppocr.utils.logging import get_logger
import logging
logger = get_logger()
logger.setLevel(logging.ERROR)

ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=True) 
number_plate_model = YOLO("best.pt")
# Load the YOLOv8 model
model = YOLO("yolov8n.pt")
names = model.names
tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()
dict_int_to_char={'0':"O",'1':"I",'2':"Z",'3':"B",'4':"A",'5':"S",'6':"G",'7':"T",'8':"B",'9':"G"}
dict_char_to_int={'O':"0",'I':"1",'Z':"2",'B':"3",'A':"4",'S':"5",'G':"6",'T':"7",'B':"8",'G':"9"}

def wait_for_cameras(camera):
    # check start_detection status
    while True:
        camera = CCTV.objects.get(id=camera.id)
        if camera.start_detection:
            break
        time.sleep(5)
        print(f"Waiting for camera to start detection: {camera.name}")

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
            print(f"Waiting for zones to be configured for camera: {camera.name}")
            time.sleep(5)  # Wait for 5 seconds before checking again

    return line_zone, polygon_zone

def process_camera_stream(camera_id):
    camera = CCTV.objects.get(id=camera_id)
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
                time_string = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
                if in_status:
                    # create or update the status of the tracker
                    tracker_status.objects.update_or_create(tracker_id=tracker_id, defaults={'status': 'In', 'last_detection_time': timezone.now(), 'completed': False})
                   
                if tracker_status.objects.filter(tracker_id=tracker_id, status='In').exists():
                    # Save cropped vehicle image
                    vehicle_image = org_frame[y1:y2, x1:x2]
                    _, buffer = cv2.imencode('.jpg', vehicle_image)
                    image_file = ContentFile(buffer.tobytes())
                    filename = f"vehicle_{tracker_id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    image_file.name = filename

                    Detection.objects.create(
                        tracker=tracker_status.objects.get(tracker_id=tracker_id),
                        camera=camera.name,
                        timestamp=timezone.now(),
                        image=image_file,
                    )
        frame_counter += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def correct_number_plate(number):
    # check the following conditions
        # 1. Number plate should have atleast 9 characters and atmost 10 characters
        # 2. first 2 characters should be alphabets
        # 3. next 2 characters should be digits
        # 4. last 4 characters should be digits
        # 5. remaining characters should be alphabets
    if number[:2].isalpha() and number[2:4].isdigit() and number[-4:].isdigit() and number[4:-4].isalpha() and len(number) >= 9 and len(number) <= 10:
        return number
    else:
        # correct the number plate
        number_list = list(number)
        length = len(number)
        changes = 0
        if length == 10:
            if not number[:2].isalpha():
                if number[0] in dict_int_to_char:
                    number_list[0] = dict_int_to_char[number[0]]
                    changes += 1
                if number[1] in dict_int_to_char:
                    number_list[1] = dict_int_to_char[number[1]]
                    changes += 1
            if not number[2:4].isdigit():
                if number[2] in dict_char_to_int:
                    number_list[2] = dict_char_to_int[number[2]]
                    changes += 1

                if number[3] in dict_char_to_int:
                    number_list[3] = dict_char_to_int[number[3]]
                    changes += 1

            if not number[4:6].isalpha():
                if number[4] in dict_int_to_char:
                    number_list[4] = dict_int_to_char[number[4]]
                    changes += 1
                if number[5] in dict_int_to_char:
                    number_list[5] = dict_int_to_char[number[5]]
                    changes += 1
            if not number[-4:].isdigit():
                if number[-4] in dict_char_to_int:
                    number_list[-4] = dict_char_to_int[number[-4]]
                    changes += 1
                if number[-3] in dict_char_to_int:
                    number_list[-3] = dict_char_to_int[number[-3]]
                    changes += 1
                if number[-2] in dict_char_to_int:
                    number_list[-2] = dict_char_to_int[number[-2]]
                    changes += 1
                if number[-1] in dict_char_to_int:
                    number_list[-1] = dict_char_to_int[number[-1]]
                    changes += 1
        elif length == 9:
            if not number[:2].isalpha():
                if number[0] in dict_int_to_char:
                    number_list[0] = dict_int_to_char[number[0]]
                    changes += 1
                if number[1] in dict_int_to_char:
                    number_list[1] = dict_int_to_char[number[1]]
                    changes += 1
            if not number[2:4].isdigit():
                if number[2] in dict_char_to_int:
                    number_list[2] = dict_char_to_int[number[2]]
                    changes += 1
                if number[3] in dict_char_to_int:
                    number_list[3] = dict_char_to_int[number[3]]
                    changes += 1
            if not number[4].isalpha():
                if number[4] in dict_int_to_char:
                    number_list[4] = dict_int_to_char[number[4]]
                    changes += 1
            if not number[-4:].isdigit():
                if number[-4] in dict_char_to_int:
                    number_list[-4] = dict_char_to_int[number[-4]]
                    changes += 1
                if number[-3] in dict_char_to_int:
                    number_list[-3] = dict_char_to_int[number[-3]]
                    changes += 1
                if number[-2] in dict_char_to_int:
                    number_list[-2] = dict_char_to_int[number[-2]]
                    changes += 1
                if number[-1] in dict_char_to_int:
                    number_list[-1] = dict_char_to_int[number[-1]]
                    changes += 1
        return "".join(number_list) , changes
    
def get_number(number_plate):
    result = ocr.ocr(number_plate)
    if None in result:
        return "UNKNOWN"
    if len(result) == 0 or len(result[0]) < 2:
        return "UNKNOWN"
    number_plate_text = result[0][0][1][0] + result[0][1][1][0]
    number_plate_text = re.sub(r'[^a-zA-Z0-9]', '', number_plate_text)
    if len(number_plate_text) == 10 or len(number_plate_text) == 9:
        return number_plate_text
    return "UNKNOWN"

def process_images(tracker_id,padding=30):
    print(f"Processing images for tracker: {tracker_id}")
    predictions = []
    largest_orginal_image = None
    largest_orginal_image_area = 0
    largest_license_plate = None
    largest_license_plate_area = 0
    tracker_object = tracker_status.objects.get(tracker_id=tracker_id)
    for detection in Detection.objects.filter(tracker=tracker_object):
        image = cv2.imdecode(np.frombuffer(detection.image.read(), np.uint8), cv2.IMREAD_COLOR)
        # calculate the area of the image
        area = image.shape[0]*image.shape[1]
        if area > largest_orginal_image_area:
            largest_orginal_image_area = area
            largest_orginal_image = image

        results = number_plate_model.predict(image,verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections.with_nms(0.5)
        if len(detections)>0:
            # crop the detected number plate
            x1,y1,x2,y2=detections.xyxy[0]
            x1=int(x1)
            y1=int(y1)
            x2=int(x2)
            y2=int(y2)
            # add some padding to the bounding box
            padding = padding
            x1 = max(0, x1-padding)
            y1 = max(0, y1-padding)
            x2 = min(image.shape[1], x2+padding)
            y2 = min(image.shape[0], y2+padding)
            number_plate = image[y1:y2,x1:x2]
            # calculate the area of the number plate
            area = number_plate.shape[0]*number_plate.shape[1]
            if area > largest_license_plate_area:
                largest_license_plate_area = area
                largest_license_plate = number_plate
            number = get_number(number_plate)
            if number[:2].isalpha() and number[2:4].isdigit() and number[-4:].isdigit() and number[4:-4].isalpha() and len(number) >= 9 and len(number) <= 10:
                predictions.append((number,0))   
            else:
                number,changes = correct_number_plate(number)
                if number!="UNKNOWN" and number[:2].isalpha() and number[2:4].isdigit() and number[-4:].isdigit() and number[4:-4].isalpha() and len(number) >= 9 and len(number) <= 10:
                    predictions.append((number,changes))      
    # most repeated number plate is the correct number
    if len(predictions) == 0:
        return  "UNKNOWN",largest_orginal_image, None
    elif len(predictions) == 1:
        return predictions[0] , largest_orginal_image, largest_license_plate
    else:
        # sort the predictions based on the least number of changes
        predictions = sorted(predictions,key=lambda x:x[1])
        return predictions[0][0] , largest_orginal_image, largest_license_plate

def detect_license_plate():
    while True:
        # Get tracker statuses that are not completed
        trackers = tracker_status.objects.filter(completed=False)
        current_time = timezone.now()
        
        if trackers:
            for track in trackers:
                tracker_id = track.tracker_id
                last_detection_time = track.last_detection_time
                # Check if 30 seconds have passed since the last detection
                time_since_last_detection = current_time - last_detection_time
                
                if time_since_last_detection >= timedelta(seconds=30):
                    print(f"Processing tracker: {tracker_id} after waiting 30 seconds")
                    
                    # Perform license plate detection
                    number_plate, largest_orginal_image, largest_license_plate = process_images(tracker_id)     
                    # Encode the NumPy arrays (images) into JPEG format
                    if largest_orginal_image is not None:
                        _, original_image_buffer = cv2.imencode('.jpg', largest_orginal_image)
                        original_image_file = ContentFile(original_image_buffer.tobytes(), name=f"original_{tracker_id}.jpg")
                    
                    if largest_license_plate is not None:
                        _, license_plate_image_buffer = cv2.imencode('.jpg', largest_license_plate)
                        license_plate_image_file = ContentFile(license_plate_image_buffer.tobytes(), name=f"license_plate_{tracker_id}.jpg")

                    # Handle case where image data is valid and known
                    if number_plate != "UNKNOWN" and largest_orginal_image is not None and largest_license_plate is not None:
                        # Save the vehicle details with valid images
                        Vehicle.objects.update_or_create(
                            tracker=track,
                            timestamp=timezone.now(),
                            camera=track.status,
                            license_plate=number_plate,
                            image=original_image_file,  # Save original image
                            license_plate_image=license_plate_image_file  # Save license plate image
                        )
                    else:
                        # Save the vehicle details without license plate info
                        if largest_orginal_image is not None and largest_license_plate is not None:
                            Vehicle.objects.update_or_create(
                                tracker=track,
                                timestamp=timezone.now(),
                                camera=track.status,
                                license_plate="UNREGISTERED",
                                image=original_image_file,
                                license_plate_image=license_plate_image_file
                            )
                    
                    Detection.objects.filter(tracker=track).delete()
                    # Mark tracker as completed
                    track.completed = True 
                    track.save()

        else:
            print("No new trackers to process")
        time.sleep(5)