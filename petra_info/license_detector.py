import cv2
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'petra_info.settings')
django.setup()
from accounts.models import Detection, tracker_status, Vehicle, ScriptStatus
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
dict_int_to_char={'0':"O",'1':"I",'2':"Z",'3':"B",'4':"A",'5':"S",'6':"G",'7':"T",'8':"B",'9':"G"}
dict_char_to_int={'O':"0",'I':"1",'Z':"2",'B':"3",'A':"4",'S':"5",'G':"6",'T':"7",'B':"8",'G':"9"}

def update_script_status(status):
    if not ScriptStatus.objects.filter(script_name="License Plate Model").exists():
        ScriptStatus.objects.create(script_name="License Plate Model", status=status)
    else: 
        obj=ScriptStatus.objects.get(script_name="License Plate Model")
        obj.status=status
        obj.save()

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
    state_codes=["TN","AP","KA","KL","DL","MH","GJ","RJ","UP","MP","HR","PB","WB","BR","OR","AS","JH","UK","HP","TR","ML","MZ","NL","SK","AR","AN","CH","DN","DD","LD","PY","JK","LA","CG","TS"]
    if None in result:
        return "UNKNOWN"
    if len(result) == 0 or len(result[0]) < 2:
        return "UNKNOWN"
    number_plate_text = result[0][0][1][0] + result[0][1][1][0]
    number_plate_text = re.sub(r'[^a-zA-Z0-9]', '', number_plate_text)
    if len(number_plate_text) == 10 or len(number_plate_text) == 9:
        return number_plate_text
    else:
        for state_code in state_codes:
            if state_code in number_plate_text:
                index = number_plate_text.index(state_code)
                number_plate_text = number_plate_text[index:]
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
        while True:
            try:
                detection.image.delete()
                detection.delete()
                break
            except Exception as e:
                print(f"Error deleting image: {e}")
                time.sleep(5)
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
    if largest_license_plate is None:
        return "UNREGISTERED",largest_orginal_image, None
    elif len(predictions) == 0:
        return  "UNKNOWN",largest_orginal_image, largest_license_plate
    elif len(predictions) == 1:
        return predictions[0][0] , largest_orginal_image, largest_license_plate
    else:
        # sort the predictions based on the least number of changes
        predictions = sorted(predictions,key=lambda x:x[1])
        return predictions[0][0] , largest_orginal_image, largest_license_plate

def detect_license_plate():
     while True:
        # Get tracker statuses that are not completed
        trackers = tracker_status.objects.filter(completed=False)
        update_script_status("Running")
        current_time = timezone.now()
        if trackers:
            for track in trackers:
                tracker_id = track.tracker_id
                last_detection_time = track.last_detection_time
                # Check if 30 seconds have passed since the last detection
                time_since_last_detection = current_time - last_detection_time
                if time_since_last_detection >= timedelta(seconds=20):
                    print(f"Processing tracker: {tracker_id} after waiting 20 seconds")
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
                        Vehicle.objects.create(
                            timestamp=timezone.now(),
                            camera=track.category,
                            license_plate=number_plate,
                            image=original_image_file,  # Save original image
                            license_plate_image=license_plate_image_file  # Save license plate image
                        )
                    elif largest_orginal_image is not None and largest_license_plate is not None:
                        Vehicle.objects.create(
                            timestamp=timezone.now(),
                            camera=track.category,
                            license_plate="UNKNOWN",
                            image=original_image_file,
                            license_plate_image=largest_license_plate
                        )
                    elif largest_license_plate is None:
                        Vehicle.objects.create(
                            timestamp=timezone.now(),
                            camera=track.category,
                            license_plate="UNREGISTERED",
                            image=original_image_file,
                            license_plate_image=None
                        )
                    # delete the tracker status
                    track.completed = True
                    track.save()
                    tracker_status.objects.filter(tracker_id=tracker_id).delete()

        else:
            print("No new trackers to process")
        time.sleep(5)

if __name__ == '__main__':
    detect_license_plate()
    try:
        update_script_status("Running")
        detect_license_plate()
    except Exception as e:
        print(f"Error: {e}")
        update_script_status("Error")
        


