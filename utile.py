import cv2
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def image_processing(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    return img

def predict_imshow(img_path, model, colab=True):
    img = image_processing(img_path)
    results = model.predict(img)
    img = results[0].plot()
    
    if colab == True:
        from google.colab.patches import cv2_imshow
        cv2_imshow(img)
    else:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        
def result_img_save(result_img, model):
    return result_img[0].plot()
        
def prediction_results(img_path, model):
    return model.predict(image_processing(img_path))

def detection_class(results):
    class_list = []
    for i in results[0].boxes.cls.cpu().numpy():
        class_list.append(results[0].names[int(i)])
    return class_list


# class ImageHandler(FileSystemEventHandler):
#     def __init__(self, model, save_folder_path):
#         self.model = prediction_results()
#         self.save_folder_path = save_folder_path

#     def on_created(self, event):
#         if event.is_directory:
#             return
#         if event.src_path.lower().endswith((".png", ".jpg", ".jpeg")):
#             try:
#                 img = self.prediction_results(event.src_path)
#                 print(f"Processed image: {event.src_path}")
#             except Exception as e:
#                 print(f"Failed to process {event.src_path}: {e}")
    
# def start_program(image_folder_path, model):
#     event_handler = ImageHandler(model)
#     observer = Observer()
#     observer.schedule(event_handler, path=image_folder_path, recursive=False)
#     observer.start()
    
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#         observer.join()

def video(video_path, model):
    if not os.path.isdir("./image_data"):
        os.mkdir("./image_data")
    if not os.path.isdir("./save"):
        os.mkdir("./save")
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("./image_data/%06d.jpg" % count, image)
        success, image = vidcap.read()
        count += 1
    image_list = os.listdir()
    attendance_check = []
    for path in image_list:
        result = prediction_results(path)
        attendance_check.append(detection_class(result))
    return attendance_check.count()