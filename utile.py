import cv2
import time
import os
import natsort
from IPython import display
from collections import Counter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def image_processing(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    return img

def predict_imshow(img_path, results, show=True, colab=True):
    img = image_processing(img_path)
    img = results[0].plot()
    if show == False:
        return img
    
    elif colab == True:
        from google.colab.patches import cv2_imshow
        cv2_imshow(img)
        time.sleep(3)
        display.clear_output(wait=True)
        return img
    else:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        time.sleep(3)
        display.clear_output(wait=True)
        return img

def result_img_save(result_img):
    return cv2.imwrite("./save_data", result_img[0].plot())
        
def prediction_results(img_path, model, cuda=True):
    if cuda == True:
        return model.predict(image_processing(img_path), device="0")
    return model.predict(image_processing(img_path))

def detection_class(results):
    class_list = []
    for i in results[0].boxes.cls.cpu().numpy():
        class_list.append(results[0].names[int(i)])
    return class_list

def matching_class(names, counts):
    for name in names:
        number = len(name) * 1000
        if number >= counts:
            print(f"{name}: 출석 O")
        else:
            print(f"{name}: 출석 X")

class ImageHandler(FileSystemEventHandler):
    def __init__(self, model, save_folder_path, cuda):
        self.model = model
        self.cuda = cuda
        self.save_folder_path = save_folder_path
        if not os.path.isdir(self.save_folder_path):
            os.makedirs(self.save_folder_path)
    
    def image_processing(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        return img

    def prediction_results(self, img_path):
        if self.cuda == True:
            return self.model.predict(image_processing(img_path), device="0")
        return self.cudamodel.predict(image_processing(img_path))

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                result = self.prediction_results(event.src_path, self.model)
                self.result_img_save(result, event.src_path)  # 결과 이미지 저장
                attendance = self.detection_class(result)
                self.log_attendance(event.src_path, attendance)  # 출석 결과 로깅
            except Exception as e:
                print(f"Failed to process {event.src_path}: {e}")

    def result_img_save(self, result_img):
        return cv2.imwrite("./save_data", result_img[0].plot())

    def matching_class(names, counts):
        for name in names:
            number = len(name) * 1000
            if number >= counts:
                print(f"{name}: 출석 O")
            else:
                print(f"{name}: 출석 X")

    def log_attendance(self, src_path, attendance):
        with open(os.path.join(self.save_folder_path, "attendance_log.txt"), "a") as f:
            f.write(f"{src_path}: {attendance}\n")

def start_program(image_folder_path, model, save_folder_path):
    event_handler = ImageHandler(model, save_folder_path)
    observer = Observer()
    observer.schedule(event_handler, path=image_folder_path, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

def video(video_path, model, colab=True, show=True, cuda=True):
    if not os.path.isdir("./image_data"):
        os.mkdir("./image_data")
    if not os.path.isdir("./save_data"):
        os.mkdir("./save_data")
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    attendance_check = []
    while success:
        img_path = f"./image_data/{count:06d}.jpg"
        cv2.imwrite(img_path, image)
        success, image = vidcap.read()
        count += 1
    count = 0
    for path in natsort.natsorted(os.listdir('image_data')):
        save_img_path = f"./save_data/{count:06d}.jpg"
        img_path = os.path.join("./image_data", path)
        result = prediction_results(img_path, model, cuda)
        detection_img = predict_imshow(img_path, result, show, colab)
        cv2.imwrite(save_img_path, detection_img)
        attendance_check.append(detection_class(result))
        count += 1
    return matching_class(attendance_check)


# 테스트용
# def video(video_path, model, colab=True, cuda=True):
    # if not os.path.isdir("./image_data"):
    #     os.mkdir("./image_data")
    # if not os.path.isdir("./save_data"):
    #     os.mkdir("./save_data")
    # vidcap = cv2.VideoCapture(video_path)
    # success, image = vidcap.read()
    # count = 0
    # for _ in range(0, 10, 1):
    #     img_path = f"./image_data/{count:06d}.jpg"
    #     cv2.imwrite(img_path, image)
    #     success, image = vidcap.read()
    #     count += 1
    # count = 0
    # attendance_check = []
    # for path in natsort.natsorted(os.listdir('image_data')):
    #     save_img_path = f"./save_data/{count:06d}.jpg"
    #     img_path = os.path.join("./image_data", path)
    #     result = prediction_results(img_path, model, cuda)
    #     detection_img = predict_imshow(img_path, result, colab)
    #     cv2.imwrite(save_img_path, detection_img)
    #     attendance_check.append(detection_class(result))
    #     count += 1
    # return attendance_check