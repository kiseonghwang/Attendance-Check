from ultralytics import YOLO

def model_import(model_pt, model_yaml=None, training=True, data_path=None, epochs=None, imgsz=None, batch=None, cuda=False):
    if training == True:
        model = YOLO(model_yaml)
        results = model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch)
        return results
    elif cuda == True:
        return YOLO(model_pt)
    return YOLO(model_pt)
        