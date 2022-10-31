from pathlib import Path

import cv2
import time

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


def setup_model(model_path: str, confidence_threshold: float = 0.5, device: str = 'cuda'):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.MODEL.DEVICE = device
    return cfg, DefaultPredictor(cfg)


def run_model(model, image):
    t0 = time.time()
    outputs = model(image)
    return outputs, time.time() - t0


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


def main(file_path: str, model_path: str, confidence_threshold: float = 0.5,
         device: str = 'cuda', target_frame: int = None):

    cfg, model = setup_model(model_path, confidence_threshold, device)

    if Path(file_path).suffix[1:] in VID_FORMATS:
        is_video, cap = True, cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        frame_number = 0
    else:
        is_video, frame = False, cv2.imread(file_path)
    while True:
        # stop at frame if target_frame is specified
        if frame_number >= target_frame:
            break

        outputs, time_elapsed = run_model(model, frame)
        people_instances = outputs['instances'][outputs['instances'].pred_classes == 0]
        console_out = f"Image: {file_path}, Time elapsed: {time_elapsed}, People found: {len(people_instances)}"
        print(console_out)

        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(people_instances.to("cpu"))
        cv2.imshow("Output", out.get_image()[:, :, ::-1])

        if is_video:
            frame_number += 1
            cv2.waitKey(1)
            ret, frame = cap.read()
            if not ret:
                break
        else:
            cv2.waitKey(0)
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    file = "people.mp4"
    model_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    confidence_threshold = 0.5
    device = 'cpu'
    target_frame = 10

    main(file, model_path, confidence_threshold, device, target_frame)
