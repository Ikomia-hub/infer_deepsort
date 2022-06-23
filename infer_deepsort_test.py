import logging
from ikomia import dataprocess
from ikomia.core import task, ParamMap, MeasureId
from ikomia.utils.tests import run_for_test
import cv2

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer_deepsort =====")
    # image input
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[:, :, ::-1]
    input_img = t.getInput(0)
    input_img.setImage(img)

    # measure input
    measure_in = t.getInput(2)
    data = []
    confidence_data = dataprocess.CObjectMeasure(
        dataprocess.CMeasure(MeasureId.CUSTOM, "Confidence"), 0.8, 0, "car")
    box_data = dataprocess.CObjectMeasure(
        dataprocess.CMeasure(MeasureId.BBOX), [50, 50, 100, 100], 0, "car")
    data.append(confidence_data)
    data.append(box_data)
    measure_in.addObjectMeasures(data)

    params = task.get_parameters(t)
    # run once on set frame 1
    run_for_test(t)

    for label in ["all", "car"]:
        params["categories"] = label
        task.set_parameters(t, params)
        yield run_for_test(t)
