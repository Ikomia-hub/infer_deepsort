import logging
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
    measure_in = t.getInput(1)
    measure_in.addObject(0, 'a', 1., 0, 0, 10, 10, [0, 0, 255])
    measure_in.addObject(1, 'b', 1., 10, 10, 20, 20, [255, 0, 255])

    params = t.get_parameters()
    # run once on set frame 1
    run_for_test(t)

    for label in ["all", "a", "b"]:
        params["categories"] = label
        t.set_parameters(params)
        yield run_for_test(t)
