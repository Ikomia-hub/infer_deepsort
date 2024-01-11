# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import copy
# Your imports below
from infer_deepsort.deep_sort_pytorch.utils.parser import get_config
from infer_deepsort.deep_sort_pytorch.deep_sort import DeepSort
import torch
import os

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


class TrackerDeepSort:
    def __init__(self):
        # initialize deepsort
        self.cfg = get_config()
        filename = os.path.join(os.path.dirname(__file__), 'deep_sort_pytorch/configs/deep_sort.yaml')
        self.cfg.merge_from_file(filename)

        self.deepsort = DeepSort(reid_model_name=self.cfg.DEEPSORT.REID_MODEL,
                                 max_dist=self.cfg.DEEPSORT.MAX_DIST,
                                 min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=self.cfg.DEEPSORT.MAX_AGE,
                                 n_init=self.cfg.DEEPSORT.N_INIT,
                                 nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

    @staticmethod
    def compute_color_for_labels(label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return color

    def track_deepsort(self, obj_detect_in, img, param):
        xywh_boxes = []
        confidences = []
        labels = []
        labels_to_track = param.categories.split(',')
        self.deepsort.min_confidence = param.conf_thres

        # Adapt detections to deep sort input format
        for obj in obj_detect_in.get_objects():
            if param.categories == "all" or obj.label in labels_to_track:
                # box to deep sort format
                xywh_boxes.append([obj.box[0]+obj.box[2]/2, obj.box[1]+obj.box[3]/2, obj.box[2], obj.box[3]])
                confidences.append(obj.confidence)
                labels.append(obj.label)

        if len(confidences):
            xywhs = torch.Tensor(xywh_boxes)
            confs = torch.Tensor(confidences)
            # pass detections to deepsort
            outputs, out_confidences, out_labels = self.deepsort.update(xywhs, confs, labels, img)
        else:
            self.deepsort.increment_ages()
            outputs = []
            out_confidences = []
            out_labels = []

        return outputs, out_confidences, out_labels


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class DeepSortParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.categories = "all"
        self.conf_thres = 0.5

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.categories = param_map["categories"]
        self.conf_thres = float(param_map["conf_thres"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {"categories": self.categories,
                     "conf_thres": str(self.conf_thres)}
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class DeepSortProcess(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.remove_input(1)
        # Add object detection input
        self.add_input(dataprocess.CObjectDetectionIO())
        # Add object detection output
        self.add_output(dataprocess.CObjectDetectionIO())

        # Create parameters class
        if param is None:
            self.set_param_object(DeepSortParam())
        else:
            self.set_param_object(copy.deepcopy(param))
        self.tracker = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def load_model(self):
        # Set cache dir in the algorithm folder to simplify deployment
        os.environ["TORCH_HOME"] = (os.path.join(os.path.dirname(__file__), "models"))

        self.tracker = TrackerDeepSort()

        # Reset torch cache dir for next algorithms in the workflow
        os.environ.pop("TORCH_HOME")

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.begin_task_run()

        if self.tracker is None:
            self.load_model()

        # Get input :
        img_in = self.get_input(0)
        obj_detect_in = self.get_input(1)

        # Get parameters :
        param = self.get_param_object()

        # Get image from input/output (numpy array):
        src_image = img_in.get_image()

        # Call to the process main routine
        outputs, confidences, labels = self.tracker.track_deepsort(obj_detect_in, src_image, param)

        # Step progress bar:
        self.emit_step_progress()

        # Set image of input/output (numpy array):
        img_out = self.get_output(0)
        img_out.set_image(src_image)

        # Object detection output
        obj_detect_out = self.get_output(1)
        obj_detect_out.init("DeepSort", 0)

        if len(outputs) > 0:
            box_xyxy = outputs[:, :4]
            identities = outputs[:, -1]

            for i in range(len(identities)):
                xyxy = box_xyxy[i]
                x = float(xyxy[0])
                y = float(xyxy[1])
                w = float(xyxy[2] - xyxy[0])
                h = float(xyxy[3] - xyxy[1])
                track_id = identities[i]
                color = self.tracker.compute_color_for_labels(track_id)
                obj_detect_out.add_object(int(track_id), labels[i], confidences[i], x, y, w, h, color)

        # Step progress bar:
        self.emit_step_progress()

        # Call endTaskRun to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class DeepSortProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_deepsort"
        self.info.short_description = "Multiple Object Tracking algorithm (MOT) combining a deep association metric" \
                                     "with the well known SORT algorithm for better performance."
        self.info.authors = "Nicolai Wojke†, Alex Bewley, Dietrich Paulus†"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Tracking"
        self.info.version = "1.0.4"
        self.info.icon_path = "icons/logo.png"
        self.info.article = "Simple Online and Realtime Tracking with a deep association metric"
        self.info.journal = ""
        self.info.year = 2017
        self.info.license = "GPL-3.0"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/1703.07402.pdf"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_deepsort"
        self.info.original_repository = "https://github.com/nwojke/deep_sort"
        # Keywords used for search
        self.info.keywords = "multiple,object,tracking,cnn,SORT,Kalman"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_TRACKING"

    def create(self, param=None):
        # Create process object
        return DeepSortProcess(self.info.name, param)
