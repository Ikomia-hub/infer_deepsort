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

    def track_deepsort(self, obj_data, img, param):
        xywh_boxes = []
        confidences = []
        labels = []
        labels_to_track = param.categories.split(',')
        self.deepsort.min_confidence = param.confidence

        # Adapt detections to deep sort input format
        for measure in obj_data:
            assert(len(measure) >= 2)
            obj_conf = measure[0]
            obj_box = measure[1]

            if param.categories == "all" or obj_box.label in labels_to_track:
                # box to deep sort format
                box = obj_box.values
                xywh_boxes.append([box[0]+box[2]/2, box[1]+box[3]/2, box[2], box[3]])
                confidences.append(obj_conf.values)
                labels.append(obj_box.label)

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
        self.confidence = 0.5

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.categories = param_map["categories"]
        self.confidence = float(param_map["confidence"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["categories"] = self.categories
        param_map["confidence"] = str(self.confidence)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class DeepSortProcess(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add numeric input
        self.addInput(dataprocess.CBlobMeasureIO())
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CBlobMeasureIO())

        # Create parameters class
        if param is None:
            self.setParam(DeepSortParam())
        else:
            self.setParam(copy.deepcopy(param))

        self.tracker = TrackerDeepSort()

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Examples :
        # Get input :
        img_in = self.getInput(0)
        measures_in = self.getInput(2)

        # Get parameters :
        param = self.getParam()

        # Get image from input/output (numpy array):
        src_image = img_in.getImage()

        # Call to the process main routine
        obj_data = measures_in.getMeasures()
        outputs, confidences, labels = self.tracker.track_deepsort(obj_data, src_image, param)

        # Step progress bar:
        self.emitStepProgress()

        # Set image of input/output (numpy array):
        img_out = self.getOutput(0)
        img_out.setImage(src_image)

        # Graphics output
        graphics_out = self.getOutput(1)
        graphics_out.setNewLayer("DeepSort")
        graphics_out.setImageIndex(0)
        # Numeric output
        numeric_out = self.getOutput(2)
        numeric_out.clearData()

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

                # Box
                prop_rect = core.GraphicsRectProperty()
                prop_rect.pen_color = color
                graphics_box = graphics_out.addRectangle(x, y, w, h, prop_rect)

                # Label
                prop_rect = core.GraphicsRectProperty()
                color_brush = color
                color_brush.append(255)
                prop_rect.pen_color = color_brush
                prop_rect.brush_color = color_brush
                graphics_out.addRectangle(x, y - 20, 20, 20, prop_rect)
                prop_text = core.GraphicsTextProperty()
                prop_text.font_size = 8
                prop_text.color = color
                prop_text.bold = True
                label = f"{track_id}:{labels[i]}"
                graphics_out.addText(label, x, y, prop_text)

                # object results
                results = []
                track_data = dataprocess.CObjectMeasure(dataprocess.CMeasure(core.MeasureId.CUSTOM, "Track ID"),
                                                        float(track_id),
                                                        graphics_box.getId(),
                                                        labels[i])
                confidence_data = dataprocess.CObjectMeasure(dataprocess.CMeasure(core.MeasureId.CUSTOM, "Confidence"),
                                                             confidences[i],
                                                             graphics_box.getId(),
                                                             labels[i])
                box_data = dataprocess.CObjectMeasure(dataprocess.CMeasure(core.MeasureId.BBOX),
                                                      [x, y, w, h],
                                                      graphics_box.getId(),
                                                      labels[i])
                results.append(track_data)
                results.append(confidence_data)
                results.append(box_data)
                numeric_out.addObjectMeasures(results)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class DeepSortProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_deepsort"
        self.info.shortDescription = "Multiple Object Tracking algorithm (MOT) combining a deep association metric" \
                                     "with the well known SORT algorithm for better performance."
        self.info.description = "Simple Online and Realtime Tracking (SORT) is a pragmatic approach to multiple " \
                                "object tracking with a focus on simple, effective algorithms. In this paper, we " \
                                "integrate appearance information to improve the performance of SORT. Due to this " \
                                "extension we are able to track objects through longer periods of occlusions, " \
                                "effectively reducing the number of identity switches. In spirit of the original " \
                                "framework we place much of the computational complexity into an offline " \
                                "pre-training stage where we learn a deep association metric on a large-scale person " \
                                "re-identification dataset. During online application, we establish " \
                                "measurement-to-track associations using nearest neighbor queries in visual " \
                                "appearance space. Experimental evaluation shows that our extensions reduce the " \
                                "number of identity switches by 45%, achieving overall competitive performance " \
                                "at high frame rates."
        self.info.authors = "Nicolai Wojke†, Alex Bewley, Dietrich Paulus†"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Tracking"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/logo.png"
        self.info.article = "Simple Online and Realtime Tracking with a deep association metric"
        self.info.journal = ""
        self.info.year = 2017
        self.info.license = "GPL-3.0"
        # URL of documentation
        self.info.documentationLink = "https://arxiv.org/pdf/1703.07402.pdf"
        # Code source repository
        self.info.repository = "https://github.com/nwojke/deep_sort"
        # Keywords used for search
        self.info.keywords = "multiple,object,tracking,cnn,SORT,Kalman"

    def create(self, param=None):
        # Create process object
        return DeepSortProcess(self.info.name, param)
