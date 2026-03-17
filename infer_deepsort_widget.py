# PyQt GUI framework
from PyQt6.QtWidgets import *

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

from infer_deepsort.infer_deepsort_process import DeepSortParam


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class DeepSortWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = DeepSortParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        self.spin_conf = pyqtutils.append_double_spin(self.grid_layout, "Confidence min",
                                                      self.parameters.conf_thres, 0.0, 1.0, 0.1, 2)

        self.edit_categories = pyqtutils.append_edit(self.grid_layout, "Categories", self.parameters.categories)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.conf_thres = self.spin_conf.value()
        self.parameters.categories = self.edit_categories.text()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class DeepSortWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_deepsort"

    def create(self, param):
        # Create widget object
        return DeepSortWidget(param, None)
