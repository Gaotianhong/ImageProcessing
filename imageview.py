from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import Qt


class ImageView(QGraphicsView):
    def __init__(self, parent):
        super().__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Alt:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Alt:
            self.setDragMode(QGraphicsView.NoDrag)
