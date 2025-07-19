import sys
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QInputDialog, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class ProcessWorker(QThread):
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.process = None
        self._terminate = False

    def run(self):
        try:
            self.process = subprocess.Popen([sys.executable] + self.args)
            while self.process.poll() is None:
                if self._terminate:
                    self.process.terminate()
                    self.process.wait()
                    self.finished.emit(-999)
                    return
                self.msleep(100)
            code = self.process.returncode
            self.finished.emit(code)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._terminate = True
        if self.process and self.process.poll() is None:
            self.process.terminate()

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System - Main Menu")
        self.setFixedSize(400, 440)
        self.setStyleSheet("background-color: #f4f7fa;")
        layout = QVBoxLayout(self)
        title = QLabel("<b>Face Recognition System</b>")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(20)
        btn_reg = QPushButton("User Registration")
        btn_reg.clicked.connect(self.run_registration)
        btn_rec = QPushButton("Face Recognition")
        btn_rec.clicked.connect(self.run_recognition)
        btn_auth = QPushButton("Authentication")
        btn_auth.clicked.connect(self.run_authentication)
        btn_monitor = QPushButton("System Monitor")
        btn_monitor.clicked.connect(self.run_monitor)
        btn_opt = QPushButton("Optimize Models")
        btn_opt.clicked.connect(self.run_optimize)
        for btn in [btn_reg, btn_rec, btn_auth, btn_monitor, btn_opt]:
            btn.setFont(QFont("Segoe UI", 14))
            btn.setFixedHeight(44)
            layout.addWidget(btn)
            layout.addSpacing(8)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.stop_btn.setStyleSheet("background: #e53935; color: #fff; border-radius: 8px;")
        self.stop_btn.setFixedHeight(44)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_process)
        layout.addWidget(self.stop_btn)
        layout.addSpacing(8)
        layout.addStretch()
        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #888;")
        layout.addWidget(self.status)
        self.worker = None

    def run_with_worker(self, args, done_msg, fail_msg):
        if self.worker is not None:
            QMessageBox.warning(self, "Already Running", "A process is already running.")
            return
        self.worker = ProcessWorker(args)
        self.worker.finished.connect(lambda code: self.on_worker_finished(code, done_msg, fail_msg))
        self.worker.error.connect(self.on_worker_error)
        self.stop_btn.setEnabled(True)
        self.status.setText("Running... Click Stop to terminate.")
        self.worker.start()

    def on_worker_finished(self, code, done_msg, fail_msg):
        self.stop_btn.setEnabled(False)
        self.worker = None
        if code == 0:
            QMessageBox.information(self, "Done", done_msg)
        elif code == -999:
            QMessageBox.information(self, "Stopped", "Process was stopped by user.")
        else:
            QMessageBox.critical(self, "Error", fail_msg)
        self.status.setText("")

    def on_worker_error(self, msg):
        self.stop_btn.setEnabled(False)
        self.worker = None
        QMessageBox.critical(self, "Error", f"Failed to start process: {msg}")
        self.status.setText("")

    def stop_process(self):
        if self.worker is not None:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopping...")

    def run_registration(self):
        self.status.setText("Running registration...")
        self.run_with_worker(["main.py", "--mode", "registration"], "Registration completed.", "Registration failed.")
        # username, ok = QInputDialog.getText(self, "User Registration", "Enter username:")
        # if ok and username:
        #     self.status.setText("Running registration...")
        #     self.run_with_worker(["main.py", "--mode", "registration", "--username", username], "Registration completed.", "Registration failed.")

    def run_recognition(self):
        self.status.setText("Running recognition...")
        self.run_with_worker(["main.py", "--mode", "recognition"], "Recognition completed.", "Recognition failed.")

    def run_authentication(self):
        mode, ok = QInputDialog.getItem(self, "Authentication Mode", "Choose mode:", ["single", "continuous"], 0, False)
        if ok:
            self.status.setText("Running authentication...")
            self.run_with_worker(["main.py", "--mode", "authentication", "--auth-mode", mode], "Authentication completed.", "Authentication failed.")

    def run_monitor(self):
        self.status.setText("Running system monitor...")
        self.run_with_worker(["system_monitor.py"], "System monitor closed.", "System monitor failed.")

    def run_optimize(self):
        self.status.setText("Optimizing models...")
        try:
            self.run_with_worker(["optimize_tensorrt.py"], "Model optimization completed.", "Model optimization failed.")
        except Exception:
            self.run_with_worker(["./optimize_models.sh"], "Model optimization completed.", "Model optimization failed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec_()) 