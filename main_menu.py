import sys
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QInputDialog, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QFont, QPalette, QColor
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
        self.setFixedSize(500, 700)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
            }
            QPushButton {
                background: rgba(255, 255, 255, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.4);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.5);
            }
            QLabel {
                color: white;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 40, 30, 30)
        
        # Add some top spacing
        layout.addSpacing(20)
        
        # Title with icon
        title = QLabel("🎯 Face Recognition System")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; margin-bottom: 15px; padding: 10px;")
        title.setWordWrap(True)
        title.setMinimumHeight(60)
        layout.addWidget(title)
        
        subtitle = QLabel("Select a feature to get started")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.8); margin-bottom: 25px; padding: 5px;")
        subtitle.setWordWrap(True)
        subtitle.setMinimumHeight(30)
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        # Feature buttons
        btn_reg = QPushButton("👤 User Registration")
        btn_reg.clicked.connect(self.run_registration)
        btn_auth = QPushButton("🔐 Authentication")
        btn_auth.clicked.connect(self.run_authentication)
        btn_monitor = QPushButton("📊 System Monitor")
        btn_monitor.clicked.connect(self.run_monitor)
        
        for btn in [btn_reg, btn_auth, btn_monitor]:
            btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
            btn.setFixedHeight(60)
            layout.addWidget(btn)
            layout.addSpacing(25)
        
        # Stop button with different styling
        self.stop_btn = QPushButton("🛑 Stop Process")
        self.stop_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: rgba(220, 53, 69, 0.8);
                border: 2px solid rgba(220, 53, 69, 0.9);
                border-radius: 15px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background: rgba(220, 53, 69, 0.9);
                border: 2px solid rgba(220, 53, 69, 1.0);
            }
            QPushButton:pressed {
                background: rgba(220, 53, 69, 1.0);
            }
            QPushButton:disabled {
                background: rgba(220, 53, 69, 0.3);
                color: rgba(255, 255, 255, 0.5);
            }
        """)
        self.stop_btn.setFixedHeight(60)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_process)
        layout.addWidget(self.stop_btn)
        
        layout.addSpacing(20)
        
        # Status area
        self.status = QLabel("Ready to use")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("""
            color: rgba(255, 255, 255, 0.9);
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            font-size: 12px;
        """)
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
        self.status.setText("Ready to use")

    def on_worker_error(self, msg):
        self.stop_btn.setEnabled(False)
        self.worker = None
        QMessageBox.critical(self, "Error", f"Failed to start process: {msg}")
        self.status.setText("Ready to use")

    def stop_process(self):
        if self.worker is not None:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopping...")

    def run_registration(self):
        self.status.setText("Running registration...")
        self.run_with_worker(["main.py", "--mode", "registration"], "Registration completed.", "Registration failed.")

    def run_authentication(self):
        mode, ok = QInputDialog.getItem(self, "Authentication Mode", "Choose mode:", ["single", "continuous"], 0, False)
        if ok:
            self.status.setText("Running authentication...")
            self.run_with_worker(["main.py", "--mode", "authentication", "--auth-mode", mode], "Authentication completed.", "Authentication failed.")

    def run_monitor(self):
        self.status.setText("Running system monitor...")
        self.run_with_worker(["system_monitor.py"], "System monitor closed.", "System monitor failed.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = MainMenu()
    menu.show()
    sys.exit(app.exec_()) 