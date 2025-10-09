import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from skgstat import Variogram
from pykrige.ok import OrdinaryKriging
import ezdxf
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QPushButton, QLabel, QLineEdit, QTextEdit, 
                             QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
                             QDoubleSpinBox, QSpinBox, QProgressBar, QGroupBox, QGridLayout,
                             QSplitter, QHeaderView, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

class KrigingThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.x = None
        self.y = None
        self.z = None
        self.variogram_params = None
        self.grid_x = None
        self.grid_y = None
        self.z_pred = None
        self.sigma = None
        
    def run(self):
        try:
            OK = OrdinaryKriging(
                self.x, self.y, self.z,
                variogram_model='gaussian',
                variogram_parameters=self.variogram_params,
                nlags=10
            )
            
            # Эмуляция прогресса
            for i in range(101):
                self.progress.emit(i)
                self.msleep(20)
                
            self.z_pred, self.sigma = OK.execute('grid', self.grid_x, self.grid_y)
            self.finished.emit()
            
        except Exception as e:
            print(f"Ошибка кригинга: {e}")

class KrigingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_data()
        self.init_ui()
        
    def init_data(self):
        self.x = None
        self.y = None
        self.z = None
        self.V = None
        self.nugget = None
        self.sill = None
        self.range_ = None
        self.grid_x = None
        self.grid_y = None
        self.z_pred = None
        self.sigma = None
        self.padding = 0.0
        self.grid_size = 100
        
    def init_ui(self):
        self.setWindowTitle("Кригинг с вариограммой Гаусса")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Создаем сплиттер для разделения на левую и правую части
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель - вкладки (уже)
        self.tab_widget = QTabWidget()
        self.tab_widget.setMaximumWidth(450)
        
        # Создаем вкладки
        self.tab_data = self.create_data_tab()
        self.tab_variogram = self.create_variogram_tab()
        self.tab_edit_variogram = self.create_edit_variogram_tab()
        self.tab_kriging = self.create_kriging_tab()
        self.tab_results = self.create_results_tab()
        
        self.tab_widget.addTab(self.tab_data, "Загрузка данных")
        self.tab_widget.addTab(self.tab_variogram, "Вариограмма")
        self.tab_widget.addTab(self.tab_edit_variogram, "Редактирование вариограммы")
        self.tab_widget.addTab(self.tab_kriging, "Кригинг")
        self.tab_widget.addTab(self.tab_results, "Сохранение результатов")
        
        # Правая панель - графики (больше места)
        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)
        plot_layout.setContentsMargins(5, 5, 5, 5)
        plot_layout.setSpacing(5)
        
        # Создаем фигуру и canvas для matplotlib
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.canvas)
        
        # Добавляем виджеты в сплиттер
        splitter.addWidget(self.tab_widget)
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([400, 1200])  # Больше места для графиков
        
        main_layout.addWidget(splitter)
        
    def create_data_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Группа загрузки данных
        group_load = QGroupBox("Загрузка данных")
        group_load.setStyleSheet("QGroupBox { font-weight: bold; }")
        load_layout = QVBoxLayout(group_load)
        load_layout.setSpacing(8)
        
        self.btn_load = QPushButton("Загрузить Excel файл")
        self.btn_load.setMinimumHeight(35)
        self.btn_load.clicked.connect(self.load_data)
        load_layout.addWidget(self.btn_load)
        
        load_layout.addWidget(QLabel("Информация о данных:"))
        self.data_info = QTextEdit()
        self.data_info.setReadOnly(True)
        self.data_info.setMaximumHeight(180)
        self.data_info.setStyleSheet("font-family: Consolas, monospace;")
        load_layout.addWidget(self.data_info)
        
        layout.addWidget(group_load)
        layout.addStretch()
        
        return widget
        
    def create_variogram_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.btn_variogram = QPushButton("Построить вариограмму")
        self.btn_variogram.setMinimumHeight(35)
        self.btn_variogram.clicked.connect(self.plot_empirical_variogram)
        layout.addWidget(self.btn_variogram)
        
        # Таблица параметров вариограммы (только для просмотра)
        layout.addWidget(QLabel("Параметры вариограммы:"))
        self.variogram_table = QTableWidget()
        self.variogram_table.setRowCount(4)
        self.variogram_table.setColumnCount(2)
        self.variogram_table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.variogram_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.variogram_table.setMaximumHeight(180)
        
        # Заполняем таблицу нередактируемыми значениями
        parameters = [
            ("Range (Диапазон)", ""),
            ("Sill (Силл)", ""),
            ("Nugget (Нугет)", ""),
            ("Модель", "Гаусс")
        ]
        
        for row, (param, value) in enumerate(parameters):
            self.variogram_table.setItem(row, 0, QTableWidgetItem(param))
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.variogram_table.setItem(row, 1, item)
        
        layout.addWidget(self.variogram_table)
        layout.addStretch()
        
        return widget
        
    def create_edit_variogram_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Поля для редактирования параметров
        group_edit = QGroupBox("Редактирование параметров")
        group_edit.setStyleSheet("QGroupBox { font-weight: bold; }")
        edit_layout = QGridLayout(group_edit)
        edit_layout.setSpacing(10)
        
        edit_layout.addWidget(QLabel("Range (Диапазон):"), 0, 0)
        self.edit_range = QDoubleSpinBox()
        self.edit_range.setRange(0.1, 1000.0)
        self.edit_range.setDecimals(2)
        self.edit_range.setMinimumHeight(30)
        edit_layout.addWidget(self.edit_range, 0, 1)
        
        edit_layout.addWidget(QLabel("Sill (Силл):"), 1, 0)
        self.edit_sill = QDoubleSpinBox()
        self.edit_sill.setRange(0.1, 1000.0)
        self.edit_sill.setDecimals(3)
        self.edit_sill.setMinimumHeight(30)
        edit_layout.addWidget(self.edit_sill, 1, 1)
        
        edit_layout.addWidget(QLabel("Nugget (Нугет):"), 2, 0)
        self.edit_nugget = QDoubleSpinBox()
        self.edit_nugget.setRange(0.0, 1000.0)
        self.edit_nugget.setDecimals(2)
        self.edit_nugget.setMinimumHeight(30)
        edit_layout.addWidget(self.edit_nugget, 2, 1)
        
        layout.addWidget(group_edit)
        
        self.btn_update_variogram = QPushButton("Обновить вариограмму")
        self.btn_update_variogram.setMinimumHeight(35)
        self.btn_update_variogram.clicked.connect(self.update_variogram)
        layout.addWidget(self.btn_update_variogram)
        
        layout.addStretch()
        return widget
        
    def create_kriging_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Параметры кригинга
        group_params = QGroupBox("Параметры кригинга")
        group_params.setStyleSheet("QGroupBox { font-weight: bold; }")
        params_layout = QGridLayout(group_params)
        params_layout.setSpacing(10)
        
        params_layout.addWidget(QLabel("Количество точек сетки:"), 0, 0)
        self.spin_grid_size = QSpinBox()
        self.spin_grid_size.setRange(10, 500)
        self.spin_grid_size.setValue(100)
        self.spin_grid_size.setMinimumHeight(30)
        params_layout.addWidget(self.spin_grid_size, 0, 1)
        
        params_layout.addWidget(QLabel("Величина отступа:"), 1, 0)
        self.spin_padding = QDoubleSpinBox()
        self.spin_padding.setRange(0.0, 100.0)
        self.spin_padding.setValue(0.0)
        self.spin_padding.setDecimals(2)
        self.spin_padding.setMinimumHeight(30)
        params_layout.addWidget(self.spin_padding, 1, 1)
        
        layout.addWidget(group_params)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        layout.addWidget(self.progress_bar)
        
        self.btn_kriging = QPushButton("Выполнить кригинг")
        self.btn_kriging.setMinimumHeight(40)
        self.btn_kriging.setStyleSheet("QPushButton { font-weight: bold; }")
        self.btn_kriging.clicked.connect(self.run_kriging)
        layout.addWidget(self.btn_kriging)
        
        layout.addStretch()
        return widget
        
    def create_results_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Информация о результатах
        group_info = QGroupBox("Информация о результатах")
        group_info.setStyleSheet("QGroupBox { font-weight: bold; }")
        info_layout = QVBoxLayout(group_info)
        self.results_info = QTextEdit()
        self.results_info.setReadOnly(True)
        self.results_info.setMaximumHeight(120)
        self.results_info.setStyleSheet("font-family: Consolas, monospace;")
        info_layout.addWidget(self.results_info)
        layout.addWidget(group_info)
        
        # Сохранение Excel
        group_excel = QGroupBox("Сохранение результатов")
        group_excel.setStyleSheet("QGroupBox { font-weight: bold; }")
        excel_layout = QVBoxLayout(group_excel)
        excel_layout.setSpacing(8)
        
        excel_layout.addWidget(QLabel("Желаемое количество точек:"))
        self.spin_target_points = QSpinBox()
        self.spin_target_points.setRange(50, 1000000)
        self.spin_target_points.setValue(10000)
        self.spin_target_points.setMinimumHeight(30)
        excel_layout.addWidget(self.spin_target_points)
        
        self.btn_save_excel = QPushButton("Сохранить в Excel")
        self.btn_save_excel.setMinimumHeight(35)
        self.btn_save_excel.clicked.connect(self.save_excel)
        excel_layout.addWidget(self.btn_save_excel)
        
        layout.addWidget(group_excel)
        
        # Сохранение DXF
        group_dxf = QGroupBox("Сохранение изополей")
        group_dxf.setStyleSheet("QGroupBox { font-weight: bold; }")
        dxf_layout = QVBoxLayout(group_dxf)
        dxf_layout.setSpacing(8)
        
        dxf_layout.addWidget(QLabel("Шаг горизонталей:"))
        self.spin_contour_step = QDoubleSpinBox()
        self.spin_contour_step.setRange(0.01, 10.0)
        self.spin_contour_step.setValue(0.15)
        self.spin_contour_step.setDecimals(3)
        self.spin_contour_step.setMinimumHeight(30)
        dxf_layout.addWidget(self.spin_contour_step)
        
        self.btn_save_dxf = QPushButton("Сохранить в DXF")
        self.btn_save_dxf.setMinimumHeight(35)
        self.btn_save_dxf.clicked.connect(self.save_dxf)
        dxf_layout.addWidget(self.btn_save_dxf)
        
        layout.addWidget(group_dxf)
        layout.addStretch()
        
        return widget
        
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузите файл Excel", "", "Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                data = pd.read_excel(file_path)
                if 'X' not in data.columns or 'Y' not in data.columns or 'Z' not in data.columns:
                    QMessageBox.critical(self, "Ошибка", "Файл должен содержать столбцы X, Y и Z.")
                    return
                    
                self.x = data['X'].values
                self.y = data['Y'].values
                self.z = data['Z'].values
                
                # Вывод информации о данных
                num_points = len(self.x)
                distances = pdist(np.vstack((self.x, self.y)).T)
                min_distance = np.min(distances)
                max_distance = np.max(distances)
                min_z = np.min(self.z)
                max_z = np.max(self.z)
                
                info_text = f"""Количество точек: {num_points}
Минимальное расстояние: {min_distance:.2f}
Максимальное расстояние: {max_distance:.2f}
Минимальное Z: {min_z:.2f}
Максимальное Z: {max_z:.2f}"""
                
                self.data_info.setText(info_text)
                QMessageBox.information(self, "Успех", "Данные успешно загружены!")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {str(e)}")
                
    def plot_empirical_variogram(self):
        if self.x is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные!")
            return
            
        try:
            self.V = Variogram(
                coordinates=np.vstack((self.x, self.y)).T,
                values=self.z, 
                model='gaussian'
            )
            range_, sill, nugget = self.V.parameters
            self.range_ = range_
            self.sill = round(sill, 3)
            self.nugget = nugget
            
            # Обновляем поля редактирования
            self.edit_range.setValue(range_)
            self.edit_sill.setValue(sill)
            self.edit_nugget.setValue(nugget)
            
            # Обновляем таблицу (только для просмотра)
            self.update_variogram_table()
            
            # Строим график
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Экспериментальная вариограмма
            ax.scatter(self.V.bins, self.V.experimental, label='Экспериментальная вариограмма')
            
            # Теоретическая вариограмма
            h = np.linspace(0, np.max(self.V.bins), 100)
            theoretical = nugget + (sill - nugget) * (1 - np.exp(-(h ** 2) / (range_ ** 2)))
            ax.plot(h, theoretical, 'r-', label='Теоретическая вариограмма (модель Гаусса)')
            
            ax.set_xlabel('Расстояние (h)')
            ax.set_ylabel('Полудисперсия (γ(h))')
            ax.set_title('Экспериментальная и теоретическая вариограмма', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True)
            
            # Увеличиваем размер шметок для лучшей читаемости
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить вариограмму: {str(e)}")
            
    def update_variogram_table(self):
        if self.range_ is not None and self.sill is not None and self.nugget is not None:
            parameters = [
                ("Range (Диапазон)", f"{self.range_:.2f}"),
                ("Sill (Силл)", f"{self.sill:.3f}"),
                ("Nugget (Нугет)", f"{self.nugget:.2f}"),
                ("Модель", "Гаусс")
            ]
            
            for row, (param, value) in enumerate(parameters):
                self.variogram_table.setItem(row, 0, QTableWidgetItem(param))
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.variogram_table.setItem(row, 1, item)
                
    def update_variogram(self):
        if self.V is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала постройте вариограмму!")
            return
            
        new_range = self.edit_range.value()
        new_sill = self.edit_sill.value()
        new_nugget = self.edit_nugget.value()
        
        if new_sill <= new_nugget:
            QMessageBox.critical(self, "Ошибка", "Sill должен быть больше Nugget.")
            return
            
        # Обновляем параметры
        self.range_ = new_range
        self.sill = round(new_sill, 3)
        self.nugget = new_nugget
        
        # Обновляем таблицу просмотра
        self.update_variogram_table()
        
        # Строим обновленный график
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Экспериментальная вариограмма
        ax.scatter(self.V.bins, self.V.experimental, label='Экспериментальная вариограмма')
        
        # Обновленная теоретическая вариограмма
        h = np.linspace(0, np.max(self.V.bins), 100)
        updated_theoretical = self.nugget + (self.sill - self.nugget) * (
            1 - np.exp(-(h ** 2) / (self.range_ ** 2)))
        ax.plot(h, updated_theoretical, 'r-', label='Обновлённая теоретическая вариограмма')
        
        ax.set_xlabel('Расстояние (h)')
        ax.set_ylabel('Полудисперсия (γ(h))')
        ax.set_title('Обновлённая теоретическая вариограмма', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True)
        
        # Увеличиваем размер шметок для лучшей читаемости
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        self.canvas.draw()
        
    def run_kriging(self):
        if self.x is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные!")
            return
            
        if self.nugget is None or self.sill is None or self.range_ is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала постройте вариограмму!")
            return
            
        try:
            self.grid_size = self.spin_grid_size.value()
            self.padding = self.spin_padding.value()
            
            self.grid_x = np.linspace(
                min(self.x) - self.padding,
                max(self.x) + self.padding,
                self.grid_size
            )
            self.grid_y = np.linspace(
                min(self.y) - self.padding,
                max(self.y) + self.padding,
                self.grid_size
            )
            
            # Запускаем кригинг в отдельном потоке
            self.kriging_thread = KrigingThread()
            self.kriging_thread.x = self.x
            self.kriging_thread.y = self.y
            self.kriging_thread.z = self.z
            self.kriging_thread.variogram_params = {
                'sill': self.sill,
                'range': self.range_,
                'nugget': self.nugget
            }
            self.kriging_thread.grid_x = self.grid_x
            self.kriging_thread.grid_y = self.grid_y
            
            self.kriging_thread.progress.connect(self.progress_bar.setValue)
            self.kriging_thread.finished.connect(self.on_kriging_finished)
            self.kriging_thread.start()
            
            self.btn_kriging.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить кригинг: {str(e)}")
            
    def on_kriging_finished(self):
        self.z_pred = self.kriging_thread.z_pred
        self.sigma = self.kriging_thread.sigma
        
        self.btn_kriging.setEnabled(True)
        self.progress_bar.setValue(100)
        
        # Обновляем информацию о результатах
        self.update_results_info()
        
        # Показываем результаты
        self.show_kriging_results()
        
    def update_results_info(self):
        if self.z_pred is not None:
            min_z_pred = np.min(self.z_pred)
            max_z_pred = np.max(self.z_pred)
            diff_z_pred = max_z_pred - min_z_pred
            total_points = len(self.grid_x) * len(self.grid_y)
            
            info_text = f"""Минимальное Z_pred: {min_z_pred:.2f}
Максимальное Z_pred: {max_z_pred:.2f}
Разница: {diff_z_pred:.2f}
Всего точек: {total_points}"""
            
            self.results_info.setText(info_text)
        
    def show_kriging_results(self):
        # Изолинии кригинга
        self.figure.clear()
        
        # 2D контур - больше места для графиков
        ax1 = self.figure.add_subplot(121)
        contour = ax1.contourf(self.grid_x, self.grid_y, self.z_pred, levels=20, cmap='viridis')
        ax1.scatter(self.x, self.y, c='red', s=15, label='Исходные точки')
        ax1.set_xlabel('X', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Y', fontsize=11, fontweight='bold')
        ax1.set_title('Изополя после кригинга', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.set_aspect('equal')
        ax1.tick_params(axis='both', which='major', labelsize=9)
        self.figure.colorbar(contour, ax=ax1)
        
        # 3D поверхность с правильными пропорциями
        ax2 = self.figure.add_subplot(122, projection='3d')
        X, Y = np.meshgrid(self.grid_x, self.grid_y)
        
        # Вычисляем правильные масштабы для осей
        x_range = np.ptp(self.grid_x)
        y_range = np.ptp(self.grid_y)
        z_range = np.ptp(self.z_pred)
        
        # Создаем поверхность с правильными пропорциями
        surf = ax2.plot_surface(X, Y, self.z_pred, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Добавляем исходные точки
        ax2.scatter(self.x, self.y, self.z, c='red', s=15, label='Исходные точки')
        
        # Настраиваем метки и заголовок
        ax2.set_xlabel('X', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Y', fontsize=11, fontweight='bold')
        ax2.set_zlabel('Z', fontsize=11, fontweight='bold')
        ax2.set_title('3D поверхность после кригинга', fontsize=12, fontweight='bold')
        
        # Устанавливаем равные масштабы для осей X и Y, а для Z настраиваем отдельно
        max_range = max(x_range, y_range)
        ax2.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range * 0.5])
        
        # Увеличиваем размер шрифта для 3D графика
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()
        
        QMessageBox.information(self, "Успех", "Кригинг успешно выполнен!")
        
    def save_excel(self):
        if self.z_pred is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните кригинг!")
            return
            
        try:
            target_points = self.spin_target_points.value()
            total_points = len(self.grid_x) * len(self.grid_y)
            
            # Сохраняем исходные данные для восстановления
            original_grid_x = self.grid_x.copy()
            original_grid_y = self.grid_y.copy()
            original_z_pred = self.z_pred.copy()
            
            grid_x_reduced = self.grid_x
            grid_y_reduced = self.grid_y
            z_pred_reduced = self.z_pred
            
            actual_points = total_points
            
            if target_points < total_points:
                # Вычисляем шаг для уменьшения количества точек
                step_ratio = np.sqrt(total_points / target_points)
                step_x = max(1, int(step_ratio))
                step_y = max(1, int(step_ratio))
                
                # Применяем шаг к сетке
                grid_x_reduced = self.grid_x[::step_x]
                grid_y_reduced = self.grid_y[::step_y]
                z_pred_reduced = self.z_pred[::step_x, ::step_y]
                
                actual_points = len(grid_x_reduced) * len(grid_y_reduced)
                
                # Корректируем шаг, чтобы получить максимально близкое к target_points количество
                while actual_points > target_points and step_x < len(self.grid_x) and step_y < len(self.grid_y):
                    step_x += 1
                    step_y += 1
                    grid_x_reduced = self.grid_x[::step_x]
                    grid_y_reduced = self.grid_y[::step_y]
                    z_pred_reduced = self.z_pred[::step_x, ::step_y]
                    actual_points = len(grid_x_reduced) * len(grid_y_reduced)
            
            # Создаем DataFrame с фактическим количеством точек
            results = pd.DataFrame({
                'X': np.repeat(grid_x_reduced, len(grid_y_reduced)),
                'Y': np.tile(grid_y_reduced, len(grid_x_reduced)),
                'Z_pred': z_pred_reduced.flatten()
            })
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить результаты", "kriging_results.xlsx", "Excel Files (*.xlsx)"
            )
            
            if file_path:
                results.to_excel(file_path, index=False)
                
                # Восстанавливаем исходные данные
                self.grid_x = original_grid_x
                self.grid_y = original_grid_y
                self.z_pred = original_z_pred
                
                QMessageBox.information(self, "Успех", 
                                      f"Результаты сохранены в {file_path}\n"
                                      f"Фактическое количество точек: {actual_points}")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результаты: {str(e)}")
            
    def save_dxf(self):
        if self.z_pred is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните кригинг!")
            return
            
        try:
            step = self.spin_contour_step.value()
            if step <= 0:
                QMessageBox.critical(self, "Ошибка", "Шаг горизонталей должен быть положительным числом.")
                return
                
            min_z_pred = np.min(self.z_pred)
            max_z_pred = np.max(self.z_pred)
            levels = np.arange(min_z_pred, max_z_pred, step)
            
            doc = ezdxf.new("R2010")
            msp = doc.modelspace()
            
            # Добавляем границу
            boundary_points = [
                (min(self.grid_x), min(self.grid_y), 0),
                (max(self.grid_x), min(self.grid_y), 0),
                (max(self.grid_x), max(self.grid_y), 0),
                (min(self.grid_x), max(self.grid_y), 0),
                (min(self.grid_x), min(self.grid_y), 0)
            ]
            msp.add_polyline3d(boundary_points)
            
            # Создаем временную фигуру для изолиний
            fig_temp = plt.figure()
            contours = plt.contour(self.grid_x, self.grid_y, self.z_pred, levels=levels)
            plt.close(fig_temp)
            
            # Добавляем изолинии в DXF
            for level_index, level in enumerate(contours.allsegs):
                for line in level:
                    if len(line) > 1:
                        height = contours.levels[level_index]
                        points = [(float(x), float(y), height) for x, y in line]
                        msp.add_polyline3d(points)
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить изополя", "isolines.dxf", "DXF Files (*.dxf)"
            )
            
            if file_path:
                doc.saveas(file_path)
                QMessageBox.information(self, "Успех", f"Изополя сохранены в {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить изополя: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = KrigingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
