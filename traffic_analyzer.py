
import os, cv2, csv, argparse, datetime
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def get_valid_image_paths(source_dir):
    # Инициализируем пустой список для хранения путей к валидным изображениям
    image_paths = []
    allowed_extensions = {'.jpg', '.jpeg', '.png'}

    # Итерируемся по отсортированному списку файлов в директории
    # sorted() обеспечивает предсказуемый порядок обработки файлов
    for filename in sorted(os.listdir(source_dir)):
        # Извлекаем расширение файла и приводим к нижнему регистру
        file_extension = os.path.splitext(filename)[1].lower()

        # Пропускаем файлы с неподходящим расширением
        if file_extension not in allowed_extensions:
            continue

        full_path = os.path.join(source_dir, filename)

        # Пытаемся прочитать изображение с помощью OpenCV
        image = cv2.imread(full_path)
        if image is None:
            continue

        # Если все проверки пройдены, добавляем путь в результирующий список
        image_paths.append(full_path)

    return image_paths


def analyze_image_metrics(detections, image_area, model_names, target_classes):    
    object_counts = {cls: 0 for cls in target_classes}
    total_bbox_area = 0

    for box in detections:
        class_id = int(box.cls[0])
        class_name = model_names[class_id]

        if class_name not in target_classes:
            continue

        object_counts[class_name] += 1
        
        # Получаем координаты и считаем площадь текущего бокса
        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2 - x1) * (y2 - y1)
        total_bbox_area += area

    total_vehicles = sum(object_counts.values())

    # Вычисляем метрики на основе переданной площади изображения
    density = total_bbox_area / image_area if image_area > 0 else 0
    avg_bbox_area = total_bbox_area / total_vehicles if total_vehicles > 0 else 0

    report = {
        'total_vehicles': total_vehicles,
        **object_counts,
        'density': round(density, 4),
        'avg_bbox_area': round(avg_bbox_area, 2)
    }
    
    return report


def find_highlight_examples(all_reports, top_n=3):
    """Находит наиболее показательные примеры в наборе данных."""
    pass


def generate_visualizations(model, examples_to_visualize, source_dir, output_dir, conf):
    """Создаёт визуальные артефакты (изображения с аннотациями) для отчёта."""
    pass


def classify_scene(report, thresholds):
    """Классифицирует сцену на основе метрик и порогов."""
    
    pass


def test_get_valid_image_paths():
    test_dir = "/Users/tochi/Documents/ML_CV_Learning/Learning_code/Yandex_practicum_conspection/БЛОК5_CV. Спринт 1. Детекция объектов/Тема1_Быстрый старт с Yolov8/3.Практика. Инференс YOLOv8 на наборе изображений/traffic_analyzer/data/test"
    
    # Файлы, которые должны быть найдены
    expected_files = ["image1.jpg", "image2.PNG"]
    # Файлы, которые должны быть проигнорированы
    ignored_files = ["corrupted.jpg", "document.txt", "archive.zip"]
    

    # Вызываем тестируемую функцию
    result_paths = get_valid_image_paths(test_dir)
    
    # Проверяем результаты с помощью assert
    assert len(result_paths) == len(expected_files), \
        f"ОШИБКА: Ожидалось {len(expected_files)} файла, но найдено {len(result_paths)}"

    result_filenames = sorted([os.path.basename(p) for p in result_paths])
    expected_filenames = sorted(expected_files)
    assert result_filenames == expected_filenames, \
        f"ОШИБКА: Имена файлов не совпадают. Найдено: {result_filenames}, Ожидалось: {expected_filenames}"
    

from types import SimpleNamespace


def test_analyze_image_metrics():
    # box.cls[0] и box.xyxy[0] должны быть похожи на тензоры
    fake_detections = [
        # Первый бокс: 'car'
        SimpleNamespace(cls=np.array([0]), xyxy=np.array([[10, 10, 110, 110]])),
        # Второй бокс: 'truck'
        SimpleNamespace(cls=np.array([1]), xyxy=np.array([[50, 50, 250, 250]])),
        # Третий бокс: еще один 'car'
        SimpleNamespace(cls=np.array([0]), xyxy=np.array([[20, 20, 70, 70]])),
        # Четвёртый бокс: неизвестный класс (id=5), должен быть проигнорирован
        SimpleNamespace(cls=np.array([5]), xyxy=np.array([[0, 0, 5, 5]]))
    ]
    
    # Эмулируем остальные входные данные
    image_area = 1000 * 1000  # 1,000,000 пикселей
    model_names = {0: 'car', 1: 'truck', 5: 'person'} # Словарь имен
    target_classes = ['car', 'truck']

    result_metrics = analyze_image_metrics(fake_detections, image_area, model_names, target_classes)

    assert result_metrics['car'] == 2, f"ОШИБКА: Ожидалось 2 машины, но найдено {result_metrics['car']}"
    assert result_metrics['truck'] == 1, f"ОШИБКА: Ожидалось 1 грузовик, но найдено {result_metrics['truck']}"

    assert result_metrics['total_vehicles'] == 3, f"ОШИБКА: Ожидалось 3 ТС, но найдено {result_metrics['total_vehicles']}"

    # Площади: (100*100) + (200*200) + (50*50) = 10000 + 40000 + 2500 = 52500
    # Плотность: 52500 / 1000000 = 0.0525
    expected_density = 0.0525
    assert np.isclose(result_metrics['density'], expected_density), \
        f"ОШИБКА: Ожидалась плотность {expected_density}, но получено {result_metrics['density']}"


...

if __name__ == "__main__":
    # Добавился вызов в основной цикл функции для теста
    test_analyze_image_metrics()
    # test_get_valid_image_paths()
