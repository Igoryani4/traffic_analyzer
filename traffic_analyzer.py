
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
        x1, y1, x2, y2 = box.xyxy[0].tolist()
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
    if len(all_reports) < top_n:
        print(f"  -> Найдено всего {len(all_reports)} отчётов, будут использованы все.")
        return {r['filename']: r for r in all_reports}

    most_crowded = sorted(all_reports, key=lambda r: r['total_vehicles'], reverse=True)[:top_n]
    most_dense = sorted(all_reports, key=lambda r: r['density'], reverse=True)[:top_n]

    top_examples = {r['filename']: r for r in most_crowded}

    top_examples.update({r['filename']: r for r in most_dense})

    return top_examples


def generate_visualizations(model, examples_to_visualize, source_dir, output_dir, conf):
    """Создаёт визуальные артефакты (изображения с аннотациями) для отчёта."""
    annotated_examples = []

    os.makedirs(output_dir, exist_ok=True)

    for filename, report in examples_to_visualize.items():
        original_path = os.path.join(source_dir, filename)

        results = model(original_path, conf=conf, verbose=False)
        annotated_img = results[0].plot()

        save_path = os.path.join(output_dir, f"highlight_{filename}")
        cv2.imwrite(save_path, annotated_img)
        
        stats_text = (
            f"Имя файла: {report['filename']}\n"
            f"Тип сцены: {report.get('scene_type', 'N/A')}\n"
            f"Всего ТС: {report['total_vehicles']}\n"
            f"  - Легковые автомобили: {report.get('car', 0)}\n"
            f"  - Грузовики: {report.get('truck', 0)}\n"
            f"Плотность объектов: {report['density']:.2%}\n"
            f"Средний размер объекта: {report['avg_bbox_area']:.0f} пикс."
        )
        
        annotated_examples.append({
            'path': save_path,
            'report': report,
            'stats': stats_text
        })
 
    return annotated_examples


def classify_scene(report, thresholds):
    count = report['total_vehicles']
    density = report['density']

    # Empty scene - нет транспортных средств
    if count == 0:
        return 'Empty'
    
    # Single Big Object - аномалия: 1-2 объекта создают высокую плотность
    if count <= 2 and density > thresholds['single_density']:
        return 'Single Big Object'

    # Traffic Jam - много машин и высокая плотность (пробка)
    if count > thresholds['jam_count'] and density >= thresholds['jam_density']:
        return 'Traffic Jam'

    # Heavy Traffic - много машин, но не пробка
    if count >= thresholds['heavy_count']:
        return 'Heavy Traffic'

    # Sparse Traffic - все остальные случаи (несколько машин, свободное движение)
    return 'Sparse Traffic'


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
    
def test_classify_scene():
    """(ТЕСТ) Проверяет корректность работы классификатора сцен classify_scene."""
    # Определяем тестовые пороги
    test_thresholds = {
        'jam_count': 10, 'jam_density': 0.3,
        'heavy_count': 5, 'single_density': 0.15,
    }

    # Создаём набор тестовых сценариев (кейсов)
    test_cases = [
        # Имя теста, Входной отчет, Ожидаемый результат
        ("Пустая сцена", {'total_vehicles': 0, 'density': 0.0}, 'Empty'),
        
        ("Явная пробка", {'total_vehicles': 15, 'density': 0.4}, 'Traffic Jam'),
        
        ("Граничный случай пробки (по количеству)", {'total_vehicles': 11, 'density': 0.31}, 'Traffic Jam'),
        
        ("Не пробка (не хватает плотности)", {'total_vehicles': 15, 'density': 0.29}, 'Heavy Traffic'),
        
        ("Не пробка (не хватает количества)", {'total_vehicles': 10, 'density': 0.4}, 'Heavy Traffic'),
        
        ("Аномалия: одна большая фура", {'total_vehicles': 1, 'density': 0.2}, 'Single Big Object'),
         
        ("Аномалия: две большие машины", {'total_vehicles': 2, 'density': 0.16}, 'Single Big Object'),
        
        ("Не аномалия (слишком низкая плотность)", {'total_vehicles': 1, 'density': 0.14}, 'Sparse Traffic'),
        
        ("Плотное движение", {'total_vehicles': 7, 'density': 0.2}, 'Heavy Traffic'),
        
        ("Граничный случай плотного движения", {'total_vehicles': 6, 'density': 0.1}, 'Heavy Traffic'),
        
        ("Свободная дорога (мало машин)", {'total_vehicles': 4, 'density': 0.1}, 'Sparse Traffic'),
    ]

    # Прогоняем все тесты в цикле
    for i, (case_name, report, expected_class) in enumerate(test_cases):
        actual_class = classify_scene(report, test_thresholds)
        
        assert actual_class == expected_class, \
            f"ОШИБКА в кейсе '{case_name}': Ожидалось '{expected_class}', но получено '{actual_class}'"


def test_find_highlight_examples():
    # Создаём искусственный набор данных (mock data)
    mock_reports = [
        # Лидер по количеству
        {'filename': 'crowded.jpg', 'total_vehicles': 20, 'density': 0.3},
        # Просто средний файл
        {'filename': 'normal_1.jpg', 'total_vehicles': 8, 'density': 0.15},
        # Лидер по плотности
        {'filename': 'dense.jpg', 'total_vehicles': 5, 'density': 0.6},
        # Второй по количеству
        {'filename': 'crowded_2.jpg', 'total_vehicles': 18, 'density': 0.25},
        # Второй по плотности и третий по количеству => топ по обоим критериям
        {'filename': 'dense_and_crowded.jpg', 'total_vehicles': 15, 'density': 0.5},
        # Ещё один средний файл
        {'filename': 'normal_2.jpg', 'total_vehicles': 2, 'density': 0.1},
        # Третий по плотности
        {'filename': 'dense_3.jpg', 'total_vehicles': 4, 'density': 0.4}
    ]

    top_examples = find_highlight_examples(mock_reports, top_n=2)

    expected_filenames = {'crowded.jpg', 'crowded_2.jpg', 'dense.jpg', 'dense_and_crowded.jpg'}

    assert len(top_examples) == 4, \
        f"ОШИБКА: Ожидалось 4 уникальных примера, но получено {len(top_examples)}"

    result_filenames = set(top_examples.keys())
    assert result_filenames == expected_filenames, \
        f"ОШИБКА: Набор файлов не совпадает. Найдено: {result_filenames}, Ожидалось: {expected_filenames}"
 
    small_reports = [{'filename': 'a.jpg', 'total_vehicles': 1, 'density': 0.1}]
    top_small = find_highlight_examples(small_reports, top_n=3)
    assert len(top_small) == 1, \
        "ОШИБКА: Неверная обработка, когда отчётов меньше, чем top_n"
    assert 'a.jpg' in top_small, \
        "ОШИБКА: Потерян единственный отчёт при обработке малого набора"



class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Пробуем использовать красивые шрифты, если они доступны (их мы скачивали в нашу папку ttf/)
        regular_font_path = os.path.join('ttf', 'DejaVuSans.ttf')
        bold_font_path = os.path.join('ttf', 'DejaVuSans-Bold.ttf')

        if os.path.exists(regular_font_path) and os.path.exists(bold_font_path):
            self.add_font('DejaVu', '', regular_font_path)
            self.add_font('DejaVu', 'B', bold_font_path)

            self.font_family = 'DejaVu'
        else:
            # Если шрифты не найдены, используем стандартный
            self.font_family = 'Arial'

    def header(self):
        """Создаёт шапку для каждой страницы"""
        self.set_font(self.font_family, 'B', 15)
        self.cell(0, 10, 'Аналитический отчёт по дорожной обстановке', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font(self.font_family, '', 8)
        self.cell(0, 5, f'Дата генерации: {datetime.date.today().strftime("%d.%m.%Y")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """Добавляет номера страниц в подвале"""
        self.set_y(-15)
        self.set_font(self.font_family, 'B', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def chapter_title(self, title):
        """Создаёт заголовок раздела"""
        self.set_font(self.font_family, 'B', 12)
        self.cell(0, 10, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_body(self, body):
        """Добавляет основной текст"""
        self.set_font(self.font_family, '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def add_image_section(self, title, image_path, stats_text):
        """Добавляет секцию с изображением и статистикой"""
        self.add_page()
        self.chapter_title(title)

        # Центрируем изображение на странице
        image_width = 100
        page_width = self.w - 2 * self.l_margin
        x_position = (page_width - image_width) / 2 + self.l_margin
        self.image(image_path, x=x_position, y=None, w=image_width)
        self.ln(5)
        self.set_font(self.font_family, '', 10) 
        self.multi_cell(0, 5, stats_text)

if __name__ == "__main__":
    # Добавился вызов в основной цикл функции для теста
    
    # test_get_valid_image_paths()
    # test_analyze_image_metrics()
    # test_classify_scene()
    # test_find_highlight_examples()

    parser = argparse.ArgumentParser(
        description="Инструмент для анализа дорожного трафика на основе YOLOv8.",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "mode", 
        choices=['experiment', 'report'],
        help="Режим работы скрипта:\n"
             " 'experiment' - запуск одного прогона для сбора CSV-статистики.\n"
             " 'report' - полный цикл анализа с генерацией PDF-отчёта."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.45, 
        help="Порог уверенности (confidence) для детекции. (По умолчанию: 0.45)"
    )

    args = parser.parse_args()

    if args.mode == 'experiment':
        print(f"experiment mode")

        TARGET_CLASSES = ['car', 'truck']

        image_paths = get_valid_image_paths('data')

        model = YOLO('yolov8n.pt')
        
        all_reports = []

        for path in tqdm(image_paths, desc=f"Анализ [conf={args.conf}]"):
            try:
                image = cv2.imread(path)
                h, w, _ = image.shape
                
                results = model(image, conf=args.conf, verbose=False)
                
                metrics = analyze_image_metrics(results[0].boxes, h, w, model.names, TARGET_CLASSES)
                
                metrics['filename'] = os.path.basename(path)
                all_reports.append(metrics)
            except Exception as e:
                print(f"\nКритическая ошибка при обработке файла {path}: {e}")

        if all_reports:
            os.makedirs('experiments', exist_ok=True)
            csv_path = os.path.join('experiments', f'analysis_conf_{args.conf}.csv')

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_reports[0].keys())
                writer.writeheader()
                writer.writerows(all_reports)

    elif args.mode == 'report':
        print(f"report mode")

        THRESHOLDS = {
            'jam_count': 10, 'jam_density': 0.3,
            'heavy_count': 5, 'single_density': 0.15,
        }
        TARGET_CLASSES = ['car', 'truck']
        SOURCE_DIR = 'data'
        REPORT_OUTPUT_DIR = 'report_output'

        model = YOLO('yolov8n.pt')
        image_paths = get_valid_image_paths(SOURCE_DIR)
        
        all_reports = []
        for path in tqdm(image_paths, desc="Анализ изображений"):
            try:
                image = cv2.imread(path)
                h, w, _ = image.shape
                image_area = h * w 
                results = model(image, conf=args.conf, verbose=False)

                metrics = analyze_image_metrics(results[0].boxes, image_area, model.names, TARGET_CLASSES)
                metrics['scene_type'] = classify_scene(metrics, THRESHOLDS)
                metrics['filename'] = os.path.basename(path)
                all_reports.append(metrics)
            except Exception as e:
                print(f"\nКритическая ошибка при обработке файла {path}: {e}")

        if not all_reports:
            exit()

        top_examples = find_highlight_examples(all_reports)
        annotated_examples = generate_visualizations(model, top_examples, SOURCE_DIR, REPORT_OUTPUT_DIR, args.conf)

        pdf = PDFReport()
        pdf.add_page()

        pdf.chapter_title("1. Общая сводка по проанализированным данным")
        scene_types = [r['scene_type'] for r in all_reports]
        summary_text = (
            f"Всего обработано изображений: {len(all_reports)}\n"
            f"Использованный порог уверенности: {args.conf}\n\n"
            f"ОБЩАЯ СТАТИСТИКА ТРАНСПОРТА:\n"
            f"  - Всего найдено ТС: {sum(r['total_vehicles'] for r in all_reports)}\n"
            f"  - Легковые автомобили: {sum(r.get('car', 0) for r in all_reports)}\n"
            f"  - Грузовики: {sum(r.get('truck', 0) for r in all_reports)}\n\n"
            f"КЛАССИФИКАЦИЯ СЦЕН:\n"
            f"  - Пробка/Затор: {scene_types.count('Traffic Jam')} изображений\n"
            f"  - Плотное движение: {scene_types.count('Heavy Traffic')} изображений\n"
            f"  - Свободная дорога: {scene_types.count('Sparse Traffic')} изображений\n"
            f"  - Аномалии (крупный объект): {scene_types.count('Single Big Object')} изображений\n"
            f"  - Пустые сцены: {scene_types.count('Empty')} изображений"
        )
        pdf.chapter_body(summary_text)

        if annotated_examples:
            pdf.chapter_title("Примеры показательных сцен")
            sorted_examples = sorted(annotated_examples, key=lambda x: x['report']['density'], reverse=True)
            for i, example in enumerate(sorted_examples):
                title = f"Пример #{i+1}: {example['report']['filename']}"
                pdf.add_image_section(title, example['path'], example['stats'])

        pdf_output_path = os.path.join(REPORT_OUTPUT_DIR, "traffic_analysis_report.pdf")
        pdf.output(pdf_output_path)

        print(f"Результат сохранён в: {pdf_output_path}")