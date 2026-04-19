"""
GTSRB - German Traffic Sign Recognition Benchmark
Жол қозғалысы белгілерін автоматты тану
Автоматическое распознавание дорожных знаков

Зависимости / Dependencies:
    pip install tensorflow numpy pillow scikit-learn matplotlib tkinter requests

Использование / Usage:
    python gtsrb_classifier.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

# ─────────────────────────────────────────────
# Названия классов GTSRB (43 класса)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Ограничение скорости 20 км/ч",    # 0
    "Ограничение скорости 30 км/ч",    # 1
    "Ограничение скорости 50 км/ч",    # 2
    "Ограничение скорости 60 км/ч",    # 3
    "Ограничение скорости 70 км/ч",    # 4
    "Ограничение скорости 80 км/ч",    # 5
    "Конец ограничения 80 км/ч",       # 6
    "Ограничение скорости 100 км/ч",   # 7
    "Ограничение скорости 120 км/ч",   # 8
    "Обгон запрещён",                  # 9
    "Обгон грузовикам запрещён",       # 10
    "Приоритет на перекрёстке",        # 11
    "Главная дорога",                  # 12
    "Уступи дорогу",                   # 13
    "Стоп",                            # 14
    "Въезд запрещён (грузовики)",      # 15
    "Въезд запрещён",                  # 16
    "Осторожно: опасность",            # 17
    "Опасный поворот влево",           # 18
    "Опасный поворот вправо",          # 19
    "Двойной поворот",                 # 20
    "Неровная дорога",                 # 21
    "Скользкая дорога",                # 22
    "Сужение дороги справа",           # 23
    "Дорожные работы",                 # 24
    "Светофор",                        # 25
    "Пешеходный переход",              # 26
    "Дети на дороге",                  # 27
    "Велосипедисты",                   # 28
    "Осторожно: лёд/снег",             # 29
    "Дикие животные",                  # 30
    "Конец всех ограничений",          # 31
    "Движение только прямо",           # 32
    "Движение только прямо или направо",# 33
    "Движение только прямо или налево", # 34
    "Только направо",                  # 35
    "Только налево",                   # 36
    "Прямо или направо",               # 37
    "Движение по кругу",               # 38
    "Конец запрета обгона",            # 39
    "Конец запрета обгона (грузовики)",# 40
    "Обязательное направление прямо",  # 41
    "Прочее",                          # 42
]

IMG_SIZE = 32
NUM_CLASSES = 43


# ═══════════════════════════════════════════════
#  МОДЕЛЬ CNN
# ═══════════════════════════════════════════════
def build_model():
    """Строим сверточную нейронную сеть (CNN)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        # Блок 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Блок 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Блок 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ═══════════════════════════════════════════════
#  ЗАГРУЗКА И АУГМЕНТАЦИЯ ДАННЫХ
# ═══════════════════════════════════════════════
def download_and_prepare_data(log_fn=print):
    """Скачивает GTSRB и подготавливает данные."""
    import urllib.request
    import zipfile

    DATA_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    DATA_DIR = "gtsrb_data"
    ZIP_PATH = os.path.join(DATA_DIR, "train.zip")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(ZIP_PATH):
        log_fn("⬇  Скачиваем GTSRB датасет (~260 MB) …")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        log_fn("✅ Загрузка завершена")

    extract_dir = os.path.join(DATA_DIR, "GTSRB")
    if not os.path.exists(extract_dir):
        log_fn("📦 Распаковываем архив …")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(DATA_DIR)
        log_fn("✅ Распаковка завершена")

    return extract_dir


def load_images_from_dir(data_dir, log_fn=print):
    """Читаем изображения и метки из папок GTSRB."""
    images, labels = [], []
    train_dir = os.path.join(data_dir, "Final_Training", "Images")

    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(train_dir, f"{class_id:05d}")
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    images.append(np.array(img))
                    labels.append(class_id)
                except Exception:
                    pass
        if class_id % 10 == 0:
            log_fn(f"   Загружено классов: {class_id + 1}/{NUM_CLASSES}")

    return np.array(images), np.array(labels)


def generate_synthetic_data(log_fn=print):
    """
    Генерируем синтетические данные для демонстрации
    (когда нет интернета или датасет не скачан).
    """
    log_fn("🔬 Генерируем синтетические данные для демонстрации …")
    np.random.seed(42)
    n_samples = 5000
    X = np.random.randint(0, 256, (n_samples, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    y = np.random.randint(0, NUM_CLASSES, n_samples)
    log_fn(f"✅ Сгенерировано {n_samples} синтетических образцов")
    return X, y


# ═══════════════════════════════════════════════
#  ОБУЧЕНИЕ
# ═══════════════════════════════════════════════
def train_model(use_real_data=True, epochs=10, log_fn=print):
    """Загружает данные и обучает модель."""
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    log_fn("🧠 Строим модель CNN …")
    model = build_model()
    log_fn(f"   Параметров модели: {model.count_params():,}")

    # Данные
    if use_real_data:
        try:
            data_dir = download_and_prepare_data(log_fn)
            X, y = load_images_from_dir(data_dir, log_fn)
            log_fn(f"✅ Загружено {len(X)} изображений")
        except Exception as e:
            log_fn(f"⚠  Не удалось загрузить реальные данные: {e}")
            log_fn("   Переключаемся на синтетические данные …")
            X, y = generate_synthetic_data(log_fn)
    else:
        X, y = generate_synthetic_data(log_fn)

    # Нормализация
    X = X.astype(np.float32) / 255.0
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Аугментация
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
    )
    datagen.fit(X_train)

    log_fn(f"\n🚀 Начинаем обучение ({epochs} эпох) …\n")

    class LogCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_fn(
                f"   Эпоха {epoch+1}/{epochs} | "
                f"loss={logs['loss']:.4f} | acc={logs['accuracy']:.4f} | "
                f"val_acc={logs['val_accuracy']:.4f}"
            )

    model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[LogCallback()],
        verbose=0,
    )

    log_fn("\n✅ Обучение завершено!")
    model.save("gtsrb_model.h5")
    log_fn("💾 Модель сохранена: gtsrb_model.h5")
    return model


# ═══════════════════════════════════════════════
#  ПРЕДСКАЗАНИЕ
# ═══════════════════════════════════════════════
def predict_image(model, image_path):
    """Классифицирует одно изображение, возвращает (class_id, confidence, top5)."""
    img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = [(int(i), float(preds[i])) for i in top5_idx]
    return top5[0][0], top5[0][1], top5


# ═══════════════════════════════════════════════
#  ГРАФИЧЕСКИЙ ИНТЕРФЕЙС (Tkinter)
# ═══════════════════════════════════════════════
class GTSRBApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GTSRB — Распознавание дорожных знаков")
        self.geometry("900x680")
        self.resizable(True, True)
        self.configure(bg="#1a1a2e")
        self.model = None
        self._build_ui()

    # ── UI ──────────────────────────────────────
    def _build_ui(self):
        # Заголовок
        header = tk.Frame(self, bg="#16213e", pady=10)
        header.pack(fill='x')
        tk.Label(header, text="🚦 GTSRB Classifier",
                 font=("Helvetica", 22, "bold"),
                 fg="#e94560", bg="#16213e").pack()
        tk.Label(header, text="Жол қозғалысы белгілерін автоматты тану",
                 font=("Helvetica", 11), fg="#a8a8b3", bg="#16213e").pack()

        # Основной контейнер
        main = tk.Frame(self, bg="#1a1a2e")
        main.pack(fill='both', expand=True, padx=20, pady=10)

        # Левая панель — управление
        left = tk.Frame(main, bg="#16213e", width=280, padx=15, pady=15)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="⚙  Обучение модели",
                 font=("Helvetica", 13, "bold"),
                 fg="#e94560", bg="#16213e").pack(anchor='w', pady=(0, 10))

        # Чекбокс
        self.use_real_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="Использовать реальный датасет\n(~260 MB, нужен интернет)",
                       variable=self.use_real_var, bg="#16213e", fg="#a8a8b3",
                       selectcolor="#0f3460", activebackground="#16213e",
                       font=("Helvetica", 9)).pack(anchor='w')

        # Эпохи
        tk.Label(left, text="Эпохи:", font=("Helvetica", 10),
                 fg="#a8a8b3", bg="#16213e").pack(anchor='w', pady=(10, 2))
        self.epochs_var = tk.IntVar(value=5)
        tk.Scale(left, from_=1, to=30, orient='horizontal',
                 variable=self.epochs_var, bg="#16213e", fg="#e2e2e2",
                 highlightthickness=0, troughcolor="#0f3460").pack(fill='x')

        self.btn_train = tk.Button(
            left, text="▶  Начать обучение",
            command=self._start_training,
            bg="#e94560", fg="white", font=("Helvetica", 11, "bold"),
            relief='flat', cursor='hand2', pady=8)
        self.btn_train.pack(fill='x', pady=(15, 5))

        self.btn_load = tk.Button(
            left, text="📂 Загрузить модель (.h5)",
            command=self._load_model,
            bg="#0f3460", fg="white", font=("Helvetica", 10),
            relief='flat', cursor='hand2', pady=6)
        self.btn_load.pack(fill='x', pady=2)

        # Статус модели
        self.model_status = tk.Label(left, text="⬤  Модель не загружена",
                                      font=("Helvetica", 9), fg="#ff6b6b", bg="#16213e")
        self.model_status.pack(anchor='w', pady=(5, 15))

        # Классификация
        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=5)
        tk.Label(left, text="🔍 Классификация",
                 font=("Helvetica", 13, "bold"),
                 fg="#e94560", bg="#16213e").pack(anchor='w', pady=(10, 5))

        self.btn_open = tk.Button(
            left, text="🖼  Открыть изображение",
            command=self._open_image,
            bg="#0f3460", fg="white", font=("Helvetica", 10),
            relief='flat', cursor='hand2', pady=6)
        self.btn_open.pack(fill='x', pady=2)

        # Лог
        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=(15, 5))
        tk.Label(left, text="📋 Лог", font=("Helvetica", 10, "bold"),
                 fg="#a8a8b3", bg="#16213e").pack(anchor='w')
        self.log_text = tk.Text(left, height=12, bg="#0a0a1a", fg="#a8ffb0",
                                 font=("Courier", 8), relief='flat', wrap='word')
        self.log_text.pack(fill='both', expand=True, pady=(5, 0))
        sb = ttk.Scrollbar(left, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)

        # Правая панель — результаты
        right = tk.Frame(main, bg="#16213e", padx=15, pady=15)
        right.pack(side='right', fill='both', expand=True)

        # Изображение
        self.img_frame = tk.Frame(right, bg="#0f3460", width=300, height=300)
        self.img_frame.pack(pady=(0, 15))
        self.img_frame.pack_propagate(False)
        self.img_label = tk.Label(self.img_frame, text="Загрузите изображение",
                                   bg="#0f3460", fg="#a8a8b3",
                                   font=("Helvetica", 12))
        self.img_label.place(relx=0.5, rely=0.5, anchor='center')

        # Результат
        self.result_var = tk.StringVar(value="Результат появится здесь")
        tk.Label(right, textvariable=self.result_var,
                 font=("Helvetica", 13, "bold"), fg="#e2e2e2",
                 bg="#16213e", wraplength=350).pack()

        self.conf_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.conf_var,
                 font=("Helvetica", 11), fg="#a8a8b3", bg="#16213e").pack()

        # Топ-5
        tk.Label(right, text="Топ-5 предсказаний:",
                 font=("Helvetica", 10, "bold"), fg="#e94560", bg="#16213e").pack(anchor='w', pady=(15, 5))

        self.top5_frame = tk.Frame(right, bg="#16213e")
        self.top5_frame.pack(fill='x')

        # Прогресс бар обучения
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(right, variable=self.progress_var,
                                         maximum=100, length=350)
        self.progress.pack(pady=(20, 0))
        self.progress_label = tk.Label(right, text="", fg="#a8a8b3", bg="#16213e",
                                        font=("Helvetica", 9))
        self.progress_label.pack()

    # ── Лог ─────────────────────────────────────
    def _log(self, msg):
        self.log_text.insert('end', msg + "\n")
        self.log_text.see('end')

    # ── Обучение ─────────────────────────────────
    def _start_training(self):
        if self.model is not None:
            if not messagebox.askyesno("Переобучить?",
                                       "Модель уже загружена. Обучить заново?"):
                return
        self.btn_train.config(state='disabled', text="⏳ Обучение …")
        self.progress_var.set(0)
        use_real = self.use_real_var.get()
        epochs = self.epochs_var.get()
        thread = threading.Thread(
            target=self._train_thread, args=(use_real, epochs), daemon=True)
        thread.start()

    def _train_thread(self, use_real, epochs):
        try:
            def log_and_progress(msg):
                self.after(0, self._log, msg)
                if "Эпоха" in msg:
                    epoch_num = int(msg.split()[1].split("/")[0])
                    pct = (epoch_num / epochs) * 100
                    self.after(0, self.progress_var.set, pct)
                    self.after(0, self.progress_label.config,
                               {"text": f"Прогресс: {pct:.0f}%"})

            self.model = train_model(
                use_real_data=use_real,
                epochs=epochs,
                log_fn=log_and_progress
            )
            self.after(0, self._training_done)
        except Exception as e:
            self.after(0, self._log, f"❌ Ошибка: {e}")
            self.after(0, self.btn_train.config,
                       {"state": "normal", "text": "▶  Начать обучение"})

    def _training_done(self):
        self.btn_train.config(state='normal', text="▶  Начать обучение")
        self.model_status.config(text="⬤  Модель готова", fg="#a8ff78")
        self.progress_var.set(100)
        self._log("🎉 Готово к классификации!")

    # ── Загрузка модели ──────────────────────────
    def _load_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("HDF5 model", "*.h5"), ("All files", "*.*")])
        if not path:
            return
        try:
            import tensorflow as tf
            self._log(f"📂 Загружаем {os.path.basename(path)} …")
            self.model = tf.keras.models.load_model(path)
            self.model_status.config(text="⬤  Модель загружена", fg="#a8ff78")
            self._log("✅ Модель успешно загружена!")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    # ── Классификация ────────────────────────────
    def _open_image(self):
        if self.model is None:
            messagebox.showwarning("Нет модели",
                                   "Сначала обучите или загрузите модель!")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.ppm *.bmp"),
                       ("All files", "*.*")])
        if not path:
            return
        self._show_image(path)
        self._classify(path)

    def _show_image(self, path):
        img = Image.open(path).convert('RGB').resize((280, 280))
        ph = ImageTk.PhotoImage(img)
        self.img_label.config(image=ph, text="")
        self.img_label.image = ph

    def _classify(self, path):
        try:
            cls_id, conf, top5 = predict_image(self.model, path)
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Класс {cls_id}"
            self.result_var.set(f"✅ {name}")
            self.conf_var.set(f"Уверенность: {conf*100:.1f}%")

            # Топ-5
            for w in self.top5_frame.winfo_children():
                w.destroy()
            for rank, (cid, prob) in enumerate(top5):
                n = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"Класс {cid}"
                row = tk.Frame(self.top5_frame, bg="#16213e")
                row.pack(fill='x', pady=1)
                color = "#e94560" if rank == 0 else "#a8a8b3"
                tk.Label(row, text=f"{rank+1}. {n[:35]}",
                         fg=color, bg="#16213e",
                         font=("Helvetica", 9 if rank > 0 else 10,
                               "bold" if rank == 0 else "normal"),
                         width=38, anchor='w').pack(side='left')
                tk.Label(row, text=f"{prob*100:.1f}%",
                         fg=color, bg="#16213e",
                         font=("Helvetica", 9)).pack(side='right')
        except Exception as e:
            messagebox.showerror("Ошибка классификации", str(e))


# ═══════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    # Скрыть консоль на Windows (при запуске из exe)
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(
                ctypes.windll.kernel32.GetConsoleWindow(), 0)
        except Exception:
            pass

    app = GTSRBApp()
    app.mainloop()
