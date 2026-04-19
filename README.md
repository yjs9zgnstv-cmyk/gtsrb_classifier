# GTSRB Classifier — Жол қозғалысы белгілерін автоматты тану

**Курстық жоба** | «Нейрондық желілерді бағдарламалау» пәні  
**Студент:** Талап Е. | **Топ:** ИС-24-01 | **Жетекші:** Бекенова А.С.  
**Университет:** Жәңгір хан атындағы БҚАТУ, Орал 2026

---

## Жоба туралы

CNN (Convolutional Neural Network) нейрон желісі арқылы GTSRB деректер жинағындағы 43 санатты жол белгілерін автоматты тану жүйесі. Графикалық интерфейс (Tkinter) арқылы жұмыс істейді.

## Мүмкіндіктері

- 43 санатты жол белгілерін тану (GTSRB деректер жинағы)
- CNN модельді оқыту (нағыз GTSRB немесе синтетикалық деректермен)
- Оқытылған моделді (.h5) сақтау және жүктеу
- Top-5 болжам нәтижелерін пайыздармен көрсету
- Графикалық интерфейс (Tkinter GUI)

## Технологиялар

| Кітапхана | Нұсқасы | Мақсаты |
|-----------|---------|---------|
| TensorFlow | 2.19+ | CNN моделі |
| NumPy | 1.24+ | Массивтер |
| Pillow | 10+ | Суреттер |
| scikit-learn | 1.3+ | train/test split |
| Tkinter | Стандарт | GUI |

## Орнату

```bash
pip install tensorflow numpy pillow scikit-learn matplotlib
```

## Іске қосу

```bash
python gtsrb_classifier.py
```

## CNN Архитектурасы

```
Input (32×32×3)
→ Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout(0.25)
→ Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout(0.25)
→ Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
→ Flatten → Dense(512) + BatchNorm + Dropout(0.5)
→ Dense(43, softmax)

Жалпы параметр: 1 213 515
```

## Нәтижелер

- Нағыз GTSRB деректерімен: **85-95% дәлдік**
- Top-5 Accuracy: **98%+**
- Болжам жасау уақыты: **~0.5 сек**

## Файлдар құрылымы

```
gtsrb_classifier/
├── gtsrb_classifier.py   # Негізгі файл (CNN + GUI)
├── requirements.txt      # Тәуелділіктер
├── README.md             # Осы файл
└── gtsrb_model.h5        # Оқытылған модель (оқытқаннан кейін)
```

## EXE / APP жасау (PyInstaller)

**Windows:**
```bash
pyinstaller --onefile --windowed --name GTSRB_Classifier gtsrb_classifier.py
```

**macOS:**
```bash
pyinstaller --onefile --windowed --name GTSRB_Classifier gtsrb_classifier.py
xattr -cr dist/GTSRB_Classifier.app
```

## GTSRB деректер жинағы туралы

- **Сурет саны:** 50 000+
- **Санат саны:** 43
- **Көз:** [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Өлшем:** ~260 МБ (бағдарлама автоматты жүктейді)
